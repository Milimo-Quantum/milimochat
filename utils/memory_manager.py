import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from typing import List, Dict, Optional, Any, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from collections import deque
import hashlib
import math
import numpy as np
import base64
import json
import pandas as pd
import io
from pathlib import Path
import numpy as np
import uuid

from services.database import Database
from utils.ollama_client import OllamaClient
from utils.file_processor import FileProcessor
from services.context_integrator import ContextIntegrator
from config import (
    MAX_SHORT_TERM_MEMORY,
    MESSAGE_TYPES,
    MODEL_PARAMETERS,
    DEFAULT_MODEL_PARAMS,
    EMBED_MODEL
)

class MemoryManager:
    def __init__(self, session_id: str):
        """Initialize memory manager with enhanced context integration."""
        if not session_id:
            raise ValueError("Session ID is required")
            
        self.session_id = session_id
        self.db = Database()
        self.ollama_client = OllamaClient()
        self.file_processor = FileProcessor()
        self.context_integrator = ContextIntegrator(
            max_context_length=MODEL_PARAMETERS["context_length"]
        )
        self._init_db()

    def _init_db(self):
        """Initialize database schema for memory storage"""
        with self.db.conn:
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 1,
                    embedding BLOB,
                    session_id TEXT
                )
            """)
        self.short_term_memory = deque(maxlen=MAX_SHORT_TERM_MEMORY)
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        """Initialize memory with enhanced vector support."""
        try:
            messages = self.db.get_messages(
                self.session_id,
                limit=MAX_SHORT_TERM_MEMORY
            )
            
            valid_messages = []
            
            for msg in messages:
                try:
                    if not isinstance(msg, dict):
                        continue

                    if not all(key in msg for key in ['role', 'content', 'timestamp']):
                        continue
                    
                    if msg['role'] not in [MESSAGE_TYPES["USER"], MESSAGE_TYPES["ASSISTANT"]]:
                        continue

                    # Process any file attachments in metadata
                    if msg.get("metadata") and msg["metadata"].get("file_data"):
                        self._process_file_metadata(msg)

                    valid_messages.append(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue

            sorted_messages = sorted(valid_messages, key=lambda x: x['timestamp'])
            
            self.short_term_memory.clear()
            for msg in sorted_messages:
                self.short_term_memory.append(msg)
        
        except Exception as e:
            logger.error(f"Error initializing memory: {str(e)}")
            self.short_term_memory.clear()

    def _process_file_metadata(self, msg: Dict[str, Any]) -> None:
        """Process file metadata in message."""
        try:
            file_data = msg["metadata"]["file_data"]
            if not isinstance(file_data, dict):
                return

            # Update file metadata with additional context
            file_data['metadata'] = file_data.get('metadata', {})
            file_data['metadata'].update({
                'processed_at': datetime.now().isoformat(),
                'session_id': self.session_id,
                'message_id': msg.get('id')
            })

            msg["metadata"]["file_data"] = file_data

        except Exception as e:
            logger.error(f"Error processing file metadata: {str(e)}")

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message with enhanced context tracking."""
        try:
            if not content or not role:
                return

            # Initialize metadata
            if metadata is None:
                metadata = {}
            metadata['session_id'] = self.session_id
            current_time = datetime.now().isoformat()
            
            # Process file attachments if present
            if metadata.get("file_data"):
                await self._process_file_content(metadata["file_data"], current_time)

            try:
                # Generate embedding
                embedding = await self.ollama_client.generate_embeddings(content)
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
                # Generate memory key
                memory_key = hashlib.sha256(
                    f"{content}-{current_time}".encode()
                ).hexdigest()
                
                # Format memory for storage
                formatted_memory = self.context_integrator.format_memory_for_context({
                    'content': content,
                    'created_at': current_time,
                    'role': role,
                    'embedding': embedding_list,
                    'key': memory_key
                })

                # Save vector with metadata
                vector_metadata = {
                    'type': 'message',
                    'role': role,
                    'timestamp': current_time,
                    'memory_key': memory_key,
                    'session_id': self.session_id
                }
                
                # Add any relevant file or context information
                if metadata.get('file_data'):
                    vector_metadata['file_type'] = metadata['file_data'].get('type')
                    vector_metadata['file_name'] = metadata['file_data'].get('metadata', {}).get('filename')
                
                if metadata.get('context_used'):
                    vector_metadata['context_used'] = True
                    vector_metadata['context_source'] = metadata.get('context_source')

                # Save to vector store
                vector_id = self.db.save_vector(
                    embedding_list,
                    vector_metadata
                )
                
                metadata['vector_id'] = vector_id
                metadata['memory_key'] = memory_key
                
            except Exception as e:
                logger.error(f"Error in embedding process: {str(e)}")

            # Prepare message for storage
            message = {
                'role': role,
                'content': content,
                'timestamp': current_time,
                'metadata': metadata,
                'id': None  # Will be set after database save
            }

            # Save to database
            message_id = self.db.save_message(
                self.session_id,
                role,
                content,
                metadata
            )

            if message_id < 0:
                raise Exception("Failed to save message")

            message['id'] = message_id
            
            # Update short-term memory with deduplication
            self._add_to_short_term_memory(message)

            # Store in long-term memory if assistant message or important user message
            if role == MESSAGE_TYPES["ASSISTANT"] or metadata.get('store_long_term', False):
                await self._store_in_long_term_memory(
                    formatted_memory,
                    {
                        'timestamp': current_time,
                        'role': role,
                        'vector_id': metadata.get('vector_id'),
                        'memory_key': memory_key,
                        'session_id': self.session_id,
                        'context_info': metadata.get('context_metadata', {}),
                        'file_info': metadata.get('file_data', {}).get('metadata', {})
                    }
                )

        except Exception as e:
            print(f"Error adding message: {str(e)}")
            raise

    def _add_to_short_term_memory(self, message: Dict[str, Any]) -> None:
        """Add message to short-term memory with deduplication."""
        try:
            # Check for duplicates
            for existing in self.short_term_memory:
                if (existing.get('content') == message['content'] and 
                    existing.get('role') == message['role']):
                    # Update existing message metadata
                    existing['metadata'].update(message.get('metadata', {}))
                    return

            self.short_term_memory.append(message)

        except Exception as e:
            logger.error(f"Error adding to short-term memory: {str(e)}")
            # Ensure message is added even if deduplication fails
            self.short_term_memory.append(message)

    async def _store_in_long_term_memory(
    self,
    memory: Dict[str, Any],
    metadata: Dict[str, Any]
) -> None:
        """Store memory in long-term storage with enhanced metadata."""
        try:
            # Ensure memory has required fields
            if not isinstance(memory, dict):
                raise ValueError("Memory must be a dictionary")

            if 'content' not in memory:
                raise ValueError("Memory must have 'content'")

            # Generate embedding if not provided
            if 'embedding' not in memory:
                try:
                    embedding = await self.ollama_client.generate_embeddings(memory['content'])
                    memory['embedding'] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                except Exception as e:
                    print(f"Error generating embedding: {str(e)}")
                    memory['embedding'] = None

            # Combine metadata with vector relationships
            enhanced_metadata = {
                **metadata,
                'vector_relationships': [],
                'memory_type': 'long_term',
                'storage_time': datetime.now().isoformat()
            }

            # Add vector relationships if embedding exists
            if memory.get('embedding') is not None:
                try:
                    related_vectors = self.db.get_similar_vectors(
                        memory['embedding'],
                        top_k=5,
                        threshold=0.7
                    )
                    enhanced_metadata['vector_relationships'] = [
                        {
                            'vector_id': v['id'],
                            'similarity': v['similarity'],
                            'metadata': v['metadata']
                        }
                        for v in related_vectors
                    ]
                except Exception as e:
                    print(f"Error finding related vectors: {str(e)}")

            # Save to database with proper error handling
            try:
                self.db.save_memory(
                    memory.get('key', str(uuid.uuid4())),
                    memory['content'],
                    memory.get('embedding'),  # May be None
                    enhanced_metadata
                )
            except Exception as e:
                print(f"Database save error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error storing in long-term memory: {str(e)}")
            raise

    async def _process_file_content(
        self,
        file_data: Dict[str, Any],
        timestamp: str
    ) -> None:
        """Process file content with enhanced context handling."""
        try:
            if not isinstance(file_data, dict):
                return

            # Format the file memory
            file_memory = self.context_integrator.format_memory_for_context(
                {
                    'content': f"File processed: {file_data.get('metadata', {}).get('filename', 'unknown')}",
                    'created_at': timestamp,
                    'file_data': file_data
                },
                memory_type="file"
            )

            # Store file content in long-term memory
            enhanced_metadata = {
                'file_type': file_data['type'],
                'timestamp': timestamp,
                'session_id': self.session_id,
                'file_metadata': file_data.get('metadata', {}),
                'processing_info': {
                    'chunks': len(file_data.get('chunks', [])),
                    'embeddings': len(file_data.get('chunk_embeddings', [])),
                    'has_analysis': bool(file_data.get('initial_analysis'))
                }
            }

            await self._store_in_long_term_memory(
                file_memory,
                enhanced_metadata
            )

            # Store individual chunks if available
            if file_data.get('chunks') and file_data.get('chunk_embeddings'):
                for idx, (chunk, embedding) in enumerate(zip(
                    file_data['chunks'],
                    file_data['chunk_embeddings']
                )):
                    chunk_memory = self.context_integrator.format_memory_for_context(
                        {
                            'content': chunk,
                            'created_at': timestamp,
                            'embedding': embedding
                        },
                        memory_type="file_chunk"
                    )
                    
                    chunk_metadata = {
                        'file_type': file_data['type'],
                        'chunk_index': idx,
                        'timestamp': timestamp,
                        'session_id': self.session_id,
                        'parent_file': file_data.get('metadata', {}).get('filename'),
                        'chunk_info': {
                            'total_chunks': len(file_data['chunks']),
                            'position': f"{idx + 1}/{len(file_data['chunks'])}"
                        }
                    }

                    await self._store_in_long_term_memory(
                        chunk_memory,
                        chunk_metadata
                    )

        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")

    async def get_relevant_context(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> str:
        """Get relevant conversation context from long-term memory."""
        try:
            # Generate query embedding using enhanced model
            query_embedding = await self.ollama_client.generate_embeddings(
                f"Query: {query} [Memory Priority: Long-Term]",
                model=EMBED_MODEL
            )
            embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            logger.info(f"Query embedding generated: {embedding_list[:10]}...") # Log first 10 elements of embedding
            logger.info(f"Session ID for memory retrieval: {self.session_id}") # Log session ID
            # Search entire long-term memory database
            similar_memories = self.db.get_memory_by_similarity(
                embedding_list,
                self.session_id,
                top_k=100,  # Search entire memory store
                min_score=0.0  # Include all memories
            )
            
            # Enhanced relevance weighting
            now = datetime.now()
            for memory in similar_memories:
                # Calculate memory age
                mem_time = datetime.fromisoformat(memory['created_at'])
                hours_old = (now - mem_time).total_seconds() / 3600
                
                # Calculate freshness boost from recent accesses
                access_count = len(memory.get('access_times', []))
                last_access_hours = (now - datetime.fromisoformat(
                    memory.get('last_accessed', now.isoformat())
                )).total_seconds() / 3600
                
                # Calculate boosted similarity score
                freshness_boost = 1 + (access_count * 0.05) + (1 / (last_access_hours + 1))
                type_boost = 1.2 if memory.get('metadata', {}).get('memory_type') == 'long_term' else 1
                memory['boosted_similarity'] = memory['similarity'] * freshness_boost * type_boost
                
                # Apply gentle age decay
                memory['boosted_similarity'] *= max(0.7, 1 - (hours_old / 8760))  # 1 year half-life

            # Sort by boosted similarity
            similar_memories.sort(key=lambda x: -x['boosted_similarity'])
            
            # Format memories with full context
            formatted_memories = [
                self.context_integrator.format_memory_for_context(m, "long_term")
                for m in similar_memories if m
            ]
            
            # Directly use long-term memories without short-term merge
            # Merge context using the existing merge_context method
            merged_context, context_metadata = self.context_integrator.merge_context(
                short_term_memories=[],  # Empty since we're only using long-term
                long_term_memories=formatted_memories,
                current_context=query,
                max_memories=len(formatted_memories)
            )
            
            # Track context usage
            context_usage = {
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store context usage in database
            self.db.save_memory(
                f"context_usage_{datetime.now().isoformat()}",
                json.dumps(context_usage),
                None,  # No embedding needed
                {
                    'type': 'context_usage',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Format for prompt
            context_string = self.context_integrator.format_context_for_prompt(
                merged_context,
                context_metadata  # Use the metadata from merge_context
            )
            
            # Validate and potentially truncate context
            validated_context, validation_info = self.context_integrator.validate_context(
                context_string
            )
            
            logger.info(f"similar_memories length: {len(similar_memories)}")
            if similar_memories:
                logger.info(f"First similar memory: {similar_memories[0].get('content')[:100]}...")
                logger.info(f"First similar memory similarity: {similar_memories[0].get('similarity')}")
                for i, mem in enumerate(similar_memories[:3]): # Log content and similarity of top 3 memories
                    logger.info(f"Top {i+1} similar memory content: {mem.get('content')[:100]}...")
                    logger.info(f"Top {i+1} similar memory similarity: {mem.get('similarity')}")
            logger.info(f"formatted_memories length: {len(formatted_memories)}")
            logger.info(f"merged_context length: {len(merged_context)}")
            logger.info(f"context_metadata: {context_metadata}")
            return validated_context, context_usage

        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return "", {}

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory statistics with enhanced tracking."""
        try:
            memory_stats = self.db.get_memory_stats(self.session_id)
            
            # Get additional memory metrics
            short_term_count = len(self.short_term_memory)
            short_term_tokens = sum(
                len(msg['content'].split()) * 1.3  # Rough token estimate
                for msg in self.short_term_memory
            )
            
            context_usages = [
                json.loads(mem['content'])
                for mem in self.db.get_all_long_term_memories(self.session_id)
                if mem.get('metadata', {}).get('type') == 'context_usage'
            ]

            context_stats = {
                'total_uses': len(context_usages),
                'avg_memories_used': np.mean([u['total_memories_used'] for u in context_usages]) if context_usages else 0,
                'avg_relevance': np.mean([
                    score for u in context_usages 
                    for score in u.get('relevance_scores', [])
                ]) if context_usages else 0
            }

            return {
                'total_memories': memory_stats['total_memories'],
                'total_vectors': memory_stats['total_vectors'],
                'active_memories': short_term_count,
                'memory_usage_bytes': memory_stats['memory_usage_bytes'],
                'short_term_tokens': int(short_term_tokens),
                'context_usage': context_stats,
                'file_memories': memory_stats['total_files'],
                'vector_relationships': len(memory_stats.get('vector_relationships', [])),
                'last_cleanup': memory_stats.get('last_cleanup')
            }

        except Exception as e:
            logger.error(f"Error getting memory summary: {str(e)}")
            return {
                'total_memories': 0,
                'total_vectors': 0,
                'active_memories': 0,
                'memory_usage_bytes': 0,
                'short_term_tokens': 0,
                'context_usage': {'total_uses': 0, 'avg_memories_used': 0, 'avg_relevance': 0},
                'file_memories': 0,
                'vector_relationships': 0,
                'last_cleanup': None
            }

    async def get_memory_by_key(self, memory_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific memory by key with enhanced metadata."""
        try:
            memories = await self.get_all_long_term_memories()
            for memory in memories:
                if memory.get('key') == memory_key:
                    return memory
            return None
        except Exception as e:
            logger.error(f"Error retrieving memory by key: {str(e)}")
            return None

    async def get_memory_by_similarity(
        self,
        query: str,
        limit: Optional[int] = None,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get memories based on semantic similarity with enhanced metadata."""
        try:
            # Ensure generate_embeddings returns a coroutine 
            embedding = await self.ollama_client.generate_embeddings(query)
            query_embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            # Get similar memories from database
            similar_memories = self.db.get_memory_by_similarity(
                query_embedding,
                self.session_id,
                top_k=limit if limit else 5
            )
            
            # Filter by similarity threshold
            filtered_memories = [
                memory for memory in similar_memories
                if memory.get('similarity', 0) >= threshold
            ]
            
            # Enrich memories with additional metadata
            for memory in filtered_memories:
                # Add timestamp if missing
                if not memory.get('timestamp'):
                    memory['timestamp'] = memory.get('created_at', datetime.now().isoformat())
                
                # Add metadata if missing
                if not memory.get('metadata'):
                    memory['metadata'] = {
                        'session_id': self.session_id,
                        'type': 'long_term_memory',
                        'retrieved_at': datetime.now().isoformat()
                    }
                
                # Update last access time
                memory['last_accessed'] = datetime.now().isoformat()
                
                # Update access tracking in database
                try:
                    memory_copy = memory.copy()
                    if 'access_times' not in memory_copy:
                        memory_copy['access_times'] = []
                    memory_copy['access_times'].append(memory_copy['last_accessed'])

                    self.db.save_memory(
                        memory['key'],
                        memory['content'],
                        None,  # No need to regenerate embedding
                        memory_copy.get('metadata', {})
                    )
                except Exception as e:
                    print(f"Error updating memory access time: {str(e)}")
            
            return filtered_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories by similarity: {str(e)}")
            return []

    async def get_all_long_term_memories(self) -> List[Dict[str, Any]]:
        """Get all long-term memories with complete metadata."""
        try:
            if not self.session_id:
                print("No session_id available")
                return []
                
            print(f"Retrieving memories from database")
            
            memories = self.db.get_all_long_term_memories(self.session_id)
            
            print(f"Retrieved {len(memories)} memories from database")
            
            if not memories:
                return []
                
            formatted_memories = []
            for memory in memories:
                if not isinstance(memory, dict) or not memory.get('content'):
                    continue
                
                # Ensure key has session prefix
                if 'key' in memory and not memory['key'].startswith(self.session_id):
                    memory['key'] = f"{self.session_id}_{memory['key']}"
                
                # Ensure metadata has session info
                if 'metadata' not in memory:
                    memory['metadata'] = {}
                memory['metadata']['session_id'] = self.session_id
                memory['metadata']['persistent_id'] = self.session_id
                
                formatted_memories.append(memory)
            
            # Sort by created_at timestamp
            formatted_memories.sort(
                key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                reverse=True
            )
            
            return formatted_memories

        except Exception as e:
            logger.error(f"Error retrieving long-term memories: {str(e)}")
            return []

    def get_recent_messages(
        self,
        limit: Optional[int] = None,
        message_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages with improved message validation and type filtering."""
        try:
            messages = []
            
            if not hasattr(self, 'short_term_memory') or self.short_term_memory is None:
                self.short_term_memory = deque(maxlen=MAX_SHORT_TERM_MEMORY)
                return []
            
            # Default to user and assistant messages if no types specified
            if not message_types:
                message_types = [MESSAGE_TYPES["USER"], MESSAGE_TYPES["ASSISTANT"]]
            
            for msg in self.short_term_memory:
                if not isinstance(msg, dict):
                    continue
                    
                role = msg.get('role')
                if not role or role not in message_types:
                    continue
                    
                if not all(msg.get(key) for key in ['role', 'content', 'timestamp']):
                    continue
                    
                messages.append(msg)

            if messages:
                messages.sort(key=lambda x: x.get('timestamp', ''))
            
            if limit and limit > 0 and messages:
                messages = messages[-limit:]
                
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving recent messages: {str(e)}")
            return []

    def clear_short_term_memory(self) -> None:
        """Clear short-term memory with proper cleanup."""
        try:
            self.short_term_memory.clear()
            self.db.delete_messages(self.session_id)
        except Exception as e:
            logger.error(f"Error clearing short-term memory: {str(e)}")
            raise

    def delete_message(self, message_id: int) -> None:
        """Delete a specific message with vector cleanup."""
        try:
            # Get message to check for vector_id
            messages = [msg for msg in self.short_term_memory if msg.get('id') == message_id]
            if messages and messages[0].get('metadata', {}).get('vector_id'):
                vector_id = messages[0]['metadata']['vector_id']
                try:
                    self.db.delete_vector(vector_id)
                except Exception as ve:
                    print(f"Error deleting vector: {str(ve)}")

            # Delete message from database
            self.db.delete_message(message_id)
            
            # Update short-term memory
            self.short_term_memory = deque(
                [msg for msg in self.short_term_memory if msg.get('id') != message_id],
                maxlen=MAX_SHORT_TERM_MEMORY
            )

        except Exception as e:
            logger.error(f"Failed to delete message: {str(e)}")

    def clear_all_session_memory(self) -> None:
        """Clear all memory for the session with complete cleanup."""
        try:
            # Clear short-term memory
            self.clear_short_term_memory()
            
            # Clear long-term memories and vectors
            self.db.clear_all_session_memory(self.session_id)
            
            # Clear context tracking
            self.context_integrator.clear_context_tracking()
            
        except Exception as e:
            logger.error(f"Error clearing session memory: {str(e)}")
            raise

    def clean_old_memories(self, days: int = 30) -> None:
        """Clean up old memory entries with proper tracking."""
        try:
            cleanup_time = datetime.now()
            cleanup_stats = {
                'cleaned_memories': 0,
                'cleaned_vectors': 0,
                'cleaned_files': 0,
                'timestamp': cleanup_time.isoformat()
            }

            # Clean memories older than specified days
            self.db.clean_old_memories(days)
            
            # Store cleanup record
            self.db.save_memory(
                f"cleanup_{cleanup_time.isoformat()}",
                json.dumps(cleanup_stats),
                None,
                {
                    'type': 'cleanup_record',
                    'session_id': self.session_id,
                    'days_threshold': days,
                    'timestamp': cleanup_time.isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error cleaning old memories: {str(e)}")