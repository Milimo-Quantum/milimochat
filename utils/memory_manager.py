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
    ) -> Tuple[str, Dict[str, Any]]:
        """Get relevant conversation context from long-term memory with comprehensive search strategy."""
        try:
            import re
            # Start timing for analytics
            start_time = datetime.now()
            
            # Log the query request
            logger.info(f"Context request: '{query[:50]}...' for session {self.session_id}")
            
            # 1. Generate query embedding with specialized prompt to improve search quality
            enhanced_query = f"Query: {query} [Find related memories, critical user information, preferences and history]"
            query_embedding = await self.ollama_client.generate_embeddings(
                enhanced_query,
                model=EMBED_MODEL
            )
            embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            logger.info(f"Query embedding generated with shape {len(embedding_list)}")
            
            # 2. Multi-strategy search approach
            
            # 2.1 First strategy: Vector similarity search with high depth
            similarity_search_start = datetime.now()
            vector_memories = self.db.get_memory_by_similarity(
                embedding_list,
                self.session_id,
                top_k=200,  # Significantly increased for thorough search
                min_score=0.0  # Include all memories for post-processing filtering
            )
            similarity_search_time = (datetime.now() - similarity_search_start).total_seconds() * 1000
            logger.info(f"Vector search retrieved {len(vector_memories)} memories in {similarity_search_time:.2f}ms")
            
            # 2.2 Second strategy: Exact text matching for critical patterns
            text_search_start = datetime.now()
            
            # Extract key terms for exact matching
            def extract_key_terms(text):
                # Extract proper nouns (capitalized words)
                proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
                
                # Extract potential entities from "what is" questions
                what_is_entities = re.findall(r'what (?:is|are) (?:my|the|a|an)? ([a-zA-Z0-9 ]+)', text.lower())
                
                # Extract potential key nouns
                key_nouns = re.findall(r'\b(?:name|email|phone|address|preference|favorite|birthday|age)\b', text.lower())
                
                return proper_nouns + what_is_entities + key_nouns
            
            key_terms = extract_key_terms(query)
            exact_match_memories = []
            
            # Only perform exact match search if we have meaningful terms
            if key_terms:
                logger.info(f"Extracted key terms for exact matching: {key_terms}")
                
                # Get all long-term memories
                all_memories = await self.get_all_long_term_memories()
                
                # Look for exact matches with each term
                for term in key_terms:
                    if len(term) < 3:  # Skip very short terms
                        continue
                        
                    term_lower = term.lower()
                    for memory in all_memories:
                        content = memory.get('content', '').lower()
                        
                        # Check if memory contains the term
                        if term_lower in content:
                            # Special boost for name-related queries
                            if 'name' in query.lower() and 'name' in content and term_lower in content:
                                memory['exact_match'] = True
                                memory['priority_boost'] = 5.0
                                memory['match_term'] = term
                                exact_match_memories.append(memory)
                                logger.info(f"Found exact name match: '{term}' in '{content[:50]}...'")
                            else:
                                memory['exact_match'] = True
                                memory['priority_boost'] = 2.0
                                memory['match_term'] = term
                                exact_match_memories.append(memory)
            
            text_search_time = (datetime.now() - text_search_start).total_seconds() * 1000
            logger.info(f"Text search found {len(exact_match_memories)} exact matches in {text_search_time:.2f}ms")
            
            # 2.3 Third strategy: Get recent messages as potential context
            recent_search_start = datetime.now()
            recent_messages = self.get_recent_messages(5)  # Last 5 messages
            recent_memories = []
            
            for msg in recent_messages:
                # Add as a memory with appropriate metadata
                recent_memories.append({
                    'content': msg.get('content', ''),
                    'created_at': msg.get('timestamp', datetime.now().isoformat()),
                    'recency_score': 0.9,  # High recency score
                    'source': 'recent_message'
                })
            
            recent_search_time = (datetime.now() - recent_search_start).total_seconds() * 1000
            logger.info(f"Recent message search found {len(recent_memories)} messages in {recent_search_time:.2f}ms")
            
            # 3. Combine and deduplicate memories from all strategies
            combined_start = datetime.now()
            
            # Merge all memory sources, prioritizing exact matches
            all_memories = exact_match_memories + vector_memories + recent_memories
            
            # Deduplicate memories using content-based comparison
            seen_contents = set()
            unique_memories = []
            
            for memory in all_memories:
                content = memory.get('content', '').strip()
                content_hash = hash(content.lower())
                
                if content and content_hash not in seen_contents:
                    unique_memories.append(memory)
                    seen_contents.add(content_hash)
            
            # 4. Enhanced scoring with multi-factor prioritization
            scored_memories = []
            now = datetime.now()
            
            for memory in unique_memories:
                # Start with base similarity score (if available)
                base_score = memory.get('similarity', 0.5)  # Default if no similarity score
                
                # Apply boosting factors
                
                # Factor 1: Exact match boost
                if memory.get('exact_match', False):
                    base_score *= memory.get('priority_boost', 2.0)
                
                # Factor 2: Recency boost
                try:
                    mem_time = datetime.fromisoformat(memory['created_at'])
                    hours_old = max(0.1, (now - mem_time).total_seconds() / 3600)
                    
                    # Logarithmic decay for recency (gentler than exponential)
                    recency_score = 1.0 / (1.0 + math.log10(hours_old / 24 + 1))
                except (ValueError, KeyError):
                    recency_score = 0.5
                
                # Factor 3: Access history boost
                access_count = len(memory.get('access_times', []))
                access_boost = min(1.5, 1.0 + (access_count * 0.1))  # Cap at 1.5x
                
                # Factor 4: Source-specific adjustments
                source_boost = 1.0
                memory_source = memory.get('source', memory.get('type', 'unknown'))
                
                if memory_source == 'message' and memory.get('role') == 'user':
                    source_boost = 1.3  # Boost user messages
                elif memory_source == 'recent_message':
                    source_boost = 1.2  # Boost recent conversation
                    
                # Calculate final score with weighted factors
                # 60% similarity, 20% recency, 10% access history, 10% source
                final_score = (
                    base_score * 0.6 + 
                    recency_score * 0.2 +
                    access_boost * 0.1 +
                    source_boost * 0.1
                )
                
                # Special case: explicitly named matches always get highest priority
                if memory.get('exact_match') and 'name' in query.lower() and 'name' in memory.get('content', '').lower():
                    final_score = 10.0  # Ensure it's at the top
                
                memory['final_score'] = final_score
                scored_memories.append(memory)
            
            # Sort by final score
            scored_memories.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            # Take top memories based on limit
            top_limit = limit or 20  # Default to 20 if no limit specified
            top_memories = scored_memories[:top_limit]
            
            # Format memories for context integration
            formatted_memories = []
            for memory in top_memories:
                formatted = self.context_integrator.format_memory_for_context(
                    memory, 
                    "long_term" if memory.get('source') != 'recent_message' else "short_term"
                )
                if formatted:
                    formatted_memories.append(formatted)
            
            combined_time = (datetime.now() - combined_start).total_seconds() * 1000
            logger.info(f"Memory combining, scoring and formatting took {combined_time:.2f}ms")
            
            # 5. Merge context with optimized params for this type of search
            merge_start = datetime.now()
            merged_context, context_metadata = self.context_integrator.merge_context(
                short_term_memories=[],  # Already included in our comprehensive search
                long_term_memories=formatted_memories,
                current_context=query,
                max_memories=top_limit  # Use our adjusted limit
            )
            
            # Format for prompt
            context_string = self.context_integrator.format_context_for_prompt(
                merged_context,
                context_metadata
            )
            
            # Validate and potentially truncate context
            validated_context, validation_info = self.context_integrator.validate_context(
                context_string,
                max_length=MODEL_PARAMETERS["context_length"] - 1000  # Leave room for the prompt and query
            )
            
            merge_time = (datetime.now() - merge_start).total_seconds() * 1000
            logger.info(f"Context merging and formatting took {merge_time:.2f}ms")
            
            # 6. Track usage and update access statistics
            for memory in top_memories:
                memory_key = memory.get('key')
                if memory_key:
                    try:
                        # Update access timestamp and count
                        mem_db = await self.get_memory_by_key(memory_key)
                        if mem_db:
                            access_times = mem_db.get('access_times', [])
                            access_times.append(datetime.now().isoformat())
                            
                            # Update memory with new access info
                            self.db.save_memory(
                                memory_key,
                                memory.get('content', ''),
                                None,  # Don't change embedding
                                {
                                    **mem_db.get('metadata', {}),
                                    'access_times': access_times,
                                    'last_accessed': datetime.now().isoformat(),
                                    'access_count': len(access_times)
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error updating memory access stats: {str(e)}")
            
            # Prepare context usage stats for return
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            context_usage = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'total_memories_searched': len(all_memories),
                'unique_memories': len(unique_memories),
                'memories_used': len(merged_context),
                'exact_matches': len(exact_match_memories),
                'vector_matches': len(vector_memories),
                'processing_time_ms': total_time,
                'search_times': {
                    'vector_search_ms': similarity_search_time,
                    'text_search_ms': text_search_time,
                    'recent_search_ms': recent_search_time,
                    'combine_score_ms': combined_time,
                    'merge_format_ms': merge_time
                }
            }
            
            # Log stats to database for analysis
            try:
                self.db.save_memory(
                    f"context_usage_{datetime.now().timestamp()}",
                    json.dumps(context_usage),
                    None,  # No embedding needed
                    {
                        'type': 'context_usage',
                        'session_id': self.session_id,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error saving context usage stats: {str(e)}")
            
            # Log completion and stats
            logger.info(f"Context retrieval complete in {total_time:.2f}ms")
            logger.info(f"Retrieved {len(merged_context)} memories for context")
            
            if top_memories:
                top_memory = top_memories[0]
                logger.info(f"Top memory ({top_memory.get('final_score', 0):.2f}): {top_memory.get('content', '')[:100]}...")
            
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
