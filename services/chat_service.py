from typing import Dict, Any, List, AsyncGenerator, Optional, Union
from datetime import datetime
import asyncio
import json
import streamlit as st
import logging

from utils.ollama_client import OllamaClient
from utils.memory_manager import MemoryManager
from services.database import Database
from utils.session_state import ChatSessionState
from services.context_format import ContextFormatter
from services.context_integrator import ContextIntegrator
from config import (
    PERSONALITY_PRESETS,
    MESSAGE_TYPES,
    SessionKeys,
    ERROR_MESSAGES,
    TONE_PRESETS,
    CREATIVITY_LEVELS,
    PROMPT_TEMPLATES,
    DEFAULT_MODEL_PARAMS,
    MODEL_PARAMETERS
)

class ChatService:
    def __init__(self, session_id: str):
        """Initialize chat service with enhanced context integration."""
        if not session_id:
            raise ValueError("Session ID is required")
            
        self.session_id = session_id
        self.ollama_client = OllamaClient()
        self.memory_manager = MemoryManager(session_id)
        self.db = Database()
        self.context_integrator = ContextIntegrator(
            max_context_length=MODEL_PARAMETERS["context_length"]
        )
        self.context_formatter = ContextFormatter(
            max_context_length=MODEL_PARAMETERS["context_length"]
        )
        
        # Initialize chat session state if not present
        if not hasattr(st.session_state, '_chat_session_state'):
            st.session_state._chat_session_state = ChatSessionState()

    async def process_message(
        self,
        message: str,
        file_data: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        is_initial_message: bool = False
    ) -> AsyncGenerator[str, None]:
        """Process a user message with enhanced context integration and exhaustive memory search."""
        logger = logging.getLogger(__name__)
        try:
            # Start timing for analytics
            start_time = datetime.now()
            
            # Generate embeddings for message with enhanced model
            logger.info(f"Generating embeddings for message: {message[:50]}...")
            message_embedding = await self.ollama_client.generate_embeddings(message)
            
            # Convert message embedding to list if it's numpy array
            query_embedding = message_embedding.tolist() if hasattr(message_embedding, 'tolist') else message_embedding
            
            # Prepare context depending on message type
            if is_initial_message:
                logger.info("Initial message detected, setting up fresh conversation context")
                long_term_memories = []
                context_messages = []
                context_metadata = {
                    "total_memories": 0, 
                    "info": "initial_message",
                    "search_time_ms": 0
                }
            else:
                search_start = datetime.now()
                logger.info("Performing comprehensive memory search for context retrieval")
                
                # Get all messages for this session
                all_messages = self.db.get_messages(self.session_id)
                
                # Phase 1: Semantic search using vector similarity
                logger.info("Phase 1: Performing vector similarity search")
                similar_memories = []
                if query_embedding:
                    # Increased search capacity for more thorough exploration
                    similar_memories = self.db.get_memory_by_similarity(
                        query_embedding,
                        self.session_id,
                        top_k=200,  # Significantly increased search depth
                        min_score=0.0,  # Include all memories for comprehensive search
                        search_messages=True  # Search both memories and message history
                    )
                    
                    logger.info(f"Retrieved {len(similar_memories)} memories through vector similarity")
                
                # Phase 2: Get all long-term memories as fallback
                logger.info("Phase 2: Retrieving all long-term memories to ensure completeness")
                all_long_term_memories = await self.memory_manager.get_all_long_term_memories()
                logger.info(f"Retrieved {len(all_long_term_memories)} long-term memories from storage")
                
                # Phase 3: Combine and deduplicate memories
                logger.info("Phase 3: Merging and deduplicating memory sources")
                combined_memories = similar_memories + all_long_term_memories
                
                # Advanced deduplication with intelligent pattern matching
                long_term_memories = self.context_integrator.deduplicate_memories(
                    combined_memories,
                    similarity_threshold=0.90  # Slightly higher threshold to preserve nuanced differences
                )
                
                logger.info(f"After deduplication: {len(long_term_memories)} unique memories")
                
                # Phase 4: Format and prepare context
                logger.info("Phase 4: Integrating memories into coherent context")
                # Use our enhanced context formatter directly
                merged_context, context_metadata = self.context_integrator.merge_context(
                    short_term_memories=self.memory_manager.get_recent_messages(5),  # Get 5 most recent messages
                    long_term_memories=long_term_memories,
                    current_context=message,
                    max_memories=25  # Increased from default to allow more comprehensive context
                )
                
                # Format context for prompt using new enhanced formatter
                context_text = self.context_integrator.format_context_for_prompt(
                    merged_context, 
                    context_metadata
                )
                
                # Validate context length
                context_text, validation_info = self.context_integrator.validate_context(
                    context_text,
                    max_length=MODEL_PARAMETERS["context_length"] - 1000  # Leave room for the message
                )
                
                # Prepare messages for the model
                context_messages = [
                    {
                        "role": MESSAGE_TYPES["SYSTEM"],
                        "content": context_text
                    }
                ]
                
                # Track search timing
                search_time = (datetime.now() - search_start).total_seconds() * 1000
                context_metadata["search_time_ms"] = search_time
                logger.info(f"Memory search and context preparation completed in {search_time:.2f}ms")
                logger.info(f"Using {context_metadata['total_memories']} memories in final context")

            # Store user message with enhanced context metadata
            await self.memory_manager.add_message(
                MESSAGE_TYPES["USER"],
                message,
                {
                    "embedding": query_embedding,
                    "file_data": file_data,
                    "context_metadata": context_metadata,
                    "validation_info": validation_info if 'validation_info' in locals() else {},
                    "timestamp": datetime.now().isoformat(),
                    "is_initial_message": is_initial_message,
                    "search_stats": {
                        "vector_matches": len(similar_memories) if 'similar_memories' in locals() else 0,
                        "long_term_memories": len(all_long_term_memories) if 'all_long_term_memories' in locals() else 0,
                        "combined_unique": len(long_term_memories) if 'long_term_memories' in locals() else 0
                    }
                }
            )

            # Prepare comprehensive messages for model
            messages = self._prepare_messages_with_context(
                message,
                context_messages,
                file_data
            )

            # Generate response with streaming and analytics tracking
            logger.info("Generating model response")
            response_text = ""
            token_count = 0
            
            # Track response generation performance
            generation_start = datetime.now()
            
            async for response_chunk in self.ollama_client.generate_chat_response(
                st.session_state.get(SessionKeys.CURRENT_MODEL),
                messages,
                stream=stream
            ):
                if stream:
                    yield response_chunk
                if response_chunk:
                    response_text += response_chunk
                    token_count += 1

            # Calculate response time
            generation_time = (datetime.now() - generation_start).total_seconds() * 1000
            logger.info(f"Response generated in {generation_time:.2f}ms with ~{token_count} chunks")

            if response_text:
                # Generate embedding for assistant response to enable future searches
                response_embedding = await self.ollama_client.generate_embeddings(response_text)
                response_embedding_list = response_embedding.tolist() if hasattr(response_embedding, 'tolist') else response_embedding
                
                # Enhanced metadata for better memory management and analytics
                response_metadata = {
                    "embedding": response_embedding_list,
                    "context_metadata": context_metadata,
                    "timestamp": datetime.now().isoformat(),
                    "generation_stats": {
                        "tokens_generated": token_count,
                        "generation_time_ms": generation_time,
                        "total_processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                    }
                }
                
                # Add validation info if available
                if 'validation_info' in locals():
                    response_metadata["validation_info"] = validation_info
                
                # Store assistant response with comprehensive tracking
                await self.memory_manager.add_message(
                    MESSAGE_TYPES["ASSISTANT"],
                    response_text,
                    response_metadata
                )
                
                # Log completion statistics
                logger.info(f"Message processing complete. Total time: {(datetime.now() - start_time).total_seconds() * 1000:.2f}ms")

            if not stream and response_text:
                yield response_text

        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            raise

    def _prepare_messages_with_context(
        self,
        message: str,
        context_messages: List[Dict[str, str]],
        file_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages with integrated context for the model."""
        try:
            messages = []
            
            # Add system prompt
            system_prompt = self.get_system_prompt()
            messages.append({
                "role": MESSAGE_TYPES["SYSTEM"],
                "content": system_prompt
            })
            
            # Add context messages
            messages.extend(context_messages)
            
            # Add file context if available
            if file_data:
                file_context = self._format_file_context(file_data)
                if file_context:
                    messages.append({
                        "role": MESSAGE_TYPES["SYSTEM"],
                        "content": file_context
                    })
            
            # Add user message
            messages.append({
                "role": MESSAGE_TYPES["USER"],
                "content": message
            })
            
            return messages
            
        except Exception as e:
            print(f"Error preparing messages: {str(e)}")
            return [{
                "role": MESSAGE_TYPES["USER"],
                "content": message
            }]

    def _format_file_context(self, file_data: Dict[str, Any]) -> Optional[str]:
        """Format file data into context string."""
        try:
            if not file_data:
                return None
                
            context_parts = []
            
            if file_data["type"] == "document":
                if file_data.get('chunks'):
                    context_parts.append("DOCUMENT CONTEXT:")
                    for i, chunk in enumerate(file_data['chunks'][:2]):  # First 2 chunks
                        context_parts.append(f"Chunk {i+1}: {chunk}")
                else:
                    context_parts.append(
                        f"Document content: {file_data.get('content', '')}"
                    )
            
            elif file_data["type"] == "image":
                context_parts.append("IMAGE CONTEXT:")
                if file_data.get('initial_analysis'):
                    context_parts.append(file_data['initial_analysis'])
            
            return "\n".join(context_parts) if context_parts else None
            
        except Exception as e:
            print(f"Error formatting file context: {str(e)}")
            return None

    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate model response with context-aware parameters."""
        try:
            current_model = st.session_state.get(SessionKeys.CURRENT_MODEL)
            if not current_model:
                raise ValueError("No AI model selected")

            # Generate response with streaming
            async for response_chunk in self.ollama_client.generate_chat_response(
                current_model,
                messages,
                stream=stream,
                **self._get_model_parameters()
            ):
                yield response_chunk

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise

    def get_system_prompt(
        self,
        personality_name: Optional[str] = None,
        db: Optional[Database] = None
    ) -> str:
        """Get the system prompt with RAG context awareness."""
        if personality_name is None:
            personality_name = self.get_current_personality()
        
        # Get the ChatSessionState instance
        chat_state = st.session_state._chat_session_state
        
        # Try to get custom prompt first
        custom_prompt = chat_state.get_custom_prompt(personality_name)
        if custom_prompt:
            return self._enhance_prompt_with_rag_context(custom_prompt)
        
        # Fall back to default prompt
        personality = PERSONALITY_PRESETS.get(personality_name)
        if personality:
            return self._enhance_prompt_with_rag_context(
                personality['system_prompt'](db or self.db)
            )
        
        return self._enhance_prompt_with_rag_context(
            PERSONALITY_PRESETS['Professional']['system_prompt'](db or self.db)
        )

    def _enhance_prompt_with_rag_context(self, base_prompt: str) -> str:
        """Enhance the base prompt with RAG-specific instructions."""
        rag_context = """
        RETRIEVAL AUGMENTED GENERATION INSTRUCTIONS:
        - Conduct exhaustive analysis of ALL memory sources (chats/documents/history)
        - Cross-reference minimum 3 distinct memory sources per response
        - Prioritize context depth over response brevity
        - Surface relevant context proactively, even when not explicitly asked
        - When a memory from the database is being used as context preserve original wording from critical sources with [Source: ...] attribution
        - Highlight temporal patterns (recency/frequency) in memory data
        - Flag and explain context conflicts/ambiguities immediately
        - Maintain conversation flow while integrating multiple context points
        """
        return f"{base_prompt}\n\n{rag_context}"

    def get_current_personality(self) -> str:
        """Get the currently selected personality."""
        return st.session_state.get(SessionKeys.PERSONALITY, "Professional")

    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters with RAG optimization."""
        params = {
            "temperature": self._get_temperature_from_creativity(
                st.session_state.get(SessionKeys.CREATIVITY, "Balanced")
            ),
            "top_p": self._get_top_p_from_tone(
                st.session_state.get(SessionKeys.TONE, "Balanced")
            ),
            "presence_penalty": DEFAULT_MODEL_PARAMS["presence_penalty"],
            "frequency_penalty": DEFAULT_MODEL_PARAMS["frequency_penalty"],
            "stop": DEFAULT_MODEL_PARAMS["stop"]
        }
        
        # RAG-specific parameters
        params.update({
            "context_length": MODEL_PARAMETERS["context_length"],
            "embedding_length": MODEL_PARAMETERS["embedding_length"],
            "num_ctx": MODEL_PARAMETERS["context_length"],
            "max_tokens": MODEL_PARAMETERS["max_sequence_length"]
        })
        
        return params

    def _get_temperature_from_creativity(self, creativity: str) -> float:
        """Convert creativity setting to temperature."""
        creativity_map = {k: v[1] for k, v in CREATIVITY_LEVELS}
        return creativity_map.get(creativity, 0.7)

    def _get_top_p_from_tone(self, tone: str) -> float:
        """Convert tone setting to top_p."""
        tone_map = {k: v[1] for k, v in TONE_PRESETS}
        return tone_map.get(tone, 0.5)
