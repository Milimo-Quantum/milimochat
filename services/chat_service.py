from typing import Dict, Any, List, AsyncGenerator, Optional, Union
from datetime import datetime
import asyncio
import json
import streamlit as st

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
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a user message with enhanced context integration."""
        try:
            # Generate embeddings for message
            message_embedding = await self.ollama_client.generate_embeddings(message)
            
            # Get relevant memories
            # Convert message embedding to list if it's numpy array
            query_embedding = message_embedding.tolist() if hasattr(message_embedding, 'tolist') else message_embedding
            
            # Comprehensive long-term memory search
            similar_memories = []
            if query_embedding:
                similar_memories = self.db.get_memory_by_similarity(
                    query_embedding,
                    self.session_id,
                    top_k=50,  # Increased capacity for exhaustive search
                    similarity_threshold=0.25,  # Broader context capture
                    include_temporal=True,
                    include_attachments=True  # Ensure file content is included
                )

            # Fetch all long-term memories
            all_long_term_memories = await self.memory_manager.get_all_long_term_memories()

            # Merge similar memories with all long-term memories, prioritizing similar ones
            long_term_memories = similar_memories + all_long_term_memories

            # Deduplicate long-term memories to avoid redundancy
            long_term_memories = self.context_integrator.deduplicate_memories(long_term_memories)

            # Format context from long-term memories only
            context_messages, context_metadata = self.context_formatter.format_memories(
                [],  # Empty short-term memories
                long_term_memories,
                message,
                include_temporal=True,
                redundancy_filter=False
            )
            
            # Validate context
            validated_messages, validation_info = self.context_formatter.validate_context(
                context_messages
            )

            # Store message with context metadata
            await self.memory_manager.add_message(
                MESSAGE_TYPES["USER"],
                message,
                {
                    "embedding": query_embedding,
                    "file_data": file_data,
                    "context_metadata": context_metadata,
                    "validation_info": validation_info,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Prepare messages for model
            messages = self._prepare_messages_with_context(
                message,
                validated_messages,
                file_data
            )

            # Generate response with streaming
            response_text = ""
            async for response_chunk in self.ollama_client.generate_chat_response(
                st.session_state.get(SessionKeys.CURRENT_MODEL),
                messages,
                stream=stream
            ):
                if stream:
                    yield response_chunk
                if response_chunk:
                    response_text += response_chunk

            if response_text:
                # Store assistant response with context tracking
                response_embedding = await self.ollama_client.generate_embeddings(response_text)
                response_embedding_list = response_embedding.tolist() if hasattr(response_embedding, 'tolist') else response_embedding
                
                # Track context usage
                usage_stats = self.context_formatter.get_context_usage_stats()
                
                await self.memory_manager.add_message(
                    MESSAGE_TYPES["ASSISTANT"],
                    response_text,
                    {
                        "embedding": response_embedding_list,
                        "context_metadata": {
                            **context_metadata,
                            "usage_stats": usage_stats
                        },
                        "validation_info": validation_info,
                        "timestamp": datetime.now().isoformat()
                    }
                )

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
