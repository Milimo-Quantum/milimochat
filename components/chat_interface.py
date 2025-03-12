import streamlit as st
import asyncio
import json
import base64
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import uuid
import io
from pathlib import Path
import numpy as np

from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

from utils.session_state import ChatSessionState
from utils.ollama_client import OllamaClient
from utils.file_processor import FileProcessor
from utils.memory_manager import MemoryManager
from services.database import Database
from services.chat_service import ChatService
from services.context_integrator import ContextIntegrator
from config import (
    PERSONALITY_PRESETS,
    MESSAGE_TYPES,
    SessionKeys,
    ERROR_MESSAGES,
    VISION_MODEL,
    EMBED_MODEL,
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
    DEFAULT_MODEL_PARAMS,
    IMAGE_PROMPT
)

class ChatInterface:
    def __init__(self, session_state=None):
        """Initialize chat interface with RAG support and context awareness."""
        self.ollama_client = OllamaClient()
        self.file_processor = FileProcessor()
        self.chat_state = session_state if session_state else ChatSessionState()

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Ensure custom prompts are initialized
        if "custom_personality_prompts" not in st.session_state:
            st.session_state.custom_personality_prompts = {}

        self.memory_manager = MemoryManager(st.session_state.session_id)
        self.context_integrator = ContextIntegrator(
            max_context_length=MODEL_PARAMETERS["context_length"]
        )
        self.db = Database()
        self.initialize_session_state()

        # Initialize chat service with session state
        self.chat_service = ChatService(st.session_state.session_id)

        # Initialize assistant icon
        if 'assistant_icon' not in st.session_state:
            self.load_assistant_icon()

        # Initialize RAG-specific state
        self._initialize_rag_state()

    def _initialize_rag_state(self):
        """Initialize RAG-specific session state variables."""
        if "vector_contexts" not in st.session_state:
            st.session_state.vector_contexts = {}
        if "chunk_metadata" not in st.session_state:
            st.session_state.chunk_metadata = {}
        if "relevance_scores" not in st.session_state:
            st.session_state.relevance_scores = {}
        if "processing_files" not in st.session_state:
            st.session_state.processing_files = set()
        if "memory_indicators" not in st.session_state:
            st.session_state.memory_indicators = {}
        if "context_tracking" not in st.session_state:
            st.session_state.context_tracking = {}
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

    def initialize_session_state(self):
        """Initialize all required session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None
        if "chat_container" not in st.session_state:
            st.session_state.chat_container = None
        if "current_image" not in st.session_state:
            st.session_state.current_image = None
        if SessionKeys.PERSONALITY not in st.session_state:
            st.session_state[SessionKeys.PERSONALITY] = "Professional"

    def load_assistant_icon(self):
        """Load and cache the assistant icon in base64 format."""
        try:
            icon_path = Path('assets/aicon.png')
            if icon_path.exists():
                with open(icon_path, "rb") as f:
                    icon_bytes = f.read()
                    st.session_state.assistant_icon = base64.b64encode(icon_bytes).decode()
            else:
                st.session_state.assistant_icon = None
        except Exception as e:
            st.error(f"Error loading assistant icon: {str(e)}")
            st.session_state.assistant_icon = None

    def encode_image_to_base64(self, image_file) -> str:
        """Convert uploaded image to base64 encoding."""
        if image_file is None:
            return None

        try:
            bytes_data = image_file.getvalue()
            base64_encoded = base64.b64encode(bytes_data).decode('utf-8')
            return base64_encoded
        except Exception as e:
            st.error(f"Error encoding image: {str(e)}")
            return None

    def get_assistant_avatar(self):
        """Get the assistant avatar image URL."""
        if st.session_state.get('assistant_icon'):
            return f"data:image/png;base64,{st.session_state.assistant_icon}"
        return None

    def render(self):
        """Render the enhanced chat interface with RAG support and context awareness."""
        colored_header(
            label="Digital Concierge",
            description="Locally powered by Streamlit & Meta Ollama LLMs",
            color_name="blue-70"
        )

        # Apply enhanced styling with context awareness
        self._apply_enhanced_styling()

        # Create main chat container
        chat_container = st.container()

        # Handle context additions
        if hasattr(st.session_state, 'use_context') and st.session_state.use_context:
            with st.chat_message("system"):
                st.markdown("🔄 Added context to the conversation:")
                st.markdown(st.session_state.use_context['content'])
                if 'relevance_score' in st.session_state.use_context:
                    st.markdown(f"*Relevance: {st.session_state.use_context['relevance_score']:.2f}*")

            st.session_state.use_context = None

        # Check for retry message
        if hasattr(st.session_state, 'retry_message') and st.session_state.retry_message:
            retry_msg = st.session_state.retry_message
            prompt = retry_msg['content']

            with st.chat_message("user"):
                st.markdown(prompt)

            self.chat_state.save_message(
                role=MESSAGE_TYPES["USER"],
                content=prompt,
                metadata=retry_msg.get('metadata', {})
            )

            with st.chat_message("assistant", avatar=self.get_assistant_avatar()):
                # Check if this is the first message in the conversation
                is_initial_message = len(self.chat_state.get_messages()) <= 1
                
                response = asyncio.run(self._generate_streaming_response(prompt, is_initial_message))
                if response:
                    self.chat_state.save_message(
                        role=MESSAGE_TYPES["ASSISTANT"],
                        content=response,
                        metadata={"is_response_to_initial": is_initial_message}
                    )

            st.session_state.retry_message = None
            st.rerun()

        # Display chat messages with enhanced context
        with chat_container:
            messages = self.chat_state.get_messages()
            for message in messages:
                context_info = st.session_state.memory_indicators.get(
                    message.get('content', '')
                )

                # Determine message classes based on memory usage
                message_classes = []
                if context_info:
                    message_classes.append('context-used')
                    if context_info['memory_types']['short_term'] > 0:
                        message_classes.append('memory-recent')
                    if context_info['memory_types']['long_term'] > 0:
                        message_classes.append('memory-past')

                # Create message container with enhanced styling
                with stylable_container(
                    key=f"message_{message.get('timestamp', '')}",
                    css_styles=f"""
                        {{
                            {''.join(message_classes)}
                            margin: 0.5rem 0;
                            padding: 1rem;
                            border-radius: 0.5rem;
                        }}
                    """
                ):
                    with st.chat_message(
                        message["role"],
                        avatar=None if message["role"] == "user" else self.get_assistant_avatar()
                    ):
                        # Format message content with context indicators
                        formatted_content = self._format_message_with_context(
                            message["content"],
                            context_info
                        )
                        st.markdown(formatted_content, unsafe_allow_html=True)

                        # Display RAG context if available
                        if message.get("metadata") and message["metadata"].get("vector_context"):
                            with st.expander("View Context", expanded=False):
                                self._render_vector_context(message["metadata"]["vector_context"])

                        # Display file attachments
                        if message.get("metadata") and message["metadata"].get("file_data"):
                            file_data = message["metadata"]["file_data"]
                            if file_data['type'] == 'image' and file_data.get('base64_data'):
                                st.image(f"data:image/png;base64,{file_data['base64_data']}", caption=file_data['name'], use_column_width=True)

                            self._render_file_preview(file_data)

        # Enhanced chat input with context awareness
        if prompt := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process current image if available
            if st.session_state.current_image:
                prompt = f"Analyze this image: {prompt}" if prompt else "What is in this image?"

            self.chat_state.save_message(
                role=MESSAGE_TYPES["USER"],
                content=prompt
            )

            with st.chat_message("assistant", avatar=self.get_assistant_avatar()):
                # Check if this is the first message in the conversation
                is_initial_message = len(self.chat_state.get_messages()) <= 1
                
                response = asyncio.run(self._generate_streaming_response(prompt, is_initial_message))
                if response:
                    self.chat_state.save_message(
                        role=MESSAGE_TYPES["ASSISTANT"],
                        content=response,
                        metadata={"is_response_to_initial": is_initial_message}
                    )
                    st.session_state.current_image = None

        # Enhanced file upload section
        self._render_file_upload_section()

    def _apply_enhanced_styling(self):
        """Apply enhanced styling with context and memory indicators."""
        st.markdown("""
            <style>
                /* Main container adjustments */
                .main > .block-container {
                    padding-bottom: 80px;
                }

                /* Fix all text visibility and code formatting */
                code {
                    color: #00C8FF !important;
                    background-color: rgba(0, 78, 146, 0.95) !important;
                }

                pre {
                    background-color: rgba(0, 78, 146, 0.95) !important;
                    border-radius: 0.5rem !important;
                    padding: 1rem !important;
                    border: 1px solid rgba(0, 200, 255, 0.1) !important;
                }

                .stMarkdown, p, span, label, .streamlit-expanderHeader {
                    color: rgba(255, 255, 255, 0.87) !important;
                }

                /* Message styling */
                [data-testid="stChatMessage"] {
                    background: rgba(0, 78, 146, 0.1);
                    border-radius: 1rem;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border: 1px solid rgba(0, 200, 255, 0.2);
                    backdrop-filter: blur(10px);
                    color: rgba(255, 255, 255, 0.87) !important;
                }

                /* User message styling */
                [data-testid="stChatMessage"][data-testid="user"] {
                    background: linear-gradient(90deg, rgba(0, 200, 255, 0.1) 0%, rgba(0, 153, 255, 0.1) 100%);
                    border-left: 3px solid #00C8FF;
                }

                /* Assistant message styling */
                [data-testid="stChatMessage"][data-testid="assistant"] {
                    background: rgba(0, 78, 146, 0.1);
                    border-left: 3px solid #00C8FF;
                }

                /* File upload styling */
                [data-testid="stFileUploader"] {
                    background: rgba(0, 78, 146, 0.7);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 0.5rem;
                    color: rgba(255, 255, 255, 0.87) !important;
                }

                [data-testid="stFileUploader"] section {
                    border: 1px dashed rgba(0, 200, 255, 0.2) !important;
                    background: rgba(0, 0, 0, 0.2) !important;
                    padding: 1rem !important;
                    border-radius: 0.5rem !important;
                }

                /* Fix file uploader text */
                [data-testid="stFileUploader"] small {
                    color: rgba(255, 255, 255, 0.6) !important;
                }

                /* Avatar image sizing and positioning */
                [data-testid="stChatMessage"] img {
                    width: 32px !important;
                    height: 32px !important;
                    border-radius: 50% !important;
                    object-fit: cover !important;
                }

                /* Fix button text color */
                button p {
                    color: rgba(255, 255, 255, 0.87) !important;
                }
            </style>
        """, unsafe_allow_html=True)

    def _format_message_with_context(
        self,
        content: str,
        context_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format message with context and memory indicators."""
        message_parts = [content]

        if context_info:
            # Add memory usage indicators with enhanced styling
            memory_indicator = (
                f"\n\n<div class='memory-indicator'>"
                f"<span style='display: flex; align-items: center; gap: 0.5rem;'>"
                # f"<span>🧠</span>"
                # f"<span>Using {context_info['total_memories']} memories "
                # f"({context_info['memory_types']['short_term']} recent, "
                # f"{context_info['memory_types']['long_term']} past)</span>"
                f"</span>"
                f"</div>"
            )
            message_parts.append(memory_indicator)

            # Add timestamp if available
            if 'timestamp' in context_info:
                time_str = datetime.fromisoformat(context_info['timestamp']).strftime("%H:%M:%S")
                message_parts.append(
                    f"<div class='memory-timestamp' style='font-size: 0.7em; "
                    f"color: rgba(255, 255, 255, 0.4); margin-top: 0.25rem;'>"
                    f"Context retrieved at {time_str}</div>"
                )

        return "".join(message_parts)

    async def _generate_streaming_response(self, prompt: str, is_initial_message: bool = False) -> Optional[str]:
        """Generate streaming response with enhanced context and file handling.
        
        Args:
            prompt: The user's message
            is_initial_message: Whether this is the first message in a conversation
        """

        try:
            current_model = st.session_state.get(SessionKeys.CURRENT_MODEL)
            if not current_model:
                st.error("No AI model selected. Please select a model in the sidebar.")
                return None

            # Show context retrieval status
            with st.status("Processing...", expanded=True) as status:
                # status.write("📚 Retrieving relevant context...")

                # Get enhanced context with metadata
                context, context_metadata = await self.memory_manager.get_relevant_context(prompt)

                if context:
                    # Get memory statistics from context metadata
                    # Calculate actual memory usage from validated counts
                    memory_types = {
                        'short_term': context_metadata.get('short_term', 0),
                        'long_term': context_metadata.get('long_term', 0)
                    }
                    total_memories = memory_types['short_term'] + memory_types['long_term']

                    # Validate and update metadata
                    context_metadata.update({
                        'total_memories_used': total_memories,
                        'short_term_count': memory_types['short_term'],
                        'long_term_count': memory_types['long_term']
                    })

                    # Show context usage in status
                    # status.write(f"🧠 Using {total_memories} relevant memories:")
                    # if total_memories > 0:
                    #     status.write(f"• {memory_types['short_term']} recent memories ({(memory_types['short_term']/total_memories)*100:.1f}%)")
                    #     status.write(f"• {memory_types['long_term']} past memories ({(memory_types['long_term']/total_memories)*100:.1f}%)")
                    # else:
                    #     status.write("• No relevant memories found")

                    # Update memory indicators with validated values
                    st.session_state.memory_indicators[prompt] = {
                        'total_memories': total_memories,
                        'memory_types': {
                            'short_term': memory_types['short_term'],
                            'long_term': memory_types['long_term']
                        } if total_memories > 0 else {'short_term': 0, 'long_term': 0},
                        'timestamp': datetime.now().isoformat(),
                        'relevance_scores': [
                            item.get('relevance_score', 0.0)
                            for item in context
                            if isinstance(item, dict)
                        ]
                    }

                # Process file context if present
                file_context = []
                if hasattr(st.session_state, '_current_file_data'):
                    status.write("Processing file data...")
                    file_data = st.session_state._current_file_data

                    if file_data["type"] == "document":
                        relevant_chunks = await self.file_processor.get_relevant_chunks(
                            prompt,
                            file_data,
                            top_k=2
                        )
                        if relevant_chunks:
                            file_context.extend(relevant_chunks)

                    elif file_data["type"] == "image":
                        if file_data.get('initial_analysis'):
                            file_context.append(file_data['initial_analysis'])

                # status.write("💭 Generating response...")

                # Pass the is_initial_message flag to process_message
                message_placeholder = st.empty()
                full_response = ""

                # Log the message status for debugging
                if is_initial_message:
                    status.write("💬 Processing initial message in conversation...")
                else:
                    status.write("💬 Processing follow-up message...")

                try:
                    async for chunk in self.chat_service.process_message(
                        prompt,
                        file_data=st.session_state.get('_current_file_data'),
                        stream=True,
                        is_initial_message=is_initial_message
                    ):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    if not full_response:
                        st.error("No response received from the model.")
                        return None

                    # Update message display with memory indicators
                    message_placeholder.markdown(
                        self._format_message_with_context(
                            full_response,
                            #context_info=st.session_state.memory_indicators.get(prompt)
                        )
                    )

                    # Track context usage
                    st.session_state.context_tracking[prompt] = {
                        'timestamp': datetime.now().isoformat(),
                        'context_used': bool(context),
                        'memory_count': total_memories if context else 0,
                        'file_context': bool(file_context),
                        'response_length': len(full_response)
                    }

                    # Clean up file context
                    if hasattr(st.session_state, '_current_file_data'):
                        del st.session_state._current_file_data

                    return full_response

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    return None

        except Exception as e:
            st.error(f"Error in response generation: {str(e)}")
            return None

    def _render_file_preview(self, file_data: Dict[str, Any]):
        """Render file preview with enhanced context awareness."""
        try:
            # Create tabs for different aspects of the file
            preview_tabs = st.tabs(["📄 Content", "📊 Chunks", "ℹ️ Metadata"])

            with preview_tabs[0]:
                # Basic content preview
                if 'content' in file_data:
                    with stylable_container(
                        key="content_preview",
                        css_styles="""
                            {
                                background: rgba(0, 78, 146, 0.1);
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin: 0.5rem 0;
                            }
                        """
                    ):
                        st.markdown("### Document Content")
                        st.markdown(file_data['content'])

            with preview_tabs[1]:
                if 'chunks' in file_data:
                    # Chunk navigation
                    st.markdown("### Document Chunks")
                    total_chunks = len(file_data['chunks'])

                    # Chunk selection
                    selected_chunk = st.selectbox(
                        "Select chunk to view",
                        range(1, total_chunks + 1),
                        format_func=lambda x: f"Chunk {x} of {total_chunks}"
                    )

                    # Display selected chunk with context info
                    chunk_idx = selected_chunk - 1
                    with stylable_container(
                        key=f"chunk_{chunk_idx}",
                        css_styles="""
                            {
                                background: rgba(0, 78, 146, 0.1);
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin: 0.5rem 0;
                            }
                        """
                    ):
                        st.markdown(f"**Chunk Content:**")
                        st.markdown(file_data['chunks'][chunk_idx])

                        # Show embedding status if available
                        if 'chunk_embeddings' in file_data:
                            st.info("📊 Vector embedding generated for this chunk")

                            # Show similarity to current context if available
                            if 'relevance_scores' in st.session_state and chunk_idx in st.session_state.relevance_scores:
                                score = st.session_state.relevance_scores[chunk_idx]
                                st.progress(score, text=f"Relevance: {score:.2f}")

                        # Show analysis if available
                        if ('chunk_analyses' in file_data and
                            chunk_idx < len(file_data['chunk_analyses'])):
                            with stylable_container(
                                key=f"analysis_{chunk_idx}",
                                css_styles="""
                                    {
                                        background: rgba(0, 78, 146, 0.1);
                                        border-radius: 0.5rem;
                                        padding: 1rem;
                                        margin: 0.5rem 0;
                                    }
                                """
                            ):
                                st.markdown("**Chunk Analysis:**")
                                st.markdown(file_data['chunk_analyses'][chunk_idx])

            with preview_tabs[2]:
                # Metadata display with vector context
                if 'metadata' in file_data:
                    with stylable_container(
                        key="metadata_display",
                        css_styles="""
                            {
                                background: rgba(0, 78, 146, 0.1);
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin: 0.5rem 0;
                            }
                        """
                    ):
                        st.markdown("### File Information")

                        # Create two columns for metadata
                        meta_col1, meta_col2 = st.columns(2)

                        with meta_col1:
                            st.markdown("**File Details:**")
                            st.markdown(f"- Name: {file_data['metadata'].get('filename', 'N/A')}")
                            st.markdown(f"- Type: {file_data['metadata'].get('mime_type', 'N/A')}")
                            st.markdown(f"- Size: {self._format_file_size(file_data['metadata'].get('size', 0))}")

                        with meta_col2:
                            st.markdown("**Processing Details:**")
                            st.markdown(f"- Chunks: {len(file_data.get('chunks', []))}")
                            st.markdown(f"- Embeddings: {len(file_data.get('chunk_embeddings', []))}")
                            st.markdown(f"- Created: {file_data['metadata'].get('created_at', 'N/A')}")

                        # Show vector information if available
                        if 'vector_context' in file_data.get('metadata', {}):
                            st.markdown("### Vector Information")
                            st.json(file_data['metadata']['vector_context'])

        except Exception as e:
            st.error(f"Error displaying file preview: {str(e)}")

    def _render_vector_context(self, context: Dict[str, Any]):
        """Render vector-based context information."""
        if not context:
            return

        st.markdown("#### Related Context")

        # Display source information
        if 'source' in context:
            st.markdown(f"**Source:** {context['source']}")

        # Show relevance score if available
        if 'relevance_score' in context:
            st.markdown(f"**Relevance Score:** {context['relevance_score']:.2f}")

        # Display context chunks with relevance
        if 'chunks' in context:
            for i, chunk in enumerate(context['chunks']):
                st.markdown(f"**Context Chunk {i + 1}:**")
                st.markdown(chunk)
                if 'chunk_scores' in context:
                    st.caption(f"Relevance: {context['chunk_scores'][i]:.2f}")
                st.markdown("---") # Separator for chunks

    def _format_file_size(self, size_in_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f} {unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f} TB"

    def _render_file_upload_section(self):
        """Render enhanced file upload section with RAG support."""
        upload_container = st.container()

        with upload_container:
            cols = st.columns(2)

            with cols[0]:
                image_file = st.file_uploader(
                    "Upload Image",
                    type=['png', 'jpg', 'jpeg', 'gif'],
                    key=f"image_upload_{st.session_state.session_id}",
                    help="Limit 200MB per file • PNG, JPG, JPEG, GIF"
                )

                # Initialize processed files tracking if not exists
                if "processed_files" not in st.session_state:
                    st.session_state.processed_files = set()

                # Initialize file history if not exists
                if "file_history" not in st.session_state:
                    st.session_state.file_history = []

                # Process image upload
                if image_file is not None and image_file.name not in st.session_state.processed_files:
                    with st.status("Processing image...", expanded=True) as status:
                        status.write("Analyzing image...")
                        asyncio.run(self._process_image_upload(image_file))

            with cols[1]:
                doc_file = st.file_uploader(
                    "Upload Document",
                    type=['pdf', 'txt', 'csv', 'xlsx'],
                    key=f"doc_upload_{st.session_state.session_id}",
                    help="Limit 200MB per file • PDF, TXT, CSV, XLSX"
                )

                # Process document upload
                if doc_file is not None and doc_file.name not in st.session_state.processed_files:
                    with st.status("Processing document...", expanded=True) as status:
                        status.write("Analyzing document structure...")
                        asyncio.run(self._process_document_upload(doc_file, status))

            # Display file history
            if st.session_state.file_history:
                with st.expander("Recent Files", expanded=False):
                    for file_info in reversed(st.session_state.file_history[-5:]):
                        # Create file history entry with context info
                        with stylable_container(
                            key=f"file_history_{file_info['name']}",
                            css_styles="""
                                {
                                    background: rgba(0, 78, 146, 0.1);
                                    border-radius: 0.5rem;
                                    padding: 0.5rem;
                                    margin-bottom: 0.5rem;
                                }
                            """
                        ):
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.markdown(f"**{file_info['name']}** ({file_info['type']})")
                                st.caption(f"Processed: {file_info['timestamp']}")
                            with cols[1]:
                                if st.button("Use Context", key=f"use_context_{file_info['name']}"):
                                    self._use_file_context(file_info)

    async def _process_image_upload(self, image_file):
        """Process image files with enhanced RAG support."""
        try:
            # Process file and generate base64
            file_data = await self.file_processor.process_file(image_file)
            base64_image = self.encode_image_to_base64(image_file)

            if base64_image:
                st.session_state.current_image = base64_image

                try:
                    # Check models first
                    models = self.ollama_client.list_models()
                    model_names = [model["name"] for model in models]

                    # Generate initial image analysis
                    if VISION_MODEL in model_names:
                        st.info("Analyzing image...")
                        analysis_text = ""
                        async for chunk in self.ollama_client.generate_chat_response(
                            model=VISION_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": IMAGE_PROMPT,
                                    "images": [base64_image]
                                }
                            ],
                            stream=True
                        ):
                            if chunk:
                                analysis_text += chunk
                        # st.write(f"Analysis: {analysis_text}") # Remove chunked output
                    st.markdown(f"Analysis: {analysis_text}") # Display complete analysis after streaming
                    file_data['initial_analysis'] = analysis_text

                        # Generate embedding for the analysis
                    analysis_embedding = await self.ollama_client.generate_embeddings(
                        analysis_text
                    )
                    file_data['embedding'] = analysis_embedding.tolist() if hasattr(analysis_embedding, 'tolist') else analysis_embedding

                except Exception as e:
                    st.error(f"Error generating image analysis: {str(e)}")

                # Update session state
                message_metadata = {
                    'file_data': file_data,
                    'type': 'image',
                    'name': image_file.name,
                    'base64_data': base64_image,
                    'timestamp': datetime.now().isoformat(),
                    'vector_context': {
                        'type': 'image_embedding',
                        'timestamp': datetime.now().isoformat(),
                        'model': VISION_MODEL
                    }
                }

                # Add to file history
                if "file_history" not in st.session_state:
                    st.session_state.file_history = []

                st.session_state.file_history.append({
                    'name': image_file.name,
                    'type': 'image',
                    'timestamp': datetime.now().isoformat(),
                    'metadata': message_metadata
                })

                # Save to memory manager
                try:
                    await self.memory_manager.add_message(
                        MESSAGE_TYPES["USER"],
                        f"Uploaded image: {image_file.name}",
                        message_metadata
                    )
                except Exception as e:
                    st.error(f"Error saving to memory: {str(e)}")
                    return

                # Display preview with enhanced context
                with st.chat_message("user"):
                    st.markdown(f"Uploaded image: {image_file.name}")
                    if base64_image: # Display image immediately after upload
                        st.image(f"data:image/png;base64,{base64_image}", use_container_width=True)
                    if 'embedding' in file_data:
                        st.success("✓ Vector embedding generated successfully")

                        # Show relevance to current context if available
                        if hasattr(st.session_state, '_current_context'):
                            relevance = self.context_integrator.calculate_relevance(
                                file_data['embedding'],
                                st.session_state._current_context
                            )
                            st.progress(relevance, text=f"Context Relevance: {relevance:.2f}")

                # Mark file as processed
                st.session_state.processed_files.add(image_file.name)

                # Store for context
                st.session_state._current_file_data = file_data

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    async def _process_document_upload(self, doc_file, progress_callback=None):
        """Process document with enhanced RAG support."""
        try:
            if progress_callback:
                progress_callback.write("Analyzing document structure...")

            # First check model availability
            models = self.ollama_client.list_models()
            model_names = [model["name"] for model in models]

            if EMBED_MODEL not in model_names:
                raise ValueError(f"Required model {EMBED_MODEL} not found")

            # Initialize document context if not exists
            if "document_contexts" not in st.session_state:
                st.session_state.document_contexts = {}

            # Process document with progress updates
            file_data = await self.file_processor.process_document(
                doc_file,
                progress_callback=progress_callback
            )

            if progress_callback:
                progress_callback.write("Processing document content...")

            # Update session state
            message_metadata = {
                'file_data': file_data,
                'type': 'document',
                'name': doc_file.name,
                'timestamp': datetime.now().isoformat(),
                'vector_context': {
                    'type': 'document_chunks',
                    'chunk_count': len(file_data.get('chunks', [])),
                    'embedding_count': len(file_data.get('chunk_embeddings', [])),
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Add to document contexts
            st.session_state.document_contexts[doc_file.name] = {
                'chunks': file_data.get('chunks', []),
                'embeddings': file_data.get('chunk_embeddings', []),
                'analyses': file_data.get('chunk_analyses', []),
                'timestamp': datetime.now().isoformat()
            }

            # Add to file history
            if "file_history" not in st.session_state:
                st.session_state.file_history = []

            st.session_state.file_history.append({
                'name': doc_file.name,
                'type': 'document',
                'timestamp': datetime.now().isoformat(),
                'metadata': message_metadata
            })

            # Save message with enhanced context
            self.chat_state.save_message(
                role=MESSAGE_TYPES["USER"],
                content=f"Uploaded document: {doc_file.name}\nContent: {file_data.get('content', '')}",
                metadata=message_metadata
            )

            # Add to memory manager
            try:
                if progress_callback:
                    progress_callback.write("Saving to memory...")

                await self.memory_manager.add_message(
                    MESSAGE_TYPES["USER"],
                    f"Uploaded document: {doc_file.name}\nContent: {file_data.get('content', '')}",
                    message_metadata
                )
            except Exception as e:
                st.error(f"Error saving to memory: {str(e)}")
                return

            # Display preview with enhanced context
            with st.chat_message("user"):
                st.markdown(f"**Document Uploaded:** {doc_file.name}")

                # Use styled container for preview
                with stylable_container(
                    key=f"doc_preview_{doc_file.name}",
                    css_styles="""
                        {
                            background: rgba(0, 78, 146, 0.1);
                            border-radius: 0.5rem;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            border: 1px solid rgba(0, 200, 255, 0.2);
                        }
                    """
                ):
                    # Document preview with context info
                    tabs = st.tabs(["Chunks"])

                    with tabs[0]:
                        st.markdown("### Document Chunks")
                        for i, chunk in enumerate(file_data.get('chunks', [])):
                            st.markdown(f"**Chunk {i+1}:**") # Replace expander with markdown
                            st.markdown(chunk)
                            if file_data.get('chunk_embeddings', []):
                                st.success("✓ Vector embedding generated")
                            st.markdown("---") # Add separator

                # Mark file as processed
                st.session_state.processed_files.add(doc_file.name)

                # Store for context
                st.session_state._current_file_data = file_data

                if progress_callback:
                    progress_callback.write("Document processed successfully!")

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

    def _use_file_context(self, file_info: Dict[str, Any]):
        """Use file content as context for next interaction."""
        try:
            # Store file data for context
            if file_data := file_info.get('metadata', {}).get('file_data'):
                st.session_state._current_file_data = file_data
                st.success(f"Using {file_info['name']} as context for next message")
        except Exception as e:
            st.error(f"Error using file context: {str(e)}")

    def _add_to_file_history(self, file_info: Dict[str, Any]):
        """Add file to history with proper metadata."""
        try:
            if "file_history" not in st.session_state:
                st.session_state.file_history = []

            # Ensure we don't add duplicates
            existing = [f for f in st.session_state.file_history if f['name'] == file_info['name']]
            if not existing:
                st.session_state.file_history.append(file_info)

                # Keep only last 10 files
                if len(st.session_state.file_history) > 10:
                    st.session_state.file_history = st.session_state.file_history[-10:]
        except Exception as e:
            print(f"Error adding to file history: {str(e)}")
