import time
import streamlit as st
from datetime import datetime, timedelta
import json
import pandas as pd
import io
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import asyncio
import uuid

from streamlit_extras.grid import grid
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
import pyperclip

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from utils.memory_manager import MemoryManager
from services.database import Database
from config import MESSAGE_TYPES

class MemoryDashboard:
    def __init__(self):
        """Initialize dashboard with improved session persistence."""
        # Initialize database connection first
        self.db = Database()
        
        # First check URL parameters for session ID
        query_params = st.query_params
        session_id = query_params.get('session_id', None)
        
        # Then check session state
        if not session_id and "session_id" in st.session_state:
            session_id = st.session_state.session_id
            
        # If we have a session ID, verify it exists in database
        if session_id:
            try:
                # Attempt to retrieve existing session
                verified_session = self.db.get_or_create_session(session_id)
                st.session_state.session_id = verified_session
                self.session_id = verified_session
            except Exception as e:
                print(f"Error verifying session: {str(e)}")
                session_id = None
        
        # If no valid session ID found, create new one
        if not session_id:
            new_session = str(uuid.uuid4())
            st.session_state.session_id = new_session
            st.session_state.session_start = datetime.now().isoformat()
            self.session_id = new_session
            # Create session in database
            self.db.get_or_create_session(new_session)
        
        # Update URL with current session_id
        st.query_params["session_id"] = st.session_state.session_id
        
        # Initialize memory manager with session_id
        self.memory_manager = MemoryManager(st.session_state.session_id)
        
        # Initialize session state variables
        if "memory_search_query" not in st.session_state:
            st.session_state.memory_search_query = ""
        
        if "memory_sort_order" not in st.session_state:
            st.session_state.memory_sort_order = "newest"
        
        if "memories_page" not in st.session_state:
            st.session_state.memories_page = 1
        
        if "memories_per_page" not in st.session_state:
            st.session_state.memories_per_page = 10
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        
        # Initialize long-term memories list if not exists
        if "long_term_memories" not in st.session_state:
            st.session_state.long_term_memories = []
        
        # Always load fresh memories from database on initialization
        self._load_long_term_memories()

    def _initialize_persistent_session(self):
        """Initialize or restore session with enhanced persistence."""
        try:
            # Check URL parameters first
            query_params = st.query_params
            session_from_url = query_params.get("session_id", [None])[0]
            
            if session_from_url:
                # Verify session exists in database
                verified_session = self.db.get_or_create_session(session_from_url)
                st.session_state.session_id = verified_session
            elif "session_id" in st.session_state:
                # Verify existing session state
                verified_session = self.db.get_or_create_session(st.session_state.session_id)
                st.session_state.session_id = verified_session
            else:
                # Create new session
                new_session = str(uuid.uuid4())
                verified_session = self.db.get_or_create_session(new_session)
                st.session_state.session_id = verified_session
            
            # Ensure session_id is in URL parameters
            st.query_params["session_id"] = st.session_state.session_id
            
            # Initialize session start time if needed
            if "session_start" not in st.session_state:
                st.session_state.session_start = datetime.now().isoformat()
                
        except Exception as e:
            print(f"Error initializing persistent session: {str(e)}")
            # Fallback to new session if error occurs
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_start = datetime.now().isoformat()

    def _initialize_session(self):
        """Initialize or restore session with enhanced persistence."""
        try:
            # Get persistent session ID from URL parameters
            query_params = st.query_params
            persistent_id = query_params.get('persistent_id', None)
            
            if persistent_id:
                # Verify session exists in database
                try:
                    verified_session = self.db.get_or_create_session(persistent_id)
                    st.session_state.session_id = verified_session
                    st.session_state.persistent_id = verified_session
                except Exception as e:
                    print(f"Error verifying session: {str(e)}")
                    # Create new session if verification fails
                    new_session = str(uuid.uuid4())
                    st.session_state.session_id = new_session
                    st.session_state.persistent_id = new_session
            else:
                # Create new session if no persistent_id found
                new_session = str(uuid.uuid4())
                st.session_state.session_id = new_session
                st.session_state.persistent_id = new_session
            
            # Update URL with persistent_id
            st.query_params["persistent_id"] = st.session_state.persistent_id
            
            # Initialize session start time if needed
            if "session_start" not in st.session_state:
                st.session_state.session_start = datetime.now().isoformat()
                
        except Exception as e:
            print(f"Error initializing session: {str(e)}")
            # Fallback to new session if error occurs
            fallback_id = str(uuid.uuid4())
            st.session_state.session_id = fallback_id
            st.session_state.persistent_id = fallback_id
            st.session_state.session_start = datetime.now().isoformat()

    def _initialize_session_state(self):
        """Initialize all required session state variables."""
        defaults = {
            "memory_search_query": "",
            "memory_sort_order": "newest",
            "memories_page": 1,
            "memories_per_page": 10,
            "current_page": 1,
            "long_term_memories": None,
            "last_memory_update": None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _load_long_term_memories(self):
        """Load or refresh long-term memories from database with improved persistence."""
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Get session ID from session state or URL params
            session_id = st.session_state.session_id
            
            print(f"Loading memories from database")
            print(f"Current session_id: {session_id}")
            
            # Get memories from database using session_id
            memories = loop.run_until_complete(
                self.memory_manager.get_all_long_term_memories()
            )
            
            print(f"Retrieved {len(memories)} memories for session {session_id}")
            
            if memories:
                # Sort memories by created_at timestamp
                sorted_memories = sorted(
                    memories,
                    key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                    reverse=True
                )
                
                # Update session state with sorted memories
                st.session_state.long_term_memories = sorted_memories
                
                # Store the last update timestamp
                st.session_state.last_memory_update = datetime.now().isoformat()
                
                print(f"Loaded {len(sorted_memories)} memories into session state")
            else:
                st.session_state.long_term_memories = []
                st.session_state.last_memory_update = datetime.now().isoformat()
                print("No memories found in database")
                
        except Exception as e:
            print(f"Error loading long-term memories: {str(e)}")
            st.session_state.long_term_memories = []
            st.session_state.last_memory_update = datetime.now().isoformat()

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent messages with improved retrieval."""
        try:
            messages = self.memory_manager.get_recent_messages(limit)
            print(f"Retrieved {len(messages)} recent messages")
            return messages
        except Exception as e:
            print(f"Error retrieving recent messages: {str(e)}")
            return []

    def _initialize_memories(self):
        """Initialize or reload memories from database with improved persistence."""
        try:
            # Always fetch fresh data from database using memory manager
            memories = self.memory_manager.db.get_all_long_term_memories(st.session_state.session_id)
            
            if memories:
                # Sort memories by creation timestamp
                sorted_memories = sorted(
                    memories,
                    key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                    reverse=True
                )
                st.session_state.long_term_memories = sorted_memories
            else:
                st.session_state.long_term_memories = []
                
            # Update last refresh timestamp
            st.session_state.last_memory_update = datetime.now().isoformat()
            
        except Exception as e:
            print(f"Error initializing memories from database: {str(e)}")
            if 'long_term_memories' not in st.session_state:
                st.session_state.long_term_memories = []

    def _check_memory_updates(self) -> bool:
        """Check if memory updates are needed based on database state."""
        try:
            if not hasattr(st.session_state, 'last_memory_update'):
                return True
                
            # Get the latest memory from database
            latest_memories = self.memory_manager.db.get_all_long_term_memories(st.session_state.session_id)
            
            if not latest_memories:
                return False
                
            # Sort to get the most recent memory
            latest_memory = sorted(
                latest_memories,
                key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                reverse=True
            )[0]
            
            # Compare with our current memories
            current_memories = st.session_state.long_term_memories
            if not current_memories:
                return True
                
            latest_timestamp = datetime.fromisoformat(latest_memory.get('created_at', '1970-01-01'))
            current_latest = sorted(
                current_memories,
                key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                reverse=True
            )[0]
            current_timestamp = datetime.fromisoformat(current_latest.get('created_at', '1970-01-01'))
            
            # Check if database has newer memories
            return latest_timestamp > current_timestamp
                
        except Exception as e:
            print(f"Error checking memory updates: {str(e)}")
            return True  # Force refresh on error

    def _sort_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort memories based on current preference."""
        if not memories:
            return []
            
        try:
            if st.session_state.memory_sort_order == "newest":
                return sorted(
                    memories,
                    key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                    reverse=True
                )
            else:
                return sorted(
                    memories,
                    key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01'))
                )
        except Exception as e:
            print(f"Error sorting memories: {str(e)}")
            return memories
        
    def render(self):
        """Render the memory dashboard with improved memory loading."""
        colored_header(
            label="Memory Dashboard",
            description="Monitor and manage conversation memory",
            color_name="red-70"
        )
        
        # Check if memories need to be loaded/refreshed
        if (not hasattr(st.session_state, 'long_term_memories') or 
            not hasattr(st.session_state, 'last_memory_update') or
            self._check_memory_updates()):
            self._load_long_term_memories()
        
        # Create main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_memory_overview()
            self._render_memory_details()
        
        with col2:
            # Add manual refresh button at top
            if st.button("🔄 Refresh Memories"):
                self._load_long_term_memories()
                st.rerun()
                
            self._render_memory_controls()
            self._render_memory_stats()

    def _render_long_term_memory(self):
        """Render long-term memory with improved persistence and display."""
        try:
            # Get memories from session state
            memories = st.session_state.get('long_term_memories', [])
            
            # Print debug information
            print(f"Rendering {len(memories)} long-term memories")
            
            # Search interface
            search = st.text_input("🔍 Search Long-term Memories", key="memory_search")
            
            # Controls in single row
            cols = st.columns([2, 1])
            with cols[0]:
                sort_order = st.selectbox(
                    "Sort by",
                    ["newest", "oldest"],
                    key="memory_sort"
                )
            with cols[1]:
                items_per_page = st.selectbox(
                    "Items per page",
                    [10, 25, 50, 100],
                    key="items_per_page"
                )
            
            # Apply filters and search
            filtered_memories = memories
            if search:
                filtered_memories = [
                    memory for memory in memories 
                    if search.lower() in str(memory.get('content', '')).lower()
                ]
            
            # Sort memories
            filtered_memories = self._sort_memories(filtered_memories)
            
            # Display memories or info message
            if not filtered_memories:
                st.info("No long-term memories found. Start a conversation to create memories.")
                return
            
            # Pagination
            total_items = len(filtered_memories)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1
            
            # Calculate slice indices
            start_idx = (st.session_state.current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            
            # Display current page of memories with indices
            for idx, memory in enumerate(filtered_memories[start_idx:end_idx], start=start_idx):
                self._render_memory_card(memory, idx)
            
            # Pagination controls
            if total_pages > 1:
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    if st.button("← Previous", disabled=st.session_state.current_page == 1):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with cols[1]:
                    st.markdown(
                        f"Page {st.session_state.current_page} of {total_pages}",
                        help=f"Showing memories {start_idx + 1}-{end_idx} of {total_items}"
                    )
                
                with cols[2]:
                    if st.button("Next →", disabled=st.session_state.current_page == total_pages):
                        st.session_state.current_page += 1
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Error displaying long-term memory: {str(e)}")
            print(f"Error in _render_long_term_memory: {str(e)}")

    def _delete_memory(self, memory: Dict[str, Any]) -> None:
        """Delete a specific memory without affecting others."""
        try:
            if not memory or 'key' not in memory:
                st.error("Invalid memory selected for deletion")
                return

            # Delete specific memory from database
            self.db.delete_memory(memory['key'])
            
            # Update only the specific memory in session state
            if 'long_term_memories' in st.session_state:
                st.session_state.long_term_memories = [
                    m for m in st.session_state.long_term_memories 
                    if m.get('key') != memory['key']
                ]
            
            # Only reload if memory was deleted
            memories = self.db.get_all_long_term_memories(st.session_state.session_id)
            if memories is not None:
                st.session_state.long_term_memories = sorted(
                    memories,
                    key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                    reverse=True
                )
                st.session_state.last_memory_update = datetime.now().isoformat()
            
            st.rerun()

        except Exception as e:
            st.error(f"Failed to delete memory: {str(e)}")
            print(f"Error deleting memory: {str(e)}")  # For debugging

    def _render_memory_card(self, memory: Dict[str, Any], index: int):
        """Render a single memory card with unique keys."""
        if not isinstance(memory, dict) or not memory.get('content'):
            return
    
        # Ensure we have a valid memory key
        memory_key = None
        if memory.get('key'):
            memory_key = memory['key']
        elif memory.get('metadata', {}).get('memory_key'):
            memory_key = memory['metadata']['memory_key']
    
        if not memory_key:
            memory_key = str(uuid.uuid4())
    
        unique_prefix = f"mem_{index}_{memory_key}"
    
        with stylable_container(
            key=f"memory_card_{unique_prefix}",
            css_styles="""
                {
                    background: rgba(0, 78, 146, 0.1);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 0.5rem;
                    border-left: 3px solid #00C8FF;
                }
            """
        ):
            # Content and timestamp
            st.markdown(memory['content'])
    
            try:
                created_time = datetime.fromisoformat(memory['created_at']).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Created: {created_time}")
            except (ValueError, KeyError):
                pass
    
            # Action buttons with unique keys
            cols = st.columns([1, 1, 1, 1])
    
            # Copy button
            with cols[0]:
                if st.button("📋", key=f"copy_{unique_prefix}", help="Copy to clipboard"):
                    try:
                        pyperclip.copy(memory['content'])
                        st.success("Copied!", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to copy: {str(e)}")
    
            # Use Context button
            with cols[1]:
                if st.button("🔄", key=f"use_{unique_prefix}", help="Use as context"):
                    asyncio.run(self._use_as_context(memory))
                    st.success("Added to context!", icon="✅")
    
            # Details button
            with cols[2]:
                if st.button("🔍", key=f"details_{unique_prefix}", help="View details"):
                    with st.expander("Memory Details"):
                        if metadata := memory.get('metadata'):
                            for k, v in metadata.items():
                                if k != 'session_id':
                                    st.markdown(f"**{k.title()}:** {v}")
    
            # Delete button and confirmation
            with cols[3]:
                delete_key = f"delete_{unique_prefix}"
                confirm_key = f"confirm_delete_{unique_prefix}"
    
                if st.button("🗑️", key=delete_key, help="Delete memory"):
                    st.session_state[confirm_key] = True
    
                if st.session_state.get(confirm_key, False):
                    if st.checkbox("Confirm deletion", key=f"checkbox_{confirm_key}"):
                        try:
                            # Extract the correct key based on memory type
                            deletion_key = None
                            if memory.get('metadata'):
                                # Try to get key from metadata first
                                try:
                                    metadata = memory['metadata']
                                    if isinstance(metadata, str):
                                        metadata = json.loads(metadata)
                                    deletion_key = (metadata.get('memory_key') or
                                                    metadata.get('file_key') or
                                                    metadata.get('key'))
                                except json.JSONDecodeError:
                                    pass
    
                            # Fallback to direct key if no metadata key found
                            if not deletion_key:
                                deletion_key = memory.get('key')
    
                            if deletion_key:
                                # Delete the memory
                                self.db.delete_memory_complete(
                                    deletion_key,
                                    self.session_id
                                )
    
                                # Update session state
                                if 'long_term_memories' in st.session_state:
                                    st.session_state.long_term_memories = [
                                        m for m in st.session_state.long_term_memories
                                        if (m.get('key') != deletion_key and
                                            m.get('metadata', {}).get('memory_key') != deletion_key)
                                    ]
    
                                # Clear confirmation state
                                if confirm_key in st.session_state:
                                    del st.session_state[confirm_key]
    
                                st.success("Memory deleted successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete memory: {str(e)}")

    def _render_memory_details(self):
        """Render the memory details section with tabs."""
        st.markdown("## 📝 Memory Details")
        
        # Memory tabs
        tab1, tab2 = st.tabs(["Short-term Memory", "Long-term Memory"])
        
        with tab1:
            messages = self.memory_manager.get_recent_messages()
            if not messages:
                st.info("No messages in short-term memory.")
            else:
                for idx, msg in enumerate(messages):
                    with stylable_container(
                        key=f"short_term_msg_{msg.get('id', '')}_{idx}",
                        css_styles=f"""
                            {{
                                background: {"rgba(255, 75, 75, 0.1)" if msg['role'] == MESSAGE_TYPES["USER"] else "rgba(66, 135, 245, 0.1)"};
                                border-radius: 0.5rem;
                                padding: 1rem;
                                margin-bottom: 0.5rem;
                                border-left: 3px solid {"#FF4B4B" if msg['role'] == MESSAGE_TYPES["USER"] else "#4287f5"};
                            }}
                        """
                    ):
                        # Display role and timestamp
                        try:
                            timestamp = datetime.fromisoformat(msg['timestamp'])
                            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            role_display = "👤 User" if msg['role'] == MESSAGE_TYPES["USER"] else "🤖 Assistant"
                            st.markdown(f"**{role_display}** - {formatted_time}")
                        except (ValueError, TypeError):
                            role_display = "👤 User" if msg['role'] == MESSAGE_TYPES["USER"] else "🤖 Assistant"
                            st.markdown(f"**{role_display}**")

                        # Display message content
                        st.markdown(msg['content'])

                        # Action buttons with unique keys using message ID and index
                        cols = st.columns(3)
                        with cols[0]:
                            key = f"copy_st_{msg.get('id', '')}_{idx}"
                            if st.button("📋 Copy", key=key):
                                try:
                                    pyperclip.copy(msg['content'])
                                    st.success("Copied to clipboard!")
                                except Exception as e:
                                    st.error(f"Failed to copy: {str(e)}")
                        
                        with cols[1]:
                            key = f"details_st_{msg.get('id', '')}_{idx}"
                            if st.button("🔍 Details", key=key):
                                with st.expander("Message Details"):
                                    st.json({k: v for k, v in msg.items() if k != 'content'})
                        
                        with cols[2]:
                            key = f"delete_st_{msg.get('id', '')}_{idx}"
                            if st.button("🗑️ Delete", key=key):
                                if msg.get('id'):
                                    self.memory_manager.delete_message(msg['id'])
                                    st.success("Message deleted!")
                                    st.rerun()

        with tab2:
            self._render_long_term_memory()

    def _handle_memory_deletion(self, memory: Dict[str, Any]):
        """Handle memory deletion with confirmation."""
        try:
            # Create unique key for confirmation
            confirm_key = f"confirm_delete_{memory.get('key', '')}"
            
            # Show confirmation checkbox
            if st.checkbox("Confirm deletion", key=confirm_key):
                # Delete from database
                self.memory_manager.delete_memory(memory.get('key', ''))
                
                # Update session state
                st.session_state.long_term_memories = [
                    m for m in st.session_state.long_term_memories 
                    if m.get('key') != memory.get('key')
                ]
                
                # Update last memory update timestamp
                st.session_state.last_memory_update = datetime.now().isoformat()
                
                st.success("Memory deleted successfully!")
                
                # Remove confirmation checkbox from session state
                if confirm_key in st.session_state:
                    del st.session_state[confirm_key]
                
                st.rerun()
            else:
                st.warning("Please confirm deletion")

        except Exception as e:
            st.error(f"Failed to delete memory: {str(e)}")

    async def _use_as_context(self, memory: Dict[str, Any]):
        """Add memory to conversation context."""
        try:
            if 'content' not in memory:
                st.error("Invalid memory content")
                return

            # Add memory as context message
            await self.memory_manager.add_message(
                MESSAGE_TYPES["SYSTEM"],
                f"Previous Context: {memory['content']}",
                {
                    'type': 'context_memory',
                    'source': 'long_term',
                    'original_timestamp': memory.get('created_at'),
                    'memory_key': memory.get('key'),
                    'added_as_context_at': datetime.now().isoformat()
                }
            )

            # Update memory access time
            memory_copy = memory.copy()
            memory_copy['last_accessed'] = datetime.now().isoformat()
            if 'access_times' not in memory_copy:
                memory_copy['access_times'] = []
            memory_copy['access_times'].append(memory_copy['last_accessed'])

            # Save updated memory
            self.db.save_memory(
                memory_copy.get('key'),
                memory_copy.get('content'),
                None,  # No embedding needed for update
                memory_copy.get('metadata', {})
            )

            # Update session state for navigation
            if "use_context" not in st.session_state:
                st.session_state.use_context = None
                
            st.session_state.use_context = {
                'content': memory['content'],
                'timestamp': datetime.now().isoformat()
            }

            # Provide feedback and redirect
            st.success("Context added successfully! Redirecting to chat...")
            time.sleep(1)  # Brief pause for user feedback
            
            # Navigate to chat view
            st.query_params["view"] = "chat"
            st.switch_page("main.py")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to add context: {str(e)}")
            raise

    def _render_memory_controls(self):
        """Render memory management controls."""
        st.markdown("### ⚙️ Memory Controls")
        
        with stylable_container(
            key="memory_controls",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    width: 100% !important;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """
        ):
            # Initialize retention days in session state
            if 'retention_days' not in st.session_state:
                st.session_state.retention_days = 30
            
            # Retention Settings first to ensure value is set
            st.markdown("#### Retention Settings")
            current_retention = st.slider(
                "Keep memories for (days)",
                min_value=0,
                max_value=365,
                value=st.session_state.retention_days,
                key="retention_slider",
                help="Set to 0 to clean up all memories"
            )
            # Update session state when slider changes
            st.session_state.retention_days = current_retention

            # Memory Operations section
            st.markdown("#### Memory Operations")
            
            # Add space before first button
            add_vertical_space(1)
            
            # Refresh button row
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_button"):
                with st.spinner("Refreshing memories..."):
                    memories = asyncio.run(self.memory_manager.get_all_long_term_memories())
                    st.session_state.long_term_memories = sorted(
                        memories,
                        key=lambda x: datetime.fromisoformat(x.get('created_at', '1970-01-01')),
                        reverse=True
                    )
                    st.rerun()
            
            # Space between buttons
            add_vertical_space(1)
            
            # Clean Up button row
            cleanup_clicked = st.button("🧹 Clean Up", use_container_width=True, key="cleanup_button")
            if cleanup_clicked:
                st.session_state.show_cleanup_confirm = True
            
            # Space between buttons
            add_vertical_space(1)
            
            # Import button row
            import_clicked = st.button("📥 Import", use_container_width=True, key="import_button")
            if import_clicked:
                st.session_state.show_import = True

            # Space between buttons
            add_vertical_space(1)
            
            # Export button row
            export_clicked = st.button("📤 Export", use_container_width=True, key="export_button")
            if export_clicked:
                st.session_state.show_export = True
            
            # Space after buttons
            add_vertical_space(1)

            # Show export section if export clicked
            if st.session_state.get('show_export', False):
                with st.expander("Export Memories", expanded=True):
                    # Export format selection
                    export_format = st.selectbox(
                        "Export Format",
                        options=["JSON", "CSV", "PDF"],
                        key="export_format"
                    )
                    
                    if st.button("Download Memories", key="download_button"):
                        try:
                            # Get all memories for export
                            memories = asyncio.run(self.memory_manager.get_all_long_term_memories())
                            if not memories:
                                st.warning("No memories to export")
                                return
                                
                            # Prepare export data
                            export_data = {
                                "memories": memories,
                                "export_info": {
                                    "timestamp": datetime.now().isoformat(),
                                    "format_version": "1.0",
                                    "session_id": st.session_state.session_id
                                }
                            }
                            
                            # Generate filename with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"memories_export_{timestamp}.{export_format.lower()}"
                            
                            # Convert to selected format and offer download
                            if export_format == "JSON":
                                output = json.dumps(export_data, indent=2)
                                st.download_button(
                                    "📥 Download JSON",
                                    output,
                                    file_name=filename,
                                    mime="application/json"
                                )
                            elif export_format == "CSV":
                                # Convert memories to pandas DataFrame
                                df = pd.DataFrame(memories)
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    file_name=filename,
                                    mime="text/csv"
                                )
                            elif export_format == "PDF":
                                buffer = io.BytesIO()
                                doc = SimpleDocTemplate(
                                    buffer,
                                    pagesize=letter,
                                    rightMargin=72,
                                    leftMargin=72,
                                    topMargin=72,
                                    bottomMargin=72
                                )
                                
                                # Get the styles
                                styles = getSampleStyleSheet()
                                story = []
                                
                                # Add title
                                story.append(Paragraph("Memory Export", styles["Title"]))
                                story.append(Spacer(1, 12))
                                
                                # Add memories
                                for memory in memories:
                                    # Add timestamp
                                    date_str = memory.get('created_at', 'Unknown date')
                                    story.append(Paragraph(f"Created: {date_str}", styles["Heading2"]))
                                    
                                    # Add content
                                    if 'content' in memory:
                                        story.append(Paragraph(memory['content'], styles["Normal"]))
                                    
                                    # Add spacing between memories
                                    story.append(Spacer(1, 12))
                                
                                # Build PDF
                                doc.build(story)
                                
                                # Offer download
                                st.download_button(
                                    "📥 Download PDF",
                                    buffer.getvalue(),
                                    file_name=filename,
                                    mime="application/pdf"
                                )
                                
                        except Exception as e:
                            st.error(f"Error exporting memories: {str(e)}")

            # Show import section if import clicked
            if st.session_state.get('show_import', False):
                with st.expander("Import Memories", expanded=True):
                    uploaded_file = st.file_uploader(
                        "Upload exported memories file",
                        type=['json'],
                        key="memory_import",
                        help="Upload a previously exported memories JSON file"
                    )
                    
                    if uploaded_file is not None:
                        try:
                            # Read and parse the JSON file
                            content = uploaded_file.getvalue().decode('utf-8')
                            imported_data = json.loads(content)
                            
                            # Validate the imported data structure
                            if not isinstance(imported_data, dict) or 'memories' not in imported_data:
                                st.error("Invalid memory file format - Please upload a properly formatted memory export file")
                                return
                                
                            memory_list = imported_data['memories']
                            if not memory_list:
                                st.warning("No memories found in import file")
                                return
                                
                            import_confirm = st.checkbox(
                                f"Confirm import of {len(memory_list)} memories?",
                                key="import_confirm"
                            )
                            
                            if import_confirm:
                                with st.spinner("Importing memories..."):
                                    imported_count = 0
                                    for memory in memory_list:
                                        try:
                                            # Validate memory structure
                                            if not isinstance(memory, dict) or 'content' not in memory:
                                                continue
                                                
                                            # Generate new key for imported memory
                                            import_key = f"{st.session_state.session_id}_imported_{imported_count}"
                                            
                                            # Add import metadata
                                            memory['metadata'] = memory.get('metadata', {})
                                            memory['metadata'].update({
                                                'imported_at': datetime.now().isoformat(),
                                                'original_key': memory.get('key'),
                                                'import_session': st.session_state.session_id
                                            })
                                            
                                            # Save to database
                                            self.db.save_memory(
                                                import_key,
                                                memory['content'],
                                                None,  # No embedding for imported memories
                                                memory['metadata']
                                            )
                                            imported_count += 1
                                            
                                        except Exception as e:
                                            print(f"Error importing memory: {str(e)}")
                                            continue
                                    
                                    if imported_count > 0:
                                        st.success(f"Successfully imported {imported_count} memories")
                                        # Refresh the memory list
                                        self._load_long_term_memories()
                                        # Clear import state
                                        del st.session_state.show_import
                                        st.rerun()
                                    else:
                                        st.warning("No valid memories found to import")
                                        
                        except json.JSONDecodeError:
                            st.error("Invalid JSON file format - Please check the file contents")
                        except Exception as e:
                            st.error(f"Error importing memories: {str(e)}")

            # Show cleanup confirmation outside columns
            if st.session_state.get('show_cleanup_confirm', False):
                # Customize confirmation message based on retention days
                if st.session_state.retention_days == 0:
                    confirm_message = "Confirm deletion of ALL memories"
                else:
                    confirm_message = f"Confirm cleanup of memories older than {st.session_state.retention_days} days"
                    
                confirm_cleanup = st.checkbox(
                    confirm_message,
                    key="cleanup_confirm"
                )
                
                if confirm_cleanup:
                    try:
                        memories = self.db.get_all_long_term_memories(st.session_state.session_id)
                        current_time = datetime.now()
                        deleted_count = 0
                        
                        for memory in memories:
                            try:
                                if st.session_state.retention_days == 0:
                                    # Delete all memories if retention is 0
                                    if memory.get('key'):
                                        self.db.delete_memory(memory['key'])
                                        deleted_count += 1
                                else:
                                    # Normal cleanup based on age
                                    memory_time = datetime.fromisoformat(memory.get('created_at', '1970-01-01'))
                                    age_days = (current_time - memory_time).days
                                    
                                    if age_days > st.session_state.retention_days:
                                        if memory.get('key'):
                                            self.db.delete_memory(memory['key'])
                                            deleted_count += 1
                            except Exception as e:
                                print(f"Error processing memory during cleanup: {str(e)}")
                                continue
                        
                        # Reload memories after cleanup
                        self._load_long_term_memories()
                        
                        # Show appropriate success message
                        if st.session_state.retention_days == 0:
                            if deleted_count > 0:
                                st.success(f"Successfully deleted all {deleted_count} memories")
                            else:
                                st.info("No memories to delete")
                        else:
                            if deleted_count > 0:
                                st.success(f"Cleaned up {deleted_count} old memories")
                            else:
                                st.info("No memories needed cleanup")
                        
                        # Clear confirmation state
                        if 'show_cleanup_confirm' in st.session_state:
                            del st.session_state.show_cleanup_confirm
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during cleanup: {str(e)}")
                        print(f"Cleanup error: {str(e)}")
                else:
                    st.info("Please confirm cleanup operation")

    def _render_memory_stats(self):
        """Render enhanced memory statistics."""
        st.markdown("### 📊 Memory Analytics")
        
        # Get current memory stats
        memory_stats = self._calculate_memory_stats()
        
        with stylable_container(
            key="memory_analytics",
            css_styles="""
                {
                    
                }
            """
        ):
            # Memory metrics
            metrics_grid = grid(2, vertical_align="center")
            
            with metrics_grid.container():
                st.metric(
                    "Total Memories",
                    memory_stats['total_count'],  # Changed from 'total_memories' to 'total_count'
                    delta=memory_stats['recent_count'],
                    delta_color="normal"
                )
            
            with metrics_grid.container():
                st.metric(
                    "Memory Age",
                    f"{memory_stats['avg_age']:.1f} days",
                    help="Average age of memories"
                )

            # Memory usage chart
            self._render_memory_usage_chart(memory_stats.get('usage_data', {}))
            
            # Memory age distribution
            self._render_age_distribution_chart(memory_stats.get('age_distribution', {}))

    def _calculate_memory_stats(self) -> Dict[str, Any]:
        """Calculate memory statistics with proper data retrieval."""
        try:
            # Ensure memories are loaded
            if not hasattr(st.session_state, 'long_term_memories'):
                self._load_long_term_memories()

            memories = st.session_state.long_term_memories
                
            # Get both memory types
            long_term_memories = st.session_state.long_term_memories
            short_term_messages = self.memory_manager.get_recent_messages()
            current_time = datetime.now()
            
            # Basic counts
            total_memories = len(memories)
            
            if not total_memories:
                return {
                    'total_count': 0,
                    'active_count': 0,
                    'avg_age': 0,
                    'recent_count': 0,
                    'usage_data': {},
                    'age_distribution': {}
                }

            # Calculate age statistics
            ages = []
            recent_count = 0
            active_count = 0
            age_distribution = {'0-7 days': 0, '8-30 days': 0, '31-90 days': 0, '90+ days': 0}
            
            for memory in memories:
                try:
                    created = datetime.fromisoformat(memory['created_at'])
                    age = (current_time - created).days
                    ages.append(age)
                    
                    # Update age distribution
                    if age <= 7:
                        age_distribution['0-7 days'] += 1
                        active_count += 1
                        if age <= 1:
                            recent_count += 1
                    elif age <= 30:
                        age_distribution['8-30 days'] += 1
                    elif age <= 90:
                        age_distribution['31-90 days'] += 1
                    else:
                        age_distribution['90+ days'] += 1
                        
                except (ValueError, KeyError):
                    continue
            
            # Calculate usage data (last 7 days)
            usage_data = {}
            for i in range(7):
                date = (current_time - timedelta(days=i)).strftime('%Y-%m-%d')
                usage_data[date] = len([
                    m for m in memories 
                    if datetime.fromisoformat(m['created_at']).strftime('%Y-%m-%d') == date
                ])
            
            return {
                'total_count': total_memories,
                'active_count': active_count,
                'avg_age': sum(ages) / len(ages) if ages else 0,
                'recent_count': recent_count,
                'usage_data': usage_data,
                'age_distribution': age_distribution
            }
            
        except Exception as e:
            print(f"Error calculating memory stats: {str(e)}")
            return {
                'total_count': 0,
                'active_count': 0,
                'avg_age': 0,
                'recent_count': 0,
                'usage_data': {},
                'age_distribution': {}
            }

    def _render_memory_usage_chart(self, usage_data: Dict[str, int]):
        """Render memory usage trend chart."""
        fig = go.Figure(data=[
            go.Scatter(
                x=list(usage_data.keys()),
                y=list(usage_data.values()),
                mode='lines+markers',
                line=dict(color='#4287f5', width=2),
                marker=dict(color='#4287f5', size=8)
            )
        ])
        
        fig.update_layout(
            title="Memory Usage Trend",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    def _render_age_distribution_chart(self, age_distribution: Dict[str, int]):
        """Render memory age distribution chart."""
        if not age_distribution:
            return
            
        fig = go.Figure(data=[
            go.Bar(
                x=list(age_distribution.keys()),
                y=list(age_distribution.values()),
                marker_color='#4287f5'
            )
        ])
        
        fig.update_layout(
            title="Memory Age Distribution",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    def _render_memory_overview(self):
        """Render memory overview with metrics."""
        st.markdown("## 🧠 Memory Overview")
        add_vertical_space(1)

        # Calculate stats once
        stats = self._calculate_memory_stats()
        
        # Display metrics in grid
        metrics_grid = grid(3, vertical_align="center")
        
        with metrics_grid.container():
            with stylable_container(
                key="metric_total",
                css_styles="""
                    {
                        
                    }
                """
            ):
                st.metric(
                    "Total Memories",
                    stats['total_count'],
                    delta=f"+{stats['recent_count']}" if stats['recent_count'] > 0 else None,
                    help="Total number of stored memories"
                )

        with metrics_grid.container():
            with stylable_container(
                key="metric_active",
                css_styles="""
                    {
                        
                    }
                """
            ):
                st.metric(
                    "Active Memories",
                    stats['active_count'],
                    help="Memories from the last 7 days"
                )

        with metrics_grid.container():
            with stylable_container(
                key="metric_age",
                css_styles="""
                    {
                        
                    }
                """
            ):
                st.metric(
                    "Average Age",
                    f"{stats['avg_age']:.1f} days",
                    help="Average age of memories"
                )

        style_metric_cards()
    
    def _render_short_term_memory(self):
        """Render short-term memory messages with improved retrieval."""
        try:
            # Get recent messages from memory manager
            messages = self.memory_manager.get_recent_messages()
            
            if not messages:
                st.info("No messages in short-term memory.")
                return
                
            # Display messages in chronological order
            for idx, msg in enumerate(messages):
                with stylable_container(
                    key=f"short_term_msg_{msg.get('timestamp', '')}_{idx}",
                    css_styles=f"""
                        {{
                            background: {"rgba(255, 75, 75, 0.1)" if msg['role'] == MESSAGE_TYPES["USER"] else "rgba(66, 135, 245, 0.1)"};
                            border-radius: 0.5rem;
                            padding: 1rem;
                            margin-bottom: 0.5rem;
                            border-left: 3px solid {"#FF4B4B" if msg['role'] == MESSAGE_TYPES["USER"] else "#4287f5"};
                        }}
                    """
                ):
                    # Display role and timestamp
                    try:
                        timestamp = datetime.fromisoformat(msg['timestamp'])
                        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        role_display = "👤 User" if msg['role'] == MESSAGE_TYPES["USER"] else "🤖 Assistant"
                        st.markdown(f"**{role_display}** - {formatted_time}")
                    except (ValueError, TypeError):
                        role_display = "👤 User" if msg['role'] == MESSAGE_TYPES["USER"] else "🤖 Assistant"
                        st.markdown(f"**{role_display}**")

                    # Display message content
                    st.markdown(msg['content'])

                    # Action buttons with unique keys using index
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button("📋 Copy", key=f"copy_st_{msg.get('timestamp', '')}_{idx}"):
                            st.write_clipboard(msg['content'])
                            st.success("Copied to clipboard!")
                    
                    with cols[1]:
                        if st.button("🔍 Details", key=f"details_st_{msg.get('timestamp', '')}_{idx}"):
                            with st.expander("Message Details"):
                                st.json({k: v for k, v in msg.items() if k != 'content'})
                    
                    with cols[2]:
                        if st.button("🗑️ Delete", key=f"delete_st_{msg.get('timestamp', '')}_{idx}"):
                            if msg.get('id'):
                                self.memory_manager.delete_message(msg['id'])
                                st.success("Message deleted!")
                                st.rerun()

        except Exception as e:
            print(f"Error displaying short-term memory: {str(e)}")
            st.error("Failed to display messages. Please try refreshing the page.")

    def _handle_memory_cleanup(self):
        """Handle memory cleanup process."""
        try:
            retention_days = st.session_state.get("retention_days", 30)
            
            # Get current memories
            current_memories = st.session_state.long_term_memories
            current_time = datetime.now()
            
            # Filter memories based on retention period
            retained_memories = [
                memory for memory in current_memories
                if (current_time - datetime.fromisoformat(memory['created_at'])).days <= retention_days
            ]
            
            # Update database and session state
            removed_count = len(current_memories) - len(retained_memories)
            st.session_state.long_term_memories = retained_memories
            
            if removed_count > 0:
                st.success(f"Removed {removed_count} old memories")
            else:
                st.info("No memories needed cleanup")
                
            st.session_state.last_memory_update = current_time.isoformat()
            
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")

    def _handle_memory_export(self, format_type: str):
        """Handle memory export process."""
        try:
            memories = st.session_state.long_term_memories
            
            if not memories:
                st.warning("No memories to export")
                return
                
            if format_type == "JSON":
                export_data = json.dumps(memories, indent=2)
                mime_type = "application/json"
                file_extension = "json"
            elif format_type == "CSV":
                df = pd.DataFrame(memories)
                export_data = df.to_csv(index=False)
                mime_type = "text/csv"
                file_extension = "csv"
            else:  # TXT
                export_data = "\n\n".join(
                    f"Memory ID: {m.get('key', 'N/A')}\n"
                    f"Created: {m.get('created_at', 'N/A')}\n"
                    f"Content: {m.get('content', 'N/A')}"
                    for m in memories
                )
                mime_type = "text/plain"
                file_extension = "txt"
            
            # Generate download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Export",
                data=export_data,
                file_name=f"memory_export_{timestamp}.{file_extension}",
                mime=mime_type
            )
            
        except Exception as e:
            st.error(f"Error during export: {str(e)}")