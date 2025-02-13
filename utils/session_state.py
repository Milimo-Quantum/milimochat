import streamlit as st
from typing import Any, Dict, Optional, TypeVar, Generic, Union, List
from datetime import datetime
import uuid
import json

from config import SessionKeys, PERSONALITY_PRESETS, MESSAGE_TYPES

T = TypeVar('T')

class SessionStateManager(Generic[T]):
    """Generic session state manager with type safety."""
    
    def __init__(self, key: str, default_value: T):
        self.key = key
        self.default_value = default_value

    def get(self) -> T:
        """Get value from session state with improved None handling."""
        if self.key not in st.session_state:
            st.session_state[self.key] = self.default_value
        return st.session_state.get(self.key, self.default_value)

    def set(self, value: T):
        """Set value in session state with validation."""
        if value is None:
            value = self.default_value
        st.session_state[self.key] = value

    def delete(self):
        """Remove value from session state."""
        if self.key in st.session_state:
            del st.session_state[self.key]

class MessageManager:
    """Enhanced message manager with RAG support."""
    
    def __init__(self):
        # Initialize with empty list as default value
        self.messages = SessionStateManager[List[Dict[str, Any]]](SessionKeys.MESSAGES, [])
        # Add vector state management
        self.vector_ids = SessionStateManager[List[str]]("vector_ids", [])
        
    def validate_role(self, role: str) -> str:
        """Validate and normalize message role with improved handling."""
        if not role:
            raise ValueError("Message role cannot be empty")
            
        # Convert role to proper MESSAGE_TYPES constant
        normalized_role = role.lower()
        for role_type, role_value in MESSAGE_TYPES.items():
            if normalized_role == role_value.lower():
                return role_value
                
        raise ValueError(f"Invalid message role: {role}")

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message structure and content with enhanced validation."""
        if not isinstance(message, dict):
            return False
            
        required_fields = {'role', 'content', 'timestamp'}
        
        # Check for required fields
        if not all(field in message for field in required_fields):
            return False
            
        try:
            # Validate role
            if not message['role'] or message['role'] not in MESSAGE_TYPES.values():
                return False
                
            # Validate content
            if not isinstance(message['content'], str) or not message['content'].strip():
                return False
                
            # Validate timestamp
            try:
                datetime.fromisoformat(message['timestamp'])
            except (ValueError, TypeError):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating message: {str(e)}")
            return False

    def save_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a message with RAG metadata and validation."""
        # Validate required fields
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")
            
        # Validate and normalize metadata
        metadata = metadata or {}
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
            
        try:
            datetime.fromisoformat(metadata['timestamp'])
        except (ValueError, TypeError):
            metadata['timestamp'] = datetime.now().isoformat()
            
        try:
            validated_role = self.validate_role(role)
            current_time = datetime.now().isoformat()
            
            # Handle vector IDs in metadata
            if metadata and 'vector_id' in metadata:
                vector_ids = self.vector_ids.get()
                vector_ids.append(metadata['vector_id'])
                self.vector_ids.set(vector_ids)
            
            message = {
                'role': validated_role,
                'content': content.strip(),
                'timestamp': current_time,
                'metadata': metadata or {}
            }
            
            if not self.validate_message(message):
                raise ValueError("Invalid message structure")
                
            messages = self.messages.get()
            if messages is None:
                messages = []
            messages.append(message)
            self.messages.set(messages)
            
        except Exception as e:
            raise Exception(f"Failed to save message: {str(e)}")

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages with vector-aware retrieval."""
        try:
            messages = self.messages.get()
            if messages is None:
                messages = []
                self.messages.set(messages)
                
            # Sort by timestamp and handle vector relationships
            messages.sort(key=lambda x: x.get('timestamp', ''))
            
            # Enrich messages with vector context
            for message in messages:
                if message.get('metadata', {}).get('vector_id'):
                    vector_ids = self.vector_ids.get()
                    if message['metadata']['vector_id'] in vector_ids:
                        message['metadata']['has_vector'] = True
            
            if limit and limit > 0:
                return messages[-limit:]
            return messages
            
        except Exception as e:
            print(f"Error retrieving messages: {str(e)}")
            return []

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get messages filtered by role with vector awareness."""
        validated_role = self.validate_role(role)
        return [
            msg for msg in self.get_messages() 
            if msg['role'] == validated_role
        ]

    def delete_message(self, timestamp: str) -> bool:
        """Delete a message and associated vectors."""
        try:
            messages = self.messages.get()
            initial_length = len(messages)
            
            # Validate timestamp
            datetime.fromisoformat(timestamp)
            
            # Find message and handle vector cleanup
            message_to_delete = None
            for msg in messages:
                if msg.get('timestamp') == timestamp:
                    message_to_delete = msg
                    break
                    
            if message_to_delete:
                # Clean up vector IDs
                if message_to_delete.get('metadata', {}).get('vector_id'):
                    vector_ids = self.vector_ids.get()
                    vector_ids.remove(message_to_delete['metadata']['vector_id'])
                    self.vector_ids.set(vector_ids)
            
            # Filter out the message
            updated_messages = [
                msg for msg in messages 
                if msg.get('timestamp') != timestamp
            ]
            
            if len(updated_messages) < initial_length:
                self.messages.set(updated_messages)
                return True
                
            return False
        except ValueError:
            print(f"Invalid timestamp format: {timestamp}")
            return False
        except Exception as e:
            print(f"Error deleting message: {str(e)}")
            return False

    def get_message_count(self, role: Optional[str] = None) -> int:
        """Get message count with role filtering."""
        if role:
            try:
                validated_role = self.validate_role(role)
                return len(self.get_messages_by_role(validated_role))
            except ValueError:
                return 0
        return len(self.get_messages())

    def clear_messages(self):
        """Clear all messages and vector references."""
        self.messages.set([])
        self.vector_ids.set([])

class ChatSessionState:
    """Manages chat-specific session state with RAG support."""
    
    def __init__(self):
        self.message_manager = MessageManager()
        self.current_model = SessionStateManager[str](SessionKeys.CURRENT_MODEL, "llama2")
        self.personality = SessionStateManager[str](SessionKeys.PERSONALITY, "Professional")
        self.tone = SessionStateManager[str](SessionKeys.TONE, "Balanced")
        self.creativity = SessionStateManager[str](SessionKeys.CREATIVITY, "Balanced")
        self.memory_enabled = SessionStateManager[bool](SessionKeys.MEMORY_ENABLED, True)
        self.user_settings = SessionStateManager[dict](SessionKeys.USER_SETTINGS, {})
        
        # RAG-specific state management
        self.custom_prompts = SessionStateManager[dict]("custom_personality_prompts", {})
        self.vector_context = SessionStateManager[dict]("vector_context", {})
        self.document_chunks = SessionStateManager[dict]("document_chunks", {})
        
        # Ensure session is initialized
        self.initialize_session()
        
    def initialize_session(self):
        """Initialize a new chat session with RAG support."""
        try:
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_start = datetime.now().isoformat()
                
                # Initialize message list
                if SessionKeys.MESSAGES not in st.session_state:
                    st.session_state[SessionKeys.MESSAGES] = []
                
                # Initialize vector context
                if "vector_context" not in st.session_state:
                    st.session_state.vector_context = {}
                
                # Initialize document chunks
                if "document_chunks" not in st.session_state:
                    st.session_state.document_chunks = {}
                
                # Initialize default user settings
                if not self.user_settings.get():
                    self.user_settings.set({
                        'theme': 'dark',
                        'language': 'en',
                        'notifications_enabled': True,
                        'auto_cleanup': True,
                        'max_memory_messages': 50,
                        'vector_similarity_threshold': 0.7,
                        'chunk_size': 500,
                        'chunk_overlap': 50
                    })
                
            self.update_session_activity()
        except Exception as e:
            st.error(f"Failed to initialize session: {str(e)}")
            # Provide default values for critical components
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            if SessionKeys.MESSAGES not in st.session_state:
                st.session_state[SessionKeys.MESSAGES] = []

    def get_custom_prompt(self, personality_name: str) -> Optional[str]:
        """Get custom prompt for a specific personality."""
        custom_prompts = self.custom_prompts.get()
        return custom_prompts.get(personality_name)

    def set_custom_prompt(self, personality_name: str, prompt: str) -> None:
        """Set custom prompt for a specific personality."""
        custom_prompts = self.custom_prompts.get()
        if prompt and prompt.strip():
            custom_prompts[personality_name] = prompt.strip()
        elif personality_name in custom_prompts:
            del custom_prompts[personality_name]
        self.custom_prompts.set(custom_prompts)
        
        # Update in user settings for persistence
        settings = self.user_settings.get()
        if 'custom_personality_prompts' not in settings:
            settings['custom_personality_prompts'] = {}
        settings['custom_personality_prompts'] = custom_prompts
        self.user_settings.set(settings)

    def get_personality_prompt(self, personality_name: Optional[str] = None) -> str:
        """Get the current prompt with RAG enhancement."""
        if personality_name is None:
            personality_name = self.personality.get()
            
        # Try to get custom prompt first
        custom_prompt = self.get_custom_prompt(personality_name)
        if custom_prompt:
            return custom_prompt
            
        # Fall back to default prompt
        personality = PERSONALITY_PRESETS.get(personality_name)
        if personality:
            return personality['system_prompt'](None)
        return PERSONALITY_PRESETS['Professional']['system_prompt'](None)

    def update_user_setting(self, key: str, value: Any):
        """Update a specific user setting."""
        settings = self.user_settings.get()
        settings[key] = value
        self.user_settings.set(settings)
        
        # Sync custom prompts if updated
        if key == "custom_personality_prompts":
            self.custom_prompts.set(value)

    def get_user_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific user setting."""
        settings = self.user_settings.get()
        return settings.get(key, default)

    def save_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Save a message with vector context."""
        self.message_manager.save_message(role, content, metadata)
        self.update_session_activity()

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages with vector-aware retrieval."""
        return self.message_manager.get_messages(limit)

    def clear_messages(self):
        """Clear all messages and vector context."""
        self.message_manager.clear_messages()
        self.vector_context.set({})
        self.document_chunks.set({})
        self.update_session_activity()

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get messages filtered by role."""
        return self.message_manager.get_messages_by_role(role)

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session with vector stats."""
        return {
            'session_id': st.session_state.session_id,
            'start_time': st.session_state.session_start,
            'last_activity': st.session_state.get('last_activity'),
            'message_count': self.message_manager.get_message_count(),
            'user_messages': self.message_manager.get_message_count(MESSAGE_TYPES["USER"]),
            'assistant_messages': self.message_manager.get_message_count(MESSAGE_TYPES["ASSISTANT"]),
            'current_model': self.current_model.get(),
            'personality': self.personality.get(),
            'memory_enabled': self.memory_enabled.get(),
            'vector_count': len(self.vector_context.get()),
            'document_chunks': len(self.document_chunks.get())
        }

    def update_session_activity(self):
        """Update session last activity timestamp."""
        st.session_state.last_activity = datetime.now().isoformat()

    def clear_session(self):
        """Clear all session data."""
        settings = self.user_settings.get()
        session_id = st.session_state.session_id
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Restore critical values
        st.session_state.session_id = session_id
        self.user_settings.set(settings)
        self.initialize_session()

    def is_session_active(self, timeout_minutes: int = 30) -> bool:
        """Check if the session is still active based on timeout."""
        if 'last_activity' not in st.session_state:
            return False
        
        last_activity = datetime.fromisoformat(st.session_state.last_activity)
        time_diff = datetime.now() - last_activity
        return time_diff.total_seconds() < timeout_minutes * 60

    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data including vector context."""
        return {
            'session_info': self.get_session_info(),
            'messages': self.message_manager.get_messages(),
            'user_settings': self.user_settings.get(),
            'vector_context': self.vector_context.get(),
            'document_chunks': self.document_chunks.get(),
            'export_time': datetime.now().isoformat()
        }

    def import_session_data(self, data: Dict[str, Any]):
        """Import session data with vector context."""
        try:
            # Validate data structure
            required_keys = ['session_info', 'messages', 'user_settings']
            if not all(key in data for key in required_keys):
                raise ValueError("Invalid session data format")
            
            # Update messages with proper role validation
            for message in data['messages']:
                self.message_manager.save_message(
                    message['role'],
                    message['content'],
                    message.get('metadata', {})
                )
            
            # Update user settings
            self.user_settings.set(data['user_settings'])
            
            # Update vector context if present
            if 'vector_context' in data:
                self.vector_context.set(data['vector_context'])
            
            # Update document chunks if present
            if 'document_chunks' in data:
                self.document_chunks.set(data['document_chunks'])
            
            # Update session info
            st.session_state.session_start = data['session_info']['start_time']
            self.update_session_activity()
            
        except Exception as e:
            raise Exception(f"Failed to import session data: {str(e)}")