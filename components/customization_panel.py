import streamlit as st
from typing import Optional, Dict, Any
import asyncio

from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards

from utils.ollama_client import OllamaClient
from styles.theme import section_header, card_container
from config import (
    PERSONALITY_PRESETS,
    TONE_PRESETS,
    CREATIVITY_LEVELS,
    SessionKeys
)

import streamlit as st
from typing import Optional, Dict, Any
import asyncio

from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

from utils.ollama_client import OllamaClient
from styles.theme import section_header, card_container
from config import SessionKeys

class CustomizationPanel:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables for customization."""
        if SessionKeys.CURRENT_MODEL not in st.session_state:
            st.session_state[SessionKeys.CURRENT_MODEL] = None

    def render(self):
        """Render the customization panel in the sidebar."""
        with st.sidebar:
            self._render_model_selector()
            add_vertical_space(2)
            
            self._render_memory_controls()
            add_vertical_space(2)
            
            self._render_action_buttons()

    async def _load_selected_model(self, model_name: str):
        """Load the selected model."""
        try:
            with st.spinner(f"Loading {model_name}..."):
                await self.ollama_client.load_model(model_name)
            st.session_state[SessionKeys.CURRENT_MODEL] = model_name
            st.success(f"Model {model_name} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")

    def _render_model_selector(self):
        """Render the model selection section."""
        try:
            available_models = self.ollama_client.list_models()
            model_names = [model["name"] for model in available_models]
            
            with stylable_container(
                key="model_selector",
                css_styles="""
                    {
                        
                    }
                """
            ):
                selected_model = st.selectbox(
                    "Select Model",
                    options=model_names,
                    index=model_names.index(st.session_state[SessionKeys.CURRENT_MODEL]) if st.session_state[SessionKeys.CURRENT_MODEL] in model_names else 0,
                    format_func=lambda x: x.split(':')[0].title(),
                    help="Choose a model to use for chat"
                )
                
                if selected_model != st.session_state[SessionKeys.CURRENT_MODEL]:
                    asyncio.run(self._load_selected_model(selected_model))
                    st.rerun()
            
            # Model info
            if st.session_state[SessionKeys.CURRENT_MODEL]:
                model_info = self.ollama_client.get_model_info(st.session_state[SessionKeys.CURRENT_MODEL])
                
                # Keep existing detailed info
                with st.expander("Model Details"):
                    st.json(model_info.get("details", {}))
                    if 'model_info' in model_info:
                        st.markdown("### Advanced Parameters")
                        st.json(model_info['model_info'])
        
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

    def _render_memory_controls(self):
        """Render memory control settings."""
        section_header("Memory", icon="🧠")
        
        with stylable_container(
            key="memory_controls",
            css_styles="""
                {
                    
                }
            """
        ):
            st.toggle(
                "Enable Memory",
                value=st.session_state.get(SessionKeys.MEMORY_ENABLED, True),
                key=f"{SessionKeys.MEMORY_ENABLED}_toggle",
                help="Toggle context memory for more coherent conversations"
            )

    def _render_action_buttons(self):
        """Render action buttons."""
        with stylable_container(
            key="action_buttons",
            css_styles="""
                {
                    
                }
            """
        ):
            if st.button("Clear Chat", use_container_width=True):
                st.session_state[SessionKeys.MESSAGES] = []
                st.rerun()
            