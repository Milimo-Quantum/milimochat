import tracemalloc
tracemalloc.start()
import streamlit as st
import uuid
from datetime import datetime
import base64
import os
from pathlib import Path
import json
import time
import pyperclip
from typing import Any, Dict, Optional
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stoggle import stoggle
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

# Must be the first Streamlit command
st.set_page_config(
    page_title="Milimo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components.chat_interface import ChatInterface
from components.customization_panel import CustomizationPanel
from components.history_analytics import HistoryAnalytics
from components.memory_dashboard import MemoryDashboard
from styles.theme import initialize_theme
from styles.custom_components import StyledComponents
from utils.session_state import ChatSessionState
from services.chat_service import ChatService
from services.export_service import ExportService
from config import SessionKeys
from config import (
    SessionKeys,
    PERSONALITY_PRESETS,
    TONE_PRESETS,
    CREATIVITY_LEVELS
)

# Handle Enter key press for message submission
st.markdown("""
    <script>
        // Auto-focusing on the input field
        const textArea = document.querySelector('textarea[data-testid="stTextArea"]');
        if (textArea) {
            textArea.focus();
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey) {
                const textArea = document.querySelector('textarea[data-testid="stTextArea"]');
                if (textArea && document.activeElement === textArea) {
                    e.preventDefault();
                    const submitButton = document.querySelector('button[kind="primary"]');
                    if (submitButton) {
                        submitButton.click();
                    }
                }
            }
        });
    </script>
""", unsafe_allow_html=True)

# Set custom styles
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(
                165deg,
                #000428 0%,
                #004e92 100%
            ) !important;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(0, 78, 146, 0.45);
            backdrop-filter: blur(10px);
        }

        /* Input area styling */
        .stTextArea > div > div > textarea {
            background-color: rgba(0, 78, 146, 0.8);
            border-color: rgba(0, 200, 255, 0.2);
            color: white;
            border-radius: 4px;
            min-height: 100px;
            padding: 0.5rem;
        }

        .stTextArea > div > div > textarea:focus {
            border-color: rgba(0, 200, 255, 0.5);
            box-shadow: 0 0 0 1px rgba(0, 200, 255, 0.5);
        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 4px;
            padding: 0.5rem;
            background-color: rgba(0, 200, 255, 0.8);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: rgba(0, 200, 255, 1);
            transform: translateY(-2px);
        }

        /* Hide Streamlit branding */
        #MainMenu, footer, header {
            visibility: hidden;
        }

        /* Improved scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 78, 146, 0.1);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(0, 200, 255, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 200, 255, 0.5);
        }

        /* Input field styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: rgba(0, 78, 146, 0.8) !important;
            color: rgb(250, 250, 250) !important;
            border-color: rgba(0, 200, 255, 0.2);
            border-radius: 4px;
        }

        /* Input field focus state */
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: rgba(0, 200, 255, 0.5);
            box-shadow: 0 0 0 1px rgba(0, 200, 255, 0.5);
            color: rgb(250, 250, 250) !important;
        }

        /* Selectbox styling */
        .stSelectbox > div > div > select {
            background-color: rgba(0, 78, 146, 0.8) !important;
            color: rgb(250, 250, 250) !important;
            border-color: rgba(0, 200, 255, 0.2);
            border-radius: 4px;
        }

        /* Label text color */
        .stTextInput label,
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label {
            color: rgb(250, 250, 250) !important;
        }

        /* Placeholder text color */
        .stTextInput > div > div > input::placeholder,
        .stNumberInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {
            color: rgba(250, 250, 250, 0.6) !important;
        }

        /* JSON/Code display styling */
        .stTextArea > div > div > textarea[disabled] {
            background-color: rgba(0, 78, 146, 0.8) !important;
            color: rgb(250, 250, 250) !important;
            border-color: rgba(0, 200, 255, 0.2);
            opacity: 1 !important;  /* Prevents the disabled state from making text too transparent */
        }

        /* Ensure text remains visible even when the textarea is disabled */
        .stTextArea > div > div > textarea[disabled]::placeholder {
            color: rgba(250, 250, 250, 0.6) !important;
        }

        /* Style for the text area when it contains code/JSON */
        pre {
            background-color: rgba(0, 78, 146, 0.8) !important;
            color: rgb(250, 250, 250) !important;
            padding: 1rem;
            border-radius: 4px;
            border: 1px solid rgba(0, 200, 255, 0.2);
        }

        code {
            color: rgb(250, 250, 250) !important;
        }
    </style>
""", unsafe_allow_html=True)

class MilimoChat:
    def __init__(self):
        self.session_state = ChatSessionState()
        self.chat_service = ChatService(self.session_state.get_session_info()['session_id'])
        self.styled = StyledComponents()
        
        # Initialize settings manager and load settings
        self.settings_manager = SettingsManager(self.session_state)
        
        # Initialize components with proper session state
        self.chat_interface = ChatInterface(session_state=self.session_state)
        self.customization_panel = CustomizationPanel()
        self.history_analytics = HistoryAnalytics()
        self.memory_dashboard = MemoryDashboard()
        self.export_service = ExportService()
        
        # Apply saved theme if exists
        theme_config = self.session_state.get_user_setting("THEME_CONFIG", None)
        if theme_config:
            self.settings_manager.apply_theme_config(theme_config)

    def initialize_session_state(self):
        """Initialize or update session state with required keys."""
        if not st.session_state.get(SessionKeys.PERSONALITY):
            st.session_state[SessionKeys.PERSONALITY] = "Professional"
        
        if not st.session_state.get(SessionKeys.TONE):
            st.session_state[SessionKeys.TONE] = "Balanced"
        
        if not st.session_state.get(SessionKeys.CREATIVITY):
            st.session_state[SessionKeys.CREATIVITY] = "Balanced"

    def main(self):
        """Main application entry point."""
        # Initialize theme
        initialize_theme()

        # Initialize session state for message input
        if 'message_input' not in st.session_state:
            st.session_state.message_input = ""

        # Create main layout
        if "current_view" not in st.session_state:
            st.session_state.current_view = "chat"

        # Initialize theme state
        if "theme" not in st.session_state:
            st.session_state.theme = "Dark"

        # Sidebar navigation
        with st.sidebar:
            self._render_navigation()
            self.customization_panel.render()

        # Main content area
        if st.session_state.current_view == "chat":
            self.render_chat_view()
        elif st.session_state.current_view == "history":
            self.render_history_view()
        elif st.session_state.current_view == "memory":
            self.render_memory_view()
        elif st.session_state.current_view == "settings":
            self.render_settings_view()

    def _render_navigation(self):
        """Render navigation menu."""
        try:
            with open("assets/logo.png", "rb") as f:
                logo_bytes = f.read()
                logo_b64 = base64.b64encode(logo_bytes).decode()
        except Exception as e:
            st.error(f"Error loading logo: {str(e)}")
            logo_b64 = None

        logo_html = f"""
            <div style="text-align: center; padding: 1rem;">
                <img src="data:image/png;base64,{logo_b64}" 
                     style="max-width: 150px; margin-bottom: 1rem;" 
                     alt="Milimo Logo">
                <h2 style="margin: 0; color: white; font-size: 1.5rem;">Milimo</h2>
            </div>
        """ if logo_b64 else """
            <div style="text-align: center; padding: 1rem;">
                <h2 style="margin: 0; color: white; font-size: 1.5rem;">Milimo</h2>
            </div>
        """

        st.markdown(logo_html, unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

        nav_options = {
            "chat": "💬 Chat",
            "history": "📊 History",
            "memory": "🧠 Memory",
            "settings": "⚙️ Settings"
        }

        for view, label in nav_options.items():
            if st.button(
                label,
                key=f"nav_{view}",
                type="secondary" if st.session_state.current_view != view else "primary"
            ):
                st.session_state.current_view = view
                st.rerun()

    def render_chat_view(self):
        """Render the chat interface."""
        # Render the chat interface
        self.chat_interface.render()

    def render_history_view(self):
        """Render the history and analytics view."""
        self.history_analytics.render()

        # Export options
        export_container = st.container()
        with export_container:
            if st.button("Export Chat History", key="export_history_btn"):
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "JSON", "PDF"],
                    key="export_format_select"
                )
                
                if export_format:
                    messages = self.session_state.get_messages()
                    exported_data = self.export_service.export_chat_history(
                        messages,
                        export_format,
                        self.session_state.get_session_info()
                    )
                    
                    st.download_button(
                        f"Download {export_format}",
                        exported_data,
                        file_name=self.export_service.get_export_filename(export_format),
                        mime=self.export_service.get_mime_type(export_format),
                        key="download_export_btn"
                    )

    def render_memory_view(self):
        """Render the memory management view."""
        self.memory_dashboard.render()

    def render_settings_view(self):
        """Render the enhanced settings view."""
        st.title("Settings")
        
        # Initialize settings manager if not already done
        if not hasattr(self, 'settings_manager'):
            self.settings_manager = SettingsManager(self.session_state)
        
        # Get current settings
        current_settings = self.settings_manager.get_all_settings()
        
        # Add tabs for different settings categories
        tabs = st.tabs([
            "🎨 Interface", "⚙️ System", "🤖 Model", "💾 Memory", "🔒 Privacy", 
            "⚡ Performance", "📤 Export", "😊 Personality", "⚙️ Advanced Config"
        ])
        
        # Interface Settings Tab
        with tabs[0]:
            st.subheader("Interface Settings")
            st.markdown("<span style='background-color: gray; padding: 0.5rem; color: black !important;'>Switch your display to dark mode for best results</span>", unsafe_allow_html=True)
            theme_config = current_settings.get("THEME_CONFIG", {})
            
            col1, col2 = st.columns(2)
            with col1:
                new_primary = st.color_picker(
                    "Primary Color",
                    value=theme_config.get("primaryColor", "#FF4B4B")
                )
                new_bg = st.color_picker(
                    "Background Color",
                    value=theme_config.get("backgroundColor", "#0E1117")
                )
            
            with col2:
                new_secondary_bg = st.color_picker(
                    "Secondary Background",
                    value=theme_config.get("secondaryBackgroundColor", "#262730")
                )
                new_text = st.color_picker(
                    "Text Color",
                    value=theme_config.get("textColor", "#FAFAFA")
                )
            
            font_options = ["sans-serif", "serif", "monospace", "system-ui"]
            new_font = st.selectbox(
                "Font Family",
                options=font_options,
                index=font_options.index(theme_config.get("font", "sans-serif"))
            )
            
            if st.button("Update Theme"):
                new_theme = {
                    "primaryColor": new_primary,
                    "backgroundColor": new_bg,
                    "secondaryBackgroundColor": new_secondary_bg,
                    "textColor": new_text,
                    "font": new_font
                }
                self.settings_manager.update_setting("THEME_CONFIG", new_theme)
                st.success("Theme updated successfully!")
                st.rerun()
        
        # System Settings Tab
        with tabs[1]:
            st.subheader("System Settings")
            
            new_title = st.text_input(
                "Page Title",
                value=current_settings.get("PAGE_TITLE", "Milimo Chat")
            )
            new_icon = st.text_input(
                "Page Icon",
                value=current_settings.get("PAGE_ICON", "🤖")
            )
            new_base_url = st.text_input(
                "Ollama Base URL",
                value=current_settings.get("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            
            if st.button("Update System Settings"):
                updates = {
                    "PAGE_TITLE": new_title,
                    "PAGE_ICON": new_icon,
                    "OLLAMA_BASE_URL": new_base_url
                }
                if self.settings_manager.batch_update(updates):
                    st.success("System settings updated successfully!")
                    st.rerun()
        
        # Model Settings Tab
        with tabs[2]:
            st.subheader("Model Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                new_temp = st.slider(
                    "Temperature",
                    0.0, 1.0,
                    value=current_settings.get("temperature", 0.7),
                    step=0.1
                )
                new_top_p = st.slider(
                    "Top P",
                    0.0, 1.0,
                    value=current_settings.get("top_p", 0.9),
                    step=0.1
                )
            
            with col2:
                new_presence_penalty = st.slider(
                    "Presence Penalty",
                    0.0, 2.0,
                    value=current_settings.get("presence_penalty", 0.6),
                    step=0.1
                )
                new_freq_penalty = st.slider(
                    "Frequency Penalty",
                    0.0, 2.0,
                    value=current_settings.get("frequency_penalty", 0.2),
                    step=0.1
                )
            
            if st.button("Update Model Settings"):
                updates = {
                    "temperature": new_temp,
                    "top_p": new_top_p,
                    "presence_penalty": new_presence_penalty,
                    "frequency_penalty": new_freq_penalty
                }
                if self.settings_manager.batch_update(updates):
                    st.success("Model settings updated successfully!")
        
        # Memory Settings Tab
        with tabs[3]:
            st.subheader("Memory Settings")
            
            memory_enabled = st.toggle(
                "Enable Memory",
                value=current_settings.get("memory_enabled", True)
            )
            
            memory_window = st.number_input(
                "Memory Window (messages)",
                min_value=5,
                max_value=1000,
                value=current_settings.get("memory_window", 250)
            )
            
            if st.button("Update Memory Settings"):
                updates = {
                    "memory_enabled": memory_enabled,
                    "memory_window": memory_window
                }
                if self.settings_manager.batch_update(updates):
                    st.success("Memory settings updated successfully!")
        
        # Privacy Settings Tab
        with tabs[4]:
            st.subheader("Privacy Settings")
            
            save_history = st.toggle(
                "Save Chat History",
                value=current_settings.get("save_history", True)
            )
            
            auto_clean = st.toggle(
                "Auto-clean Old Data",
                value=current_settings.get("auto_clean", True)
            )
            
            if st.button("Update Privacy Settings"):
                updates = {
                    "save_history": save_history,
                    "auto_clean": auto_clean
                }
                if self.settings_manager.batch_update(updates):
                    st.success("Privacy settings updated successfully!")
        
        # Performance Settings Tab
        with tabs[5]:
            st.subheader("Performance Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                new_num_ctx = st.number_input(
                    "Context Window",
                    min_value=1024,
                    max_value=32768,
                    value=current_settings.get("num_ctx", 32768)
                )
                new_num_batch = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=2048,
                    value=current_settings.get("num_batch", 512)
                )
            
            with col2:
                new_num_thread = st.number_input(
                    "Number of Threads",
                    min_value=1,
                    max_value=32,
                    value=current_settings.get("num_thread", 16)
                )
            
            if st.button("Update Performance Settings"):
                updates = {
                    "num_ctx": new_num_ctx,
                    "num_batch": new_num_batch,
                    "num_thread": new_num_thread
                }
                if self.settings_manager.batch_update(updates):
                    st.success("Performance settings updated successfully!")
        
        # Export Settings Tab
        with tabs[6]:
            st.subheader("Export Settings")
            
            # Display current settings using st.code instead of st.text_area
            # This provides better visibility and syntax highlighting
            st.code(
                json.dumps(current_settings, indent=2),
                language="json",
                line_numbers=True
            )
            
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("Export Settings", type="primary"):
                    settings_json = json.dumps(current_settings, indent=2)
                    st.download_button(
                        "📥 Download Settings",
                        settings_json,
                        file_name="milimo_settings.json",
                        mime="application/json",
                        key="download_settings_button"
                    )
            
            with export_col2:
                uploaded_file = st.file_uploader(
                    "Import Settings",
                    type=["json"],
                    key="settings_file_uploader",
                    help="Upload a previously exported settings file"
                )
                
                if uploaded_file is not None:
                    try:
                        import_settings = json.loads(uploaded_file.getvalue().decode())
                        if st.button("Apply Imported Settings", type="primary"):
                            if self.settings_manager.batch_update(import_settings):
                                st.success("✅ Settings imported successfully!")
                                time.sleep(1)
                                st.rerun()
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON file. Please check the file format.")
                    except Exception as e:
                        st.error(f"❌ Error importing settings: {str(e)}")
        
        # Personality Settings Tab
        with tabs[7]:
            st.subheader("Personality Settings")
            
            # Current personality display
            current_personality = self.session_state.get_user_setting(SessionKeys.PERSONALITY, "Professional")
            
            # Style the personality selection with colored boxes
            st.markdown("""
                <style>
                    .personality-box {
                        background-color: rgba(45, 47, 51, 0.8);
                        border-radius: 8px;
                        padding: 16px;
                        margin: 8px 0;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    .personality-box:hover {
                        border-color: rgba(255, 75, 75, 0.5);
                    }
                    .personality-title {
                        color: rgb(250, 250, 250);
                        font-size: 1.2em;
                        margin-bottom: 8px;
                    }
                    .personality-description {
                        color: rgb(200, 200, 200);
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Create tabs for different personality settings sections
            personality_tabs = st.tabs(["🎭 Select Personality", "✏️ Edit System Prompts", "🎚️ Response Settings"])
            
            # Personality Selection Tab
            with personality_tabs[0]:
                st.write("### Choose Personality")
                
                # Get current personality from session state
                current_personality = st.session_state.get(SessionKeys.PERSONALITY, "Professional")
                
                # Create columns for personality boxes
                cols = st.columns(len(PERSONALITY_PRESETS))
                
                # Display each personality option
                for col, (name, preset) in zip(cols, PERSONALITY_PRESETS.items()):
                    with col:
                        # Create a container for better styling
                        st.markdown(f"""
                            <div style="
                                background-color: {'rgba(255, 75, 75, 0.1)' if current_personality == name else 'rgba(45, 47, 51, 0.8)'};
                                border: 2px solid {'rgba(255, 75, 75, 0.8)' if current_personality == name else 'rgba(255, 255, 255, 0.1)'};
                                border-radius: 8px;
                                padding: 16px;
                                margin: 8px 0;
                                text-align: center;
                            ">
                                <div style="font-size: 2em; margin-bottom: 8px;">{preset['icon']}</div>
                                <div style="font-size: 1.2em; color: rgb(250, 250, 250); margin-bottom: 8px;">{name}</div>
                                <div style="color: rgb(200, 200, 200); font-size: 0.9em; margin-bottom: 16px;">{preset['description']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Selection button
                        if st.button(
                            "Select" if current_personality != name else "Selected ✓",
                            key=f"select_personality_{name}",
                            type="primary" if current_personality == name else "secondary",
                            use_container_width=True
                        ):
                            # Update both session state and settings manager
                            st.session_state[SessionKeys.PERSONALITY] = name
                            self.settings_manager.update_setting(SessionKeys.PERSONALITY, name)
                            st.success(f"✨ Switched to {name} personality!")
                            time.sleep(0.5)  # Brief pause for feedback
                            st.rerun()
                
                # Show current personality status
                st.markdown("---")
                st.info(f"🎭 Currently using: **{current_personality}** personality", icon="ℹ️")
            
            # System Prompts Editor Tab
            with personality_tabs[1]:
                st.write("### System Prompt Editor")
                st.warning("⚠️ Editing system prompts will affect how the AI personality behaves. Edit with care.")
                
                # Get custom prompts from session state
                custom_prompts = st.session_state.get("custom_personality_prompts", {})
                
                # Create tabs for each personality
                prompt_tabs = st.tabs([f"{PERSONALITY_PRESETS[name]['icon']} {name}" for name in PERSONALITY_PRESETS.keys()])
                
                for tab, (name, preset) in zip(prompt_tabs, PERSONALITY_PRESETS.items()):
                    with tab:
                        # Show current prompt status
                        if self.settings_manager.has_custom_prompt(name):
                            st.info("✏️ This personality has a custom prompt")
                        else:
                            st.info("📝 Using default prompt")
                        
                        # Default prompt display
                        st.subheader("Default Prompt")
                        default_prompt = self.settings_manager.get_default_prompt_for_display(name)
                        st.code(default_prompt, language="text")
                        
                        st.markdown("---")
                        
                        # Prompt editor
                        st.subheader("Edit Prompt")
                        current_prompt = self.settings_manager.get_prompt_for_display(name)
                        new_prompt = st.text_area(
                            "Edit System Prompt",
                            value=current_prompt,
                            height=300,
                            key=f"prompt_editor_{name}",
                            help="Edit the system prompt for this personality"
                        )
                        
                        # Action buttons
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            if st.button("Save Changes", key=f"save_prompt_{name}", type="primary"):
                                if self.settings_manager.save_custom_prompt(name, new_prompt):
                                    st.success(f"✅ {name} prompt updated successfully!")
                                    time.sleep(0.5)
                                    st.rerun()
                        
                        with col2:
                            if st.button("Reset to Default", key=f"reset_prompt_{name}"):
                                if self.settings_manager.save_custom_prompt(name, default_prompt):
                                    st.success(f"🔄 {name} prompt reset to default!")
                                    time.sleep(0.5)
                                    st.rerun()
                        
                        # Show diff if modified
                        if name in custom_prompts:
                            st.markdown("---")
                            st.subheader("Changes from Default")
                            try:
                                from difflib import unified_diff
                                diff = list(unified_diff(
                                    default_prompt.splitlines(),
                                    current_prompt.splitlines(),
                                    fromfile='Default',
                                    tofile='Current',
                                    lineterm=''
                                ))
                                
                                if diff:
                                    colored_diff = []
                                    for line in diff:
                                        if line.startswith('+'):
                                            colored_diff.append(f'<span style="color: #4CAF50">{line}</span>')
                                        elif line.startswith('-'):
                                            colored_diff.append(f'<span style="color: #F44336">{line}</span>')
                                        else:
                                            colored_diff.append(f'<span style="color: #FAFAFA">{line}</span>')
                                    
                                    st.markdown(f"""
                                        <div style="background-color: rgba(45, 47, 51, 0.8); padding: 10px; border-radius: 5px;">
                                            <pre style="margin: 0; white-space: pre-wrap;">{'<br>'.join(colored_diff)}</pre>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info("No differences from default")
                            except Exception as e:
                                st.error(f"Error showing diff: {str(e)}")
                        
                        # Show help if needed
                        with st.expander("Need help?"):
                            st.markdown("""
                                ### How to Edit Prompts
                                1. The default prompt is shown above for reference
                                2. Edit the prompt in the text area
                                3. Click 'Save Changes' to apply your changes
                                4. Click 'Reset to Default' to revert to the original prompt
                                
                                ### Tips
                                - Keep the core personality traits consistent
                                - Test the personality after making changes
                                - You can always reset to the default if needed
                            """)
            
            # Response Settings Tab
            with personality_tabs[2]:
                st.write("### Response Settings")
                
                # Column layout for settings
                settings_col1, settings_col2 = st.columns([1, 1])
                
                with settings_col1:
                    st.write("#### Tone Adjustment")
                    current_tone = self.session_state.get_user_setting(SessionKeys.TONE, "Balanced")
                    new_tone = st.select_slider(
                        "Select tone of responses",
                        options=[tone[0] for tone in TONE_PRESETS],
                        value=current_tone,
                        help="Adjust how formal or casual the responses should be"
                    )
                    if new_tone != current_tone:
                        self.session_state.update_user_setting(SessionKeys.TONE, new_tone)
                        
                    # Add tone preview
                    st.markdown(f"""
                        <div style="background-color: rgba(45, 47, 51, 0.8); padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="color: #FAFAFA; margin: 0;">Current Tone: <strong>{new_tone}</strong></p>
                            <p style="color: #B0B0B0; margin: 5px 0 0 0; font-size: 0.9em;">
                                {dict(TONE_PRESETS).get(new_tone, "Balanced tone with moderate formality")}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with settings_col2:
                    st.write("#### Creativity Level")
                    current_creativity = self.session_state.get_user_setting(SessionKeys.CREATIVITY, "Balanced")
                    new_creativity = st.select_slider(
                        "Select creativity level",
                        options=[level[0] for level in CREATIVITY_LEVELS],
                        value=current_creativity,
                        help="Adjust how creative or conservative the responses should be"
                    )
                    if new_creativity != current_creativity:
                        self.session_state.update_user_setting(SessionKeys.CREATIVITY, new_creativity)
                        
                    # Add creativity preview
                    st.markdown(f"""
                        <div style="background-color: rgba(45, 47, 51, 0.8); padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <p style="color: #FAFAFA; margin: 0;">Current Creativity: <strong>{new_creativity}</strong></p>
                            <p style="color: #B0B0B0; margin: 5px 0 0 0; font-size: 0.9em;">
                                {dict(CREATIVITY_LEVELS).get(new_creativity, "Balanced creativity level")}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Configuration Preview
                st.write("### Current Configuration")
                preview_data = {
                    "current_personality": current_personality,
                    "tone": new_tone,
                    "creativity": new_creativity,
                    "has_custom_prompt": current_personality in custom_prompts,
                    "active_settings": {
                        "personality_description": PERSONALITY_PRESETS[current_personality]["description"],
                        "tone_preset": dict(TONE_PRESETS).get(new_tone, "Balanced"),
                        "creativity_level": dict(CREATIVITY_LEVELS).get(new_creativity, "Balanced")
                    }
                }
                
                # Display configuration in a styled code block
                st.code(
                    json.dumps(preview_data, indent=2),
                    language="json",
                    line_numbers=True
                )
                
                # Quick Actions
                st.write("### Quick Actions")
                quick_col1, quick_col2, quick_col3 = st.columns(3)
                
                with quick_col1:
                    if st.button("💾 Save Settings", type="primary"):
                        settings = {
                            "personality": current_personality,
                            "tone": new_tone,
                            "creativity": new_creativity
                        }
                        if self.settings_manager.batch_update(settings):
                            st.success("Settings saved successfully!")
                
                with quick_col2:
                    if st.button("🔄 Reset All Personality Settings"):
                        if st.checkbox("Confirm reset", key="confirm_personality_reset"):
                            self.session_state.update_user_setting("custom_personality_prompts", {})
                            self.session_state.update_user_setting(SessionKeys.PERSONALITY, "Professional")
                            self.session_state.update_user_setting(SessionKeys.TONE, "Balanced")
                            self.session_state.update_user_setting(SessionKeys.CREATIVITY, "Balanced")
                            st.success("All personality settings reset to defaults!")
                            st.rerun()
                
                with quick_col3:
                    if st.button("📋 Copy Current Config"):
                        config_str = json.dumps(preview_data, indent=2)
                        pyperclip.copy(config_str)
                        st.success("Configuration copied to clipboard!")

        # Advanced Config Tab
        with tabs[8]:
            st.subheader("Advanced Configuration")
            
            st.warning("""
            ⚠️ Warning: Modifying these settings may affect system stability.
            Please proceed with caution.
            """)
            
            config_text = st.text_area(
                "Edit Configuration (JSON)",
                value=json.dumps(current_settings, indent=2),
                height=300
            )
            
            if st.button("Update Advanced Configuration"):
                try:
                    new_config = json.loads(config_text)
                    if self.settings_manager.batch_update(new_config):
                        st.success("Configuration updated successfully!")
                        st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input.")
                except Exception as e:
                    st.error(f"Error updating configuration: {str(e)}")
        
        # Save All Settings
        st.markdown("---")
        save_col1, save_col2, save_col3 = st.columns([1, 1, 2])
        
        with save_col1:
            if st.button("💾 Save All", type="primary"):
                try:
                    self.settings_manager.save_settings()
                    st.success("All settings saved successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving settings: {str(e)}")
        
        with save_col2:
            if st.button("🔄 Reset All"):
                if st.checkbox("Confirm reset to defaults", key="confirm_reset"):
                    self.settings_manager.reset_to_defaults()
                    st.success("Settings reset to defaults!")
                    time.sleep(1)
                    st.rerun()
        
        with save_col3:
            st.info("💡 Settings are automatically saved when changed. Use 'Save All' to force a save.")

class SettingsManager:
    """Handles settings management and persistence."""
    
    def __init__(self, session_state):
        self.session_state = session_state
        self.settings_file = Path("data/settings.json")
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_settings()  # Load settings when initialized

    def get_prompt_for_display(self, personality_name: str) -> str:
        """Get the current prompt (custom or default) for display in settings."""
        # Get custom prompts if any exist
        custom_prompts = self.session_state.get_user_setting("custom_personality_prompts", {})
        
        # If there's a custom prompt for this personality, use it
        if personality_name in custom_prompts:
            return custom_prompts[personality_name]
        
        # Otherwise, use the default prompt
        return self.get_default_prompt_for_display(personality_name)

    def get_default_prompt_for_display(self, personality_name: str) -> str:
        """Get the default prompt for display in settings."""
        personality = PERSONALITY_PRESETS.get(personality_name)
        if personality:
            return personality['system_prompt'](None)  # Pass None as db since it's just for display
        return PERSONALITY_PRESETS['Professional']['system_prompt'](None)

    def has_custom_prompt(self, personality_name: str) -> bool:
        """Check if a personality has a custom prompt."""
        custom_prompts = self.session_state.get_user_setting("custom_personality_prompts", {})
        return personality_name in custom_prompts

    def save_custom_prompt(self, personality_name: str, prompt: str) -> bool:
        """Save a custom prompt for a specific personality."""
        try:
            # Get existing custom prompts
            custom_prompts = self.session_state.get_user_setting("custom_personality_prompts", {})
            
            # Get the default prompt for comparison
            default_prompt = self.get_default_prompt_for_display(personality_name)
            
            # Update the prompt for this personality
            if prompt.strip() != default_prompt.strip():
                custom_prompts[personality_name] = prompt
            elif personality_name in custom_prompts:
                # Remove if it's the same as default
                custom_prompts.pop(personality_name)
            
            # Save to session state
            self.session_state.update_user_setting("custom_personality_prompts", custom_prompts)
            
            # Save to persistent storage
            self.save_settings()
            return True
        except Exception as e:
            st.error(f"Error saving custom prompt: {str(e)}")
            return False

    def load_settings(self) -> None:
        """Load settings from file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Update session state with loaded settings
                    for key, value in settings.items():
                        self.session_state.update_user_setting(key, value)
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

    def save_settings(self) -> None:
        """Save current settings to file."""
        try:
            settings = self.get_all_settings()
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

    def get_all_settings(self) -> dict:
        """Get all current settings as a dictionary."""
        settings = {
            "PAGE_TITLE": self.session_state.get_user_setting("PAGE_TITLE", "Milimo Chat"),
            "PAGE_ICON": self.session_state.get_user_setting("PAGE_ICON", "🤖"),
            "THEME_CONFIG": self.session_state.get_user_setting("THEME_CONFIG", {
                "primaryColor": "#FF4B4B",
                "backgroundColor": "#0E1117",
                "secondaryBackgroundColor": "#262730",
                "textColor": "#FAFAFA",
                "font": "sans-serif"
            }),
            "OLLAMA_BASE_URL": self.session_state.get_user_setting("OLLAMA_BASE_URL", "http://localhost:11434"),
            "DEFAULT_MODEL": self.session_state.get_user_setting("DEFAULT_MODEL", "llama3.3:latest"),
            "EMBED_MODEL": self.session_state.get_user_setting("EMBED_MODEL", "nomic-embed-text"),
            "custom_personality_prompts": self.session_state.get_user_setting("custom_personality_prompts", {}),
            "temperature": self.session_state.get_user_setting("temperature", 0.7),
            "top_p": self.session_state.get_user_setting("top_p", 0.9),
            "presence_penalty": self.session_state.get_user_setting("presence_penalty", 0.6),
            "frequency_penalty": self.session_state.get_user_setting("frequency_penalty", 0.2),
            "num_ctx": self.session_state.get_user_setting("num_ctx", 32768),
            "num_batch": self.session_state.get_user_setting("num_batch", 512),
            "num_thread": self.session_state.get_user_setting("num_thread", 16),
            "MAX_SHORT_TERM_MEMORY": self.session_state.get_user_setting("MAX_SHORT_TERM_MEMORY", 100),
            "EMBEDDINGS_DIMENSION": self.session_state.get_user_setting("EMBEDDINGS_DIMENSION", 384),
            "memory_enabled": self.session_state.get_user_setting("memory_enabled", True),
            "memory_window": self.session_state.get_user_setting("memory_window", 250),
            "enable_tracking": self.session_state.get_user_setting("enable_tracking", False),
            "save_history": self.session_state.get_user_setting("save_history", True),
            "auto_clean": self.session_state.get_user_setting("auto_clean", True),
            SessionKeys.PERSONALITY: self.session_state.get_user_setting(SessionKeys.PERSONALITY, "Professional"),
            SessionKeys.TONE: self.session_state.get_user_setting(SessionKeys.TONE, "Balanced"),
            SessionKeys.CREATIVITY: self.session_state.get_user_setting(SessionKeys.CREATIVITY, "Balanced")
        }
        return settings

    def update_setting(self, key: str, value: Any) -> bool:
        """Update a single setting and persist changes."""
        try:
            self.session_state.update_user_setting(key, value)
            self.save_settings()  # Save after each update
            return True
        except Exception as e:
            st.error(f"Error updating setting {key}: {str(e)}")
            return False

    def batch_update(self, settings: Dict[str, Any]) -> bool:
        """Update multiple settings at once and persist changes."""
        success = True
        for key, value in settings.items():
            if not self.update_setting(key, value):
                success = False
        return success

    def reset_to_defaults(self) -> None:
        """Reset settings to default values."""
        from config import THEME_CONFIG  # Import default config
        
        default_settings = {
            "PAGE_TITLE": "Milimo Chat",
            "PAGE_ICON": "🤖",
            "THEME_CONFIG": THEME_CONFIG,
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "DEFAULT_MODEL": "llama3.3:latest",
            "EMBED_MODEL": "nomic-embed-text",
            "custom_personality_prompts": {},
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.2,
            "num_ctx": 32768,
            "num_batch": 512,
            "num_thread": 16,
            "MAX_SHORT_TERM_MEMORY": 100,
            "EMBEDDINGS_DIMENSION": 384,
            "memory_enabled": True,
            "memory_window": 250,
            "enable_tracking": False,
            "save_history": True,
            "auto_clean": True,
            SessionKeys.PERSONALITY: "Professional",
            SessionKeys.TONE: "Balanced",
            SessionKeys.CREATIVITY: "Balanced"
        }
        
        self.batch_update(default_settings)
        self.save_settings()
    
    def validate_json_config(self, config_str: str) -> Optional[Dict[str, Any]]:
        """Validate JSON configuration string."""
        try:
            config = json.loads(config_str)
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a JSON object")
            return config
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
            return None
        except Exception as e:
            st.error(f"Configuration error: {str(e)}")
            return None
    
    def apply_theme_config(self, theme_config: Dict[str, str]) -> None:
        """Apply theme configuration to Streamlit."""
        try:
            # Apply theme using st.markdown for custom CSS
            css = f"""
                <style>
                    :root {{
                        --primary-color: {theme_config["primaryColor"]};
                        --background-color: {theme_config["backgroundColor"]};
                        --secondary-background-color: {theme_config["secondaryBackgroundColor"]};
                        --text-color: {theme_config["textColor"]};
                        --font-family: {theme_config["font"]};
                    }}
                    
                    .stApp {{
                        background-color: var(--background-color);
                        color: var(--text-color);
                        font-family: var(--font-family);
                    }}
                    
                    .stButton > button {{
                        background-color: var(--primary-color);
                        color: white;
                    }}
                    
                    .stTextInput > div > div > input,
                    .stSelectbox > div > div > select,
                    .stTextArea > div > div > textarea {{
                        background-color: var(--secondary-background-color);
                        color: var(--text-color);
                    }}
                </style>
            """
            st.markdown(css, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error applying theme: {str(e)}")
    
    def export_settings(self) -> Optional[str]:
        """Export current settings as JSON string."""
        try:
            settings = self.session_state.get_all_settings()
            return json.dumps(settings, indent=2)
        except Exception as e:
            st.error(f"Error exporting settings: {str(e)}")
            return None
    
    def import_settings(self, settings_json: str) -> bool:
        """Import settings from JSON string."""
        config = self.validate_json_config(settings_json)
        if config:
            return self.batch_update(config)
        return False

if __name__ == "__main__":
    try:
        app = MilimoChat()
        app.main()
    finally:
        tracemalloc.stop()