import streamlit as st
from typing import Optional, Dict, Any, List, Union, Callable
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space

def styled_container(key: str) -> stylable_container:
    """Create a styled container with consistent theme."""
    return stylable_container(
        key=key,
        css_styles="""
            {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """
    )

class StyledComponents:
    @staticmethod
    def card(
        content: str,
        key: str,
        padding: str = "1rem",
        with_border: bool = True,
        on_click: Optional[Callable] = None
    ):
        """Create a styled card component."""
        with styled_container(f"card_{key}"):
            if on_click:
                if st.button(
                    content,
                    key=f"card_button_{key}",
                    use_container_width=True
                ):
                    on_click()
            else:
                st.markdown(content, unsafe_allow_html=True)

    @staticmethod
    def metric_card(
        title: str,
        value: Any,
        delta: Optional[Union[float, str]] = None,
        help_text: Optional[str] = None
    ):
        """Create a styled metric card."""
        with styled_container("metric_card"):
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )

    @staticmethod
    def action_button(
        label: str,
        key: str,
        icon: Optional[str] = None,
        type: str = "primary",
        help_text: Optional[str] = None
    ) -> bool:
        """Create a styled action button."""
        with styled_container(f"button_{key}"):
            button_text = f"{icon} {label}" if icon else label
            return st.button(
                button_text,
                key=key,
                help=help_text,
                use_container_width=True,
                type=type
            )

    @staticmethod
    def info_box(
        message: str,
        type: str = "info",
        icon: Optional[str] = None,
        dismissible: bool = True
    ):
        """Create a styled info box."""
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        display_icon = icon or icons.get(type, "ℹ️")
        
        with styled_container(f"info_box_{type}"):
            cols = st.columns([0.1, 0.8, 0.1])
            with cols[0]:
                st.markdown(display_icon)
            with cols[1]:
                st.markdown(message)
            if dismissible:
                with cols[2]:
                    if st.button("✕", key=f"dismiss_{type}"):
                        return False
            return True

    @staticmethod
    def search_bar(
        placeholder: str = "Search...",
        key: str = "search",
        on_change: Optional[Callable] = None
    ) -> str:
        """Create a styled search bar."""
        with styled_container("search_bar"):
            return st.text_input(
                "",
                placeholder=placeholder,
                key=key,
                on_change=on_change if on_change else None
            )

    @staticmethod
    def file_upload_area(
        key: str,
        accepted_types: List[str],
        max_size_mb: int = 5,
        help_text: Optional[str] = None
    ):
        """Create a styled file upload area."""
        with styled_container("file_upload"):
            uploaded_file = st.file_uploader(
                "Drop files here or click to upload",
                type=accepted_types,
                key=key,
                help=help_text or f"Max file size: {max_size_mb}MB"
            )
            if uploaded_file:
                with styled_container("file_preview"):
                    cols = st.columns([0.8, 0.2])
                    with cols[0]:
                        st.markdown(f"**Selected:** {uploaded_file.name}")
                    with cols[1]:
                        if st.button("Clear", key=f"clear_{key}"):
                            st.session_state[key] = None
                            st.rerun()
            return uploaded_file

    @staticmethod
    def tabbed_container(
        tabs: Dict[str, Callable],
        key: str = "tabs",
        default_tab: Optional[str] = None
    ):
        """Create a styled tabbed container."""
        with styled_container("tabbed_container"):
            tab_titles = list(tabs.keys())
            active_tab = st.radio(
                "Select Tab",
                tab_titles,
                horizontal=True,
                key=f"{key}_selector",
                index=tab_titles.index(default_tab) if default_tab else 0
            )
            
            with styled_container(f"tab_content_{active_tab}"):
                tabs[active_tab]()

    @staticmethod
    def collapsible_section(
        header: str,
        content: Callable,
        key: str,
        icon: Optional[str] = None,
        default_open: bool = False
    ):
        """Create a styled collapsible section."""
        with styled_container(f"collapsible_{key}"):
            header_text = f"{icon} {header}" if icon else header
            with st.expander(header_text, expanded=default_open):
                content()

    @staticmethod
    def notification(
        message: str,
        type: str = "info",
        duration: int = 3
    ):
        """Create a styled notification."""
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
        
        notification = {
            "message": message,
            "type": type,
            "timestamp": st.session_state.get("_current_time", 0) + duration
        }
        
        st.session_state.notifications.append(notification)

    @staticmethod
    def progress_tracker(
        steps: List[str],
        current_step: int,
        key: str = "progress"
    ):
        """Create a styled progress tracker."""
        with styled_container("progress_tracker"):
            cols = st.columns(len(steps))
            for i, step in enumerate(steps):
                with cols[i]:
                    if i < current_step:
                        st.markdown(f"✅ {step}")
                    elif i == current_step:
                        st.markdown(f"🔵 {step}")
                    else:
                        st.markdown(f"⚪ {step}")

    @staticmethod
    def dual_range_slider(
        label: str,
        min_value: float,
        max_value: float,
        key: str,
        step: Optional[float] = None,
        help_text: Optional[str] = None
    ) -> tuple:
        """Create a styled dual range slider."""
        with styled_container("range_slider"):
            col1, col2 = st.columns(2)
            
            with col1:
                start_value = st.number_input(
                    f"{label} (Start)",
                    min_value=min_value,
                    max_value=max_value,
                    step=step or (max_value - min_value) / 100,
                    key=f"{key}_start"
                )
            
            with col2:
                end_value = st.number_input(
                    f"{label} (End)",
                    min_value=start_value,
                    max_value=max_value,
                    step=step or (max_value - min_value) / 100,
                    key=f"{key}_end"
                )
            
            if help_text:
                st.caption(help_text)
            
            return start_value, end_value

    @staticmethod
    def tag_selector(
        options: List[str],
        key: str,
        max_selections: Optional[int] = None,
        help_text: Optional[str] = None
    ) -> List[str]:
        """Create a styled tag selector."""
        with styled_container("tag_selector"):
            selected = []
            cols = st.columns(4)
            for i, option in enumerate(options):
                with cols[i % 4]:
                    if st.checkbox(
                        option,
                        key=f"{key}_{i}",
                        disabled=len(selected) >= max_selections if max_selections else False
                    ):
                        selected.append(option)
            
            if help_text:
                st.caption(help_text)
            
            return selected

    @staticmethod
    def copy_to_clipboard(
        text: str,
        button_text: str = "Copy",
        key: str = "copy"
    ):
        """Create a styled copy to clipboard button."""
        with styled_container("clipboard"):
            if st.button(button_text, key=key):
                st.write_clipboard(text)
                st.success("Copied to clipboard!")

    @staticmethod
    def loading_placeholder(key: str):
        """Create a styled loading placeholder."""
        with styled_container(f"loading_{key}"):
            st.markdown(
                """
                <div class="loading-shimmer" style="height: 100px;">
                    <div class="loading-content"></div>
                </div>
                """,
                unsafe_allow_html=True
            )