import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from streamlit_extras.add_vertical_space import add_vertical_space
from config import THEME_CONFIG

def apply_theme():
    """Apply enhanced custom theme to the Streamlit app with comprehensive improvements."""
    st.markdown("""
        <style>
            /* Global Styles and Typography */
            :root {
                --primary-color: #00C8FF;
                --primary-dark: #004e92;
                --primary-light: rgba(0, 200, 255, 0.2);
                --background-gradient: linear-gradient(165deg, #000428 0%, #004e92 100%);
                --text-primary: rgba(255, 255, 255, 0.87);
                --text-secondary: rgba(255, 255, 255, 0.6);
                --border-color: rgba(0, 200, 255, 0.2);
                --shadow-color: rgba(0, 0, 0, 0.2);
                --transition-speed: 0.3s;
            }

            /* Base Application Styling */
            .stApp {
                background: var(--background-gradient) !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                color: var(--text-primary);
                line-height: 1.5;
            }

            /* Improved Text Readability */
            .stMarkdown, p, span, label, .streamlit-expanderHeader,
            .stTextInput label, .stTextArea label, .stSelectbox label,
            .stDateInput label, .element-container div {
                color: var(--text-primary) !important;
                font-weight: 400;
                letter-spacing: 0.015em;
            }

            /* Enhanced Sidebar Design */
            [data-testid="stSidebar"] {
                background: rgba(0, 78, 146, 0.45) !important;
                backdrop-filter: blur(10px) !important;
                border-right: 1px solid var(--border-color);
                padding: 1rem 1rem !important;
                width: 300px !important;
                transition: transform var(--transition-speed) ease;
            }

            [data-testid="stSidebar"] .sidebar-content {
                background: transparent !important;
            }

            /* Improved Button Styling */
            .stButton > button {
                background: rgba(0, 78, 146, 0.8) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem 1.5rem !important;
                transition: all var(--transition-speed) ease !important;
                backdrop-filter: blur(5px) !important;
                width: 90%;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--shadow-color);
                border-color: var(--primary-color) !important;
            }

            /* Enhanced Message Containers */
            .chat-message {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1.25rem !important;
                margin: 1rem 0 !important;
                border-left: 3px solid var(--primary-color) !important;
                transition: all var(--transition-speed) ease !important;
                box-shadow: 0 2px 8px var(--shadow-color);
            }

            .chat-message:hover {
                transform: translateX(4px);
                background: rgba(0, 78, 146, 0.3) !important;
            }

            /* User Message Styling */
            .chat-message.user {
                background: rgba(0, 200, 255, 0.1) !important;
                margin-left: auto !important;
                margin-right: 1rem !important;
                max-width: 80% !important;
            }

            /* Assistant Message Styling */
            .chat-message.assistant {
                background: rgba(0, 78, 146, 0.2) !important;
                margin-right: auto !important;
                margin-left: 1rem !important;
                max-width: 80% !important;
            }

            /* Code Block Styling */
            pre {
                background: rgba(0, 78, 146, 0.3) !important;
                border-radius: 0.5rem !important;
                padding: 1rem !important;
                border: 1px solid var(--border-color) !important;
                overflow-x: auto !important;
            }

            code {
                color: #00C8FF !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 0.9rem !important;
            }

            /* File Upload Styling */
            [data-testid="stFileUploader"] {
                background: rgba(0, 78, 146, 0.3) !important;
                border-radius: 0.75rem !important;
                padding: 1.5rem !important;
                border: 2px dashed var(--border-color) !important;
                transition: all var(--transition-speed) ease !important;
            }

            [data-testid="stFileUploader"]:hover {
                border-color: var(--primary-color) !important;
                background: rgba(0, 78, 146, 0.4) !important;
            }

            /* Enhanced Select Box Styling */
            .stSelectbox > div > div {
                background: rgba(0, 78, 146, 0.8) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 0.5rem !important;
                color: var(--text-primary) !important;
            }

            .stSelectbox > div > div:hover {
                border-color: var(--primary-color) !important;
            }

            /* Metric Card Styling */
            [data-testid="stMetricValue"] {
                background: rgba(0, 78, 146, 0.2) !important;
                padding: 1rem !important;
                border-radius: 0.5rem !important;
                border: 1px solid var(--border-color) !important;
                transition: all var(--transition-speed) ease !important;
            }

            [data-testid="stMetricValue"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--shadow-color);
            }

            /* Chart and Graph Styling */
            .js-plotly-plot {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1rem !important;
                border: 1px solid var(--border-color) !important;
            }

            .js-plotly-plot .plotly .modebar {
                background: rgba(0, 78, 146, 0.8) !important;
                border-radius: 0.5rem !important;
            }

            /* Loading Animation */
            @keyframes pulse {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }

            .stProgress .st-bo {
                background-color: var(--primary-color) !important;
                animation: pulse 2s infinite ease-in-out;
            }

            /* Enhanced Memory Metric Cards */
            .element-container > .stMetric {
                background: rgba(0, 78, 146, 0.2) !important;
                border: 1px solid rgba(0, 200, 255, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1.25rem !important;
                margin: 0.5rem 0 !important;
                width: 100% !important;
                box-sizing: border-box !important;
                display: flex !important;
                flex-direction: column !important;
                gap: 0.5rem !important;
            }

            /* Memory Metric Value Styling */
            .stMetric [data-testid="stMetricValue"] {
                font-size: 2rem !important;
                font-weight: 600 !important;
                color: var(--text-primary) !important;
                text-align: center !important;
                padding: 0.5rem !important;
                background: rgba(0, 78, 146, 0.3) !important;
                border-radius: 0.5rem !important;
                margin: 0.25rem 0 !important;
            }

            /* Memory Controls Section */
            [data-testid="stVerticalBlock"] > div:has(.memory-controls) {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1.5rem !important;
                margin: 1rem 0 !important;
                border: 1px solid rgba(0, 200, 255, 0.2) !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            }

            /* Model Selection Styling */
            .stSelectbox > div {
                background: rgba(0, 78, 146, 0.3) !important;
                border: 1px solid rgba(0, 200, 255, 0.2) !important;
                border-radius: 0.5rem !important;
                padding: 0.25rem !important;
                margin: 0.5rem 0 !important;
                width: 90% !important;
                box-sizing: border-box !important;
            }

            .stSelectbox > div > div {
                color: var(--text-primary) !important;
            }

            /* Memory Settings Container */
            .memory-settings-container {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1.5rem !important;
                margin: 1rem 0 !important;
                border: 1px solid rgba(0, 200, 255, 0.2) !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            }

            /* Retention Slider Improvements */
            .stSlider > div {
                padding: 1rem 0.5rem !important;
            }

            .stSlider [data-testid="stThumbValue"] {
                background: rgba(0, 200, 255, 0.9) !important;
                border: none !important;
                padding: 0.25rem 0.5rem !important;
                border-radius: 0.25rem !important;
            }

            /* Operation Buttons Container */
            .memory-operations {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.75rem !important;
                padding: 0.5rem !important;
            }

            .memory-operations button {
                background: rgba(0, 78, 146, 0.4) !important;
                border: 1px solid rgba(0, 200, 255, 0.2) !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem !important;
                transition: all var(--transition-speed) ease !important;
                width: 100% !important;
            }
                
            /* Chat input container with improved layering */
            [data-testid="stChatInput"] {
                width: auto !important;
                position: auto !important;
                bottom: auto !important;
                left: auto !important;
                right: auto !important;
                background: transparent !important;
                padding: 1rem 2rem !important;
                margin: auto !important;
                z-index: 999 !important;
                backdrop-filter: blur(10px) !important;
            }

            .stChatInput textarea {
                background: rgba(255, 255, 255, 0.05) !important;
                
                border-radius: 1rem !important;
                padding: 0.75rem 1rem !important;
                
                width: 100% !important;
                max-width: auto !important;
                margin: auto !important;
                display: block !important;
            }
            
            .stChatInput textarea:focus {
                border-color: var(--primary-color) !important;
                width: 100% !important;
            }

            .memory-card {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.75rem !important;
                padding: 1.25rem !important;
                margin: 1rem 0 !important;
                border: 1px solid var(--border-color) !important;
                transition: all var(--transition-speed) ease !important;
                max-width: 100% !important;
                overflow: hidden !important;
                word-wrap: break-word !important;
                box-sizing: border-box !important;
            }

            /* Memory Details Section */
            [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
                max-width: 100% !important;
                padding: 0 1rem !important;
                box-sizing: border-box !important;
            }

            /* Memory Tabs Navigation */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem !important;
                background-color: transparent !important;
                border-bottom: 1px solid var(--border-color) !important;
                width: 100% !important;
                overflow-x: auto !important;
                padding: 0 1rem !important;
            }

            .stTabs [data-baseweb="tab"] {
                flex: 0 0 auto !important;
                white-space: nowrap !important;
            }

            /* Memory Operations Buttons */
            .memory-operations button {
                width: 100% !important;
                margin: 0.5rem 0 !important;
                padding: 0.75rem 1rem !important;
            }

            /* Search Bar Container */
            .stTextInput > div:first-child {
                width: 100% !important;
                max-width: 100% !important;
                padding: 0 1rem !important;
            }

            /* Metrics Container */
            [data-testid="metric-container"] {
                background: rgba(0, 78, 146, 0.2) !important;
                border-radius: 0.5rem !important;
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            /* Sort and Items Per Page Controls */
            .stSelectbox {
                max-width: 100% !important;
                min-width: 0 !important;
                margin: 0.5rem 0 !important;
            }

            /* Memory Content */
            .memory-content {
                word-break: break-word !important;
                white-space: pre-wrap !important;
                max-width: 100% !important;
            }

            /* Memory Settings Panel */
            .memory-settings {
                padding: 1rem !important;
                background: rgba(0, 78, 146, 0.3) !important;
                border-radius: 0.75rem !important;
                margin: 1rem 0 !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            /* Retention Slider */
            .stSlider {
                padding: 1rem 0 !important;
                width: 90% !important;
            }

            /* Analytics Section */
            .memory-analytics {
                padding: 1rem !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }

            .memory-analytics .plotly-graph-div {
                width: 100% !important;
                max-width: 100% !important;
            }

            /* Memory Operations Buttons Layout */
            [data-testid="stHorizontalBlock"] {
                gap: 0.5rem !important;
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
                padding-bottom: 0.5rem !important;
                margin: 0 -1rem !important;
                padding: 0 1rem !important;
            }

            /* Container Width Control */
            .block-container, [data-testid="stVerticalBlock"] {
                max-width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                box-sizing: border-box !important;
            }

            .memory-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--shadow-color);
            }

            /* Toast Notifications */
            .stToast {
                background: rgba(0, 78, 146, 0.95) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 0.5rem !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
            }

            /* Mobile and Responsive Layout Fixes */
            @media (max-width: 768px) {
                /* Improved Container Spacing */
                .block-container {
                    padding: 0.5rem !important;
                }

                /* Stack Memory Cards */
                .memory-card {
                    margin: 0.5rem 0 !important;
                }

                /* Adjust Button Sizes */
                button {
                    padding: 0.5rem !important;
                    min-height: 2.5rem !important;
                }

                /* Memory Details Responsive Layout */
                [data-testid="stHorizontalBlock"] {
                    flex-direction: column !important;
                }

                [data-testid="stHorizontalBlock"] > div {
                    width: 100% !important;
                    margin-bottom: 0.5rem !important;
                }

                /* Responsive Controls */
                .stSelectbox, .stTextInput {
                    width: 100% !important;
                }

                /* Memory Analytics Mobile View */
                .memory-analytics .plotly-graph-div {
                    height: 200px !important;
                }

                /* Tab Navigation Scroll */
                .stTabs [data-baseweb="tab-list"] {
                    overflow-x: auto !important;
                    -webkit-overflow-scrolling: touch !important;
                    scrollbar-width: none !important;
                }

                /* Memory Settings Mobile Layout */
                .memory-settings {
                    padding: 0.75rem !important;
                }

                /* Retention Slider Mobile */
                .stSlider {
                    padding: 0.75rem 0 !important;
                }
                [data-testid="stSidebar"] {
                    width: 100% !important;
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    height: auto !important;
                    transform: translateY(-100%);
                }

                [data-testid="stSidebar"].expanded {
                    transform: translateY(0);
                }

                .chat-message {
                    margin: 0.5rem !important;
                    max-width: 90% !important;
                }
            }

            /* Hide Streamlit Branding */
            #MainMenu, footer, header {
                display: none !important;
            }
                
        </style>
    """, unsafe_allow_html=True)

def section_header(title: str, description: str = None, icon: str = None):
    """Create an enhanced section header with improved styling."""
    if icon:
        title = f"{icon} {title}"
    
    st.markdown(f"""
        <div class="section-header" style="
            background: rgba(0, 78, 146, 0.2);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin: 1rem 0;
            border-left: 3px solid var(--primary-color);
        ">
            <h2 style="
                color: var(--text-primary);
                margin: 0;
                font-size: 1.5rem;
                font-weight: 600;
            ">{title}</h2>
            {f'<p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">{description}</p>' if description else ''}
        </div>
    """, unsafe_allow_html=True)

def card_container(content: str, padding: str = "1.25rem", bg_opacity: float = 0.2):
    """Create an enhanced card container with improved visual hierarchy."""
    st.markdown(f"""
        <div class="card-container" style="
            background: rgba(0, 78, 146, {bg_opacity});
            border-radius: 0.75rem;
            padding: {padding};
            margin: 1rem auto;
            width: 90%;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px var(--shadow-color);
            transition: all var(--transition-speed) ease;
        ">
            {content}
        </div>
    """, unsafe_allow_html=True)

def message_container(role: str, content: str, timestamp: str = None):
    """Create an enhanced message container with improved interaction design."""
    bg_color = "rgba(0, 200, 255, 0.1)" if role == "user" else "rgba(0, 78, 146, 0.2)"
    border_color = "var(--primary-color)"
    
    timestamp_html = f'''
        <div style="
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 0.5rem;
        ">{timestamp}</div>
    ''' if timestamp else ''
    
    st.markdown(f"""
        <div class="message-container {role}" style="
            background: {bg_color};
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin: 1rem auto;
            width: 90%;
            border-left: 3px solid {border_color};
            transition: all var(--transition-speed) ease;
            box-shadow: 0 2px 8px var(--shadow-color);
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
            ">
                <strong style="
                    color: var(--text-primary);
                    font-size: 1rem;
                ">{role.title()}</strong>
                {timestamp_html}
            </div>
            <div style="
                color: var(--text-primary);
                line-height: 1.6;
                font-size: 1rem;
            ">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)

def initialize_theme():
    """Initialize the enhanced theme with all components and font loading."""
    # Apply base theme
    apply_theme()
    
    # Load custom fonts
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            /* Typography Enhancements */
            body {
                font-feature-settings: "liga" 1, "kern" 1;
                text-rendering: optimizeLegibility;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }

            /* Theme Initialization Specific Styles */
            .theme-initialized {
                opacity: 0;
                animation: fadeIn 0.5s ease forwards;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Additional Interactive Elements */
            .interactive-element {
                position: relative;
                overflow: hidden;
            }

            .interactive-element::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    45deg,
                    transparent 0%,
                    rgba(255, 255, 255, 0.05) 50%,
                    transparent 100%
                );
                transform: translateX(-100%);
                transition: transform 0.6s ease;
            }

            .interactive-element:hover::after {
                transform: translateX(100%);
            }

            /* Enhanced Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }

            ::-webkit-scrollbar-track {
                background: rgba(0, 78, 146, 0.1);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb {
                background: rgba(0, 200, 255, 0.3);
                border-radius: 4px;
                transition: background var(--transition-speed) ease;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: rgba(0, 200, 255, 0.5);
            }

            /* Focus States */
            *:focus {
                outline: none;
                box-shadow: 0 0 0 2px var(--primary-color) !important;
                transition: box-shadow var(--transition-speed) ease;
            }

            /* Keyboard Navigation Improvements */
            *:focus-visible {
                outline: 2px solid var(--primary-color);
                outline-offset: 2px;
            }

            /* Progress Indicators */
            .progress-indicator {
                background: linear-gradient(
                    90deg,
                    var(--primary-color) 0%,
                    rgba(0, 200, 255, 0.5) 100%
                );
                height: 2px;
                width: 100%;
                position: relative;
                overflow: hidden;
            }

            .progress-indicator::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent 0%,
                    rgba(255, 255, 255, 0.5) 50%,
                    transparent 100%
                );
                animation: shimmer 2s infinite linear;
            }

            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
        </style>
    """, unsafe_allow_html=True)

def create_grid_layout(columns: int = 2):
    """Create a responsive grid layout with enhanced styling."""
    st.markdown(f"""
        <style>
            .grid-layout {{
                display: grid;
                grid-template-columns: repeat({columns}, 1fr);
                gap: 1rem;
                margin: 1rem 0;
            }}

            @media (max-width: 768px) {{
                .grid-layout {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    """, unsafe_allow_html=True)
    return st.container()

def create_tab_group(tabs: list):
    """Create a custom styled tab group."""
    tab_html = ""
    for i, tab in enumerate(tabs):
        tab_html += f"""
            <button class="tab-button {'active' if i == 0 else ''}"
                    onclick="switchTab({i})"
                    style="
                        background: transparent;
                        border: none;
                        color: var(--text-primary);
                        padding: 0.75rem 1.5rem;
                        border-bottom: 2px solid {('var(--primary-color)' if i == 0 else 'transparent')};
                        transition: all var(--transition-speed) ease;
                        cursor: pointer;
                    ">{tab}</button>
        """

    st.markdown(f"""
        <div class="tab-group" style="
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        ">
            {tab_html}
        </div>
        <script>
            function switchTab(index) {{
                const tabs = document.querySelectorAll('.tab-button');
                tabs.forEach((tab, i) => {{
                    if (i === index) {{
                        tab.classList.add('active');
                        tab.style.borderBottom = '2px solid var(--primary-color)';
                    }} else {{
                        tab.classList.remove('active');
                        tab.style.borderBottom = '2px solid transparent';
                    }}
                }});
            }}
        </script>
    """, unsafe_allow_html=True)

def create_accordion(title: str, content: str):
    """Create a custom styled accordion component."""
    st.markdown(f"""
        <div class="accordion" style="
            background: rgba(0, 78, 146, 0.2);
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            overflow: hidden;
        ">
            <button class="accordion-header"
                    onclick="toggleAccordion(this.parentElement)"
                    style="
                        width: 100%;
                        background: transparent;
                        border: none;
                        padding: 1rem;
                        color: var(--text-primary);
                        text-align: left;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        cursor: pointer;
                    ">
                <span>{title}</span>
                <span class="accordion-icon">▼</span>
            </button>
            <div class="accordion-content" style="
                max-height: 0;
                overflow: hidden;
                transition: max-height var(--transition-speed) ease;
                padding: 0 1rem;
            ">
                {content}
            </div>
        </div>
        <script>
            function toggleAccordion(accordion) {{
                const content = accordion.querySelector('.accordion-content');
                const icon = accordion.querySelector('.accordion-icon');
                
                if (content.style.maxHeight === '0px' || !content.style.maxHeight) {{
                    content.style.maxHeight = content.scrollHeight + 'px';
                    content.style.padding = '1rem';
                    icon.style.transform = 'rotate(180deg)';
                }} else {{
                    content.style.maxHeight = '0px';
                    content.style.padding = '0 1rem';
                    icon.style.transform = 'rotate(0deg)';
                }}
            }}
        </script>
    """, unsafe_allow_html=True)

def create_tooltip(content: str, tooltip_text: str):
    """Create a custom styled tooltip component."""
    st.markdown(f"""
        <div class="tooltip-container" style="
            position: relative;
            display: inline-block;
        ">
            <span>{content}</span>
            <div class="tooltip" style="
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 0.5rem;
                background: rgba(0, 78, 146, 0.95);
                border-radius: 0.25rem;
                color: var(--text-primary);
                font-size: 0.875rem;
                white-space: nowrap;
                opacity: 0;
                visibility: hidden;
                transition: all var(--transition-speed) ease;
            ">
                {tooltip_text}
            </div>
        </div>
        <style>
            .tooltip-container:hover .tooltip {{
                opacity: 1;
                visibility: visible;
                transform: translateX(-50%) translateY(-0.5rem);
            }}
        </style>
    """, unsafe_allow_html=True)

def create_notification(message: str, type: str = "info"):
    """Create a custom styled notification component."""
    colors = {
        "info": "#00C8FF",
        "success": "#00E676",
        "warning": "#FFD740",
        "error": "#FF5252"
    }
    
    st.markdown(f"""
        <div class="notification" style="
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 1rem;
            background: rgba(0, 78, 146, 0.95);
            border-left: 4px solid {colors.get(type, colors['info'])};
            border-radius: 0.5rem;
            color: var(--text-primary);
            box-shadow: 0 4px 12px var(--shadow-color);
            z-index: 9999;
            animation: slideIn 0.3s ease forwards;
        ">
            {message}
            <button onclick="this.parentElement.remove()" style="
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                background: transparent;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
            ">×</button>
        </div>
        <style>
            @keyframes slideIn {{
                from {{ transform: translateX(100%); }}
                to {{ transform: translateX(0); }}
            }}
        </style>
    """, unsafe_allow_html=True)

def apply_motion_effects():
    """Apply subtle motion effects to enhance interactivity."""
    st.markdown("""
        <style>
            /* Hover lift effect */
            .hover-lift {
                transition: transform var(--transition-speed) ease;
            }
            
            .hover-lift:hover {
                transform: translateY(-2px);
            }

            /* Smooth page transitions */
            .main {
                animation: fadeIn 0.3s ease;
            }

            /* Loading states */
            .loading {
                position: relative;
                overflow: hidden;
            }

            .loading::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent 0%,
                    rgba(255, 255, 255, 0.1) 50%,
                    transparent 100%
                );
                animation: shimmer 1.5s infinite linear;
            }
        </style>
    """, unsafe_allow_html=True)