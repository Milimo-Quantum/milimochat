import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pytz

# Base paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
for dir_path in [ASSETS_DIR, DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.3:latest"
EMBED_MODEL = "nomic-embed-text:latest"
VISION_MODEL = "llama3.2-vision:11b-instruct-fp16"

IMAGE_PROMPT = f"""Analyze the image thoroughly and describe its contents in a structured manner. Begin by identifying key objects, entities,
                    and their spatial relationships. Include details about the environment (e.g., indoor/outdoor, time of day, weather), visual elements
                    (colors, textures, lighting), and any notable actions, emotions, or interactions. If text, symbols, or cultural references are present,
                     interpret their significance. Highlight ambiguities or uncertainties if elements are unclear. Format your response as:

                    Primary Subjects: [List and describe main objects/people]
                    Context/Setting: [Location, time, ambiance]
                    Details & Interactions: [Activities, relationships, emotions]
                    Stylistic Features: [Artistic style, colors, lighting]
                    Notable Ambiguities: [Unclear elements, if any]."""

# Memory Configuration
MAX_SHORT_TERM_MEMORY = 10
MEMORY_DB_PATH = DATA_DIR / "memory.db"
EMBEDDINGS_DIMENSION = 384

# UI Configuration
PAGE_TITLE = "Milimo Chat"
PAGE_ICON = "🤖"

# Theme Configuration
THEME_CONFIG = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA",
    "font": "sans-serif"
}

# Personality Presets with enhanced datetime awareness
PERSONALITY_PRESETS = {
    "Professional": {
        "description": "Formal and business-oriented responses",
        "icon": "👔",
        "system_prompt": lambda db: f"""You are a professional AI assistant named Milimo.

ROLE CHARACTERISTICS:
1. You have access to a memory database use it when you need to remember prior conversation
2. Maintain formal and business-appropriate language
3. Provide structured, clear responses
4. Focus on accuracy and precision
5. Use professional terminology when appropriate
6. Maintain professional precision

IMPORTANT: When a user specifically requests the current date/time:
1. Only provide the date and time from the most recent message timestamp
2. Format response as: [Date] [Time]
3. Do not add any other text or context
4. Do not reference the conversation
5. Do not respond with date/time unless explicitly requested

For all other messages: Respond normally without including date/time information.
"""
    },
    "Friendly": {
        "description": "Casual and approachable conversation style",
        "icon": "😊",
        "system_prompt": lambda db: f"""You are a friendly AI assistant named Milimo.

ROLE CHARACTERISTICS:
1. You also have access to a memory database use it when you need to remember prior conversations
2. Use casual, warm, and approachable language
3. Engage in conversational dialogue
4. Show empathy and understanding
5. Keep responses light and accessible
6. Stay time-aware in all responses

IMPORTANT: When a user specifically requests the current date/time:
1. Only provide the date and time from the most recent message timestamp
2. Format response as: [Date] [Time]
3. Do not add any other text or context
4. Do not reference the conversation
5. Do not respond with date/time unless explicitly requested

For all other messages: Respond normally without including date/time information.
"""
    },
    "Technical": {
        "description": "Detailed technical explanations",
        "icon": "⚡",
        "system_prompt": lambda db: f"""You are a technical AI assistant named Milimo.

ROLE CHARACTERISTICS:
1. You also have access to a memory database use it when you need to remember prior conversations
2. Provide in-depth technical explanations
3. Use precise terminology and citations
4. Include code examples when relevant
5. Focus on accuracy and detail
6. Stay time-aware in all responses

IMPORTANT: When a user specifically requests the current date/time:
1. Only provide the date and time from the most recent message timestamp
2. Format response as: [Date] [Time]
3. Do not add any other text or context
4. Do not reference the conversation
5. Do not respond with date/time unless explicitly requested

For all other messages: Respond normally without including date/time information.
"""
    }
}

# Message Types
MESSAGE_TYPES = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "SYSTEM": "system",
    "ERROR": "error"
}

# File Upload Configuration
ALLOWED_EXTENSIONS = {
    'image': ['.jpg', '.jpeg', '.png', '.gif'],
    'document': ['.pdf', '.txt', '.csv', '.xlsx']
}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Export Configuration
EXPORT_FORMATS = {
    "PDF": "application/pdf",
    "CSV": "text/csv",
    "JSON": "application/json"
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "system_base": """ATTENTION - YOUR TEMPORAL CONTEXT:
{personality_prompt}

CONVERSATION CONTEXT:
{context}

IMPORTANT: When a user specifically requests the current date/time:
1. Only provide the date and time from the most recent message timestamp
2. Format response as: [Date] [Time]
3. Do not add any other text or context
4. Do not reference the conversation
5. Do not respond with date/time unless explicitly requested

For all other messages: Respond normally without including date/time information."""
}

# Customization Options
TONE_PRESETS = [
    ("Formal", 0.2),
    ("Balanced", 0.5),
    ("Casual", 0.8)
]

CREATIVITY_LEVELS = [
    ("Conservative", 0.3),
    ("Balanced", 0.7),
    ("Creative", 1.0)
]

# API Response Settings
STREAM_DELAY = 0.01
RESPONSE_TIMEOUT = 300

# Session State Keys
class SessionKeys:
    MESSAGES = "messages"
    CURRENT_MODEL = "current_model"
    PERSONALITY = "personality"
    TONE = "tone"
    CREATIVITY = "creativity"
    MEMORY_ENABLED = "memory_enabled"
    CHAT_HISTORY = "chat_history"
    USER_SETTINGS = "user_settings"

# Error Messages
ERROR_MESSAGES = {
    "model_load": "Failed to load the model. Please try again.",
    "file_too_large": "The uploaded file exceeds the maximum size limit.",
    "invalid_file_type": "This file type is not supported.",
    "memory_error": "Failed to access memory storage.",
    "api_error": "Error communicating with Ollama API.",
    "export_error": "Failed to export chat history.",
    "connection_error": "Cannot connect to Ollama server. Please ensure it's running."
}

# Memory Settings
MEMORY_SETTINGS = {
    "short_term_limit": 100,
    "long_term_limit": 10000,
    "relevance_threshold": 0.75,
    "cleanup_days": 30
}

# Model Parameters
DEFAULT_MODEL_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "presence_penalty": 0.6,
    "frequency_penalty": 0.2,
    "num_ctx": 32768,
    "max_tokens": 32768,
    "llama.embedding_length": 32768,  # Maximum possible value
    "num_batch": 512,
    "num_thread": 8,
    "stop": ["User:", "Assistant:", "System:"]
}

MODEL_PARAMETERS = {
    "context_length": 32768,
    "embedding_length": 32768,
    "max_sequence_length": 32768
}

# UI Customization
UI_SETTINGS = {
    "max_message_width": "800px",
    "animation_duration": "0.3s",
    "border_radius": "0.5rem",
    "shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "spacing": {
        "xs": "0.25rem",
        "sm": "0.5rem",
        "md": "1rem",
        "lg": "1.5rem",
        "xl": "2rem"
    }
}

# Analytics Settings
ANALYTICS_CONFIG = {
    "enable_tracking": False,
    "metrics": [
        "response_time",
        "message_count",
        "token_usage",
        "error_rate"
    ],
    "log_level": "INFO"
}