# config.py
"""
Application configuration and constants
"""

# Model configurations
MODELS_AVAILABLE = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
DEFAULT_MODEL = "gpt-3.5-turbo"

# RAG configurations
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 3

# File processing configurations
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx"]
FILE_PREVIEW_LENGTH = 2000

# UI configurations
TYPING_SPEED = 0.01
MESSAGE_TRUNCATE_LENGTH = 8000

# Session state keys
SESSION_KEYS = {
    "model": "MAXCHAT_model_chosen",
    "previous_model": "previous_model",
    "messages": "messages",
    "uploaded_files": "uploaded_files",
    "show_content_prefix": "show_content_",
    "embeddings": "embeddings",
    "document_chunks": "document_chunks"
}

# Default messages
DEFAULT_ASSISTANT_MESSAGE = "Hi there! What would you like to know?"
MODEL_SWITCH_MESSAGE = "Switched model to **{model}**"
FILE_PROCESSED_MESSAGE = "✅ File '{filename}' has been processed and is ready for analysis."
FILE_UPLOAD_MESSAGE = "📎 Uploaded file: **{filename}**"