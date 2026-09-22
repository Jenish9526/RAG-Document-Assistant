"""Central configuration module for application parameters, model settings, and paths."""

import os

# Base directory and vector database storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
DOC_REGISTRY_PATH = os.path.join(VECTOR_DB_DIR, "document_registry.pkl")

os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Chunking settings
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

# LLM provider settings
LLM_BASE_URL_DEFAULT = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL_DEFAULT = "gemini-3.5-flash-lite"
LLM_MAX_TOKENS = 800
LLM_TEMPERATURE = 0.3

# Upload validation
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 25

# Application metadata
APP_TITLE = "RAG Document Assistant"
APP_SUBTITLE = "AI-powered Question Answering from Your Documents"

