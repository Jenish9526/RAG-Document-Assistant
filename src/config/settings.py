"""Central configuration module for application parameters, model settings, and paths."""

import os
from dotenv import load_dotenv

load_dotenv()

# Base directory and filesystem storage paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")

FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.pkl")
DOC_REGISTRY_PATH = os.path.join(VECTORSTORE_DIR, "document_registry.pkl")

# Ensure persistent directories exist
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, VECTORSTORE_DIR):
    os.makedirs(_dir, exist_ok=True)

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# Embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Retrieval settings
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# LLM provider settings
LLM_BASE_URL_DEFAULT = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL_DEFAULT = "gemini-3.6-flash"
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Upload validation
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

# Application metadata
APP_TITLE = "RAG Document Assistant"
APP_SUBTITLE = "AI-powered Question Answering from Your Documents"
