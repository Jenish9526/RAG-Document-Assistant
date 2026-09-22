"""
config.py
=========

Purpose
-------
Central place for every "tunable" value used across the project
(chunk size, model names, folder paths, etc.).

Why this file exists
---------------------
Instead of hard-coding numbers like 800 (chunk size) or "all-MiniLM-L6-v2"
(embedding model) inside 5 different files, we put them here ONCE.
If you want to experiment (e.g. change chunk size to 600), you only
change it in one place.

This file does NOT contain secrets (API keys). Secrets live in `.env`
and are loaded through `llm_service.py` using `python-dotenv`.

Used by
-------
Almost every other module imports values from here:
document_processor.py, embeddings.py, vector_store.py, rag_engine.py, app.py
"""

import os

# ---------------------------------------------------------------------
# FOLDER PATHS
# ---------------------------------------------------------------------
# BASE_DIR = the folder this config.py file lives in.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where uploaded raw documents are copied/stored.
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")

# Where processed/intermediate data (if any) is stored.
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Where the FAISS index + metadata are persisted between runs.
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
DOC_REGISTRY_PATH = os.path.join(VECTOR_DB_DIR, "document_registry.pkl")

# Make sure these folders exist the moment config.py is imported.
for _folder in (DOCUMENTS_DIR, PROCESSED_DIR, VECTOR_DB_DIR):
    os.makedirs(_folder, exist_ok=True)

# ---------------------------------------------------------------------
# CHUNKING SETTINGS
# ---------------------------------------------------------------------
# Chunk size is measured in words.
CHUNK_SIZE = 350          # words per chunk (~roughly 450-500 tokens)
CHUNK_OVERLAP = 60        # words repeated between consecutive chunks

# ---------------------------------------------------------------------
# EMBEDDING SETTINGS
# ---------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, good-quality model
EMBEDDING_DIMENSION = 384              # output vector size of the model above

# ---------------------------------------------------------------------
# RETRIEVAL SETTINGS
# ---------------------------------------------------------------------
TOP_K = 5                 # how many chunks to retrieve per question
SIMILARITY_THRESHOLD = 0.25  # chunks below this score are treated as "not relevant"
# (Score is cosine similarity converted to a 0-1 range; see vector_store.py)

# ---------------------------------------------------------------------
# LLM SETTINGS
# ---------------------------------------------------------------------
# The project is designed so the LLM provider can be swapped without
# touching rag_engine.py — only llm_service.py needs to change.
#
# Default provider: Google Gemini (1M TPM free tier, fast, OpenAI-compatible).
# You can switch to Groq, OpenAI, Ollama, etc. by changing
# LLM_BASE_URL and LLM_MODEL in your .env file.
LLM_BASE_URL_DEFAULT = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL_DEFAULT = "gemini-3.5-flash-lite"
LLM_MAX_TOKENS = 800
LLM_TEMPERATURE = 0.3

# ---------------------------------------------------------------------
# UPLOAD / VALIDATION SETTINGS
# ---------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 25

# ---------------------------------------------------------------------
# APP SETTINGS
# ---------------------------------------------------------------------
APP_TITLE = "RAG Document Assistant"
APP_SUBTITLE = "AI-powered Question Answering from Your Documents"
