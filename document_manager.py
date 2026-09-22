"""Document management module handling document upload, registry, and indexing."""

import os
import pickle
from typing import Dict, List
import streamlit as st

from config import (
    DOC_REGISTRY_PATH,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from document_processor import process_document
from embeddings import generate_embeddings
from vector_store import VectorStore
from utils import compute_file_hash, format_file_size


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    """Return the cached singleton VectorStore instance."""
    store = VectorStore()
    loaded = store.load_index()
    if not loaded:
        if os.path.exists(DOC_REGISTRY_PATH):
            os.remove(DOC_REGISTRY_PATH)
        if "doc_registry" in st.session_state:
            st.session_state.doc_registry = {}
    return store


def _load_registry() -> Dict[str, Dict]:
    """Load the document registry from disk if valid index files exist."""
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        if os.path.exists(DOC_REGISTRY_PATH):
            os.remove(DOC_REGISTRY_PATH)
        return {}
    if os.path.exists(DOC_REGISTRY_PATH):
        with open(DOC_REGISTRY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_registry(registry: Dict[str, Dict]):
    """Persist the document registry mapping to disk."""
    with open(DOC_REGISTRY_PATH, "wb") as f:
        pickle.dump(registry, f)


def get_document_registry() -> Dict[str, Dict]:
    """Retrieve the in-memory or persisted document registry."""
    if "doc_registry" not in st.session_state:
        st.session_state.doc_registry = _load_registry()
    return st.session_state.doc_registry


def validate_file(filename: str, file_bytes: bytes) -> str:
    """Validate file extension, non-emptiness, and size constraints."""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}."

    if len(file_bytes) == 0:
        return "The uploaded file is empty."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"File is too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB."

    return ""


def add_document(filename: str, file_bytes: bytes) -> Dict:
    """Validate, deduplicate, chunk, embed, and index a new uploaded document."""
    error = validate_file(filename, file_bytes)
    if error:
        return {"success": False, "message": error, "duplicate": False}

    file_hash = compute_file_hash(file_bytes)
    registry = get_document_registry()

    if file_hash in registry:
        return {
            "success": False,
            "message": f"'{filename}' has already been added.",
            "duplicate": True,
        }

    result = process_document(file_bytes, filename)
    if not result["success"]:
        return {"success": False, "message": result["error"], "duplicate": False}

    chunks = result["chunks"]
    embeddings_matrix = generate_embeddings(chunks)

    store = get_vector_store()
    store.add_documents(chunks, embeddings_matrix)
    store.save_index()

    registry[file_hash] = {
        "filename": filename,
        "pages": result["pages"],
        "characters": result["characters"],
        "chunks": len(chunks),
        "size": format_file_size(len(file_bytes)),
    }
    st.session_state.doc_registry = registry
    _save_registry(registry)

    return {
        "success": True,
        "message": f"'{filename}' processed successfully — {result['pages']} pages, "
                    f"{len(chunks)} chunks indexed.",
        "duplicate": False,
    }


def clear_all_documents():
    """Clear in-memory and on-disk vector store and document registry."""
    store = get_vector_store()
    store.clear_index()

    st.session_state.doc_registry = {}
    if os.path.exists(DOC_REGISTRY_PATH):
        os.remove(DOC_REGISTRY_PATH)


def list_document_names() -> List[str]:
    """Return filenames of all currently indexed documents."""
    return [info["filename"] for info in get_document_registry().values()]

