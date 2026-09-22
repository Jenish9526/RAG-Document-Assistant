"""
document_manager.py
====================

Purpose
-------
Manages the "document layer" of the app: which documents have been
uploaded, duplicate detection, per-document statistics, and wiring
together document_processor.py + embeddings.py + vector_store.py
whenever a new file arrives.

This file also owns the single shared VectorStore instance (loaded
once via Streamlit caching) so every part of the app searches/writes
to the same index.

Imported libraries
-------------------
- streamlit (for @st.cache_resource / session_state)
- pickle    (persisting the "document registry" — the list of already
             processed documents / hashes — between app restarts)

Used by
-------
app.py -> file upload handler, sidebar document list, "Clear Documents".
"""

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


# ---------------------------------------------------------------------
# SHARED VECTOR STORE (singleton, loaded once per Streamlit session)
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    """
    Function: get_vector_store()

    Purpose:
        Returns the single shared VectorStore instance for the app.
        `@st.cache_resource` ensures this is created only once and then
        reused across every user interaction (Streamlit re-runs the
        whole script on every click, so without this we'd lose the
        index every time).

    Output:
        A VectorStore with any previously saved index already loaded.

    Used by:
        Every function below, and app.py directly.
    """
    store = VectorStore()
    loaded = store.load_index()
    if not loaded:
        if os.path.exists(DOC_REGISTRY_PATH):
            os.remove(DOC_REGISTRY_PATH)
        if "doc_registry" in st.session_state:
            st.session_state.doc_registry = {}
    return store


def _load_registry() -> Dict[str, Dict]:
    """Loads the {file_hash: {"filename":.., "pages":.., "characters":.., "chunks":..}} registry."""
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        if os.path.exists(DOC_REGISTRY_PATH):
            os.remove(DOC_REGISTRY_PATH)
        return {}
    if os.path.exists(DOC_REGISTRY_PATH):
        with open(DOC_REGISTRY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_registry(registry: Dict[str, Dict]):
    with open(DOC_REGISTRY_PATH, "wb") as f:
        pickle.dump(registry, f)


def get_document_registry() -> Dict[str, Dict]:
    """
    Function: get_document_registry()

    Purpose:
        Exposes the current list of processed documents (with stats)
        to the UI layer (sidebar document list).

    Output:
        Dict keyed by file hash -> document info dict.

    Used by:
        app.py -> sidebar.
    """
    if "doc_registry" not in st.session_state:
        st.session_state.doc_registry = _load_registry()
    return st.session_state.doc_registry


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_file(filename: str, file_bytes: bytes) -> str:
    """
    Function: validate_file()

    Purpose:
        Runs basic sanity checks before spending time processing a file.

    Input:
        filename: original filename.
        file_bytes: raw file bytes.

    Output:
        An empty string "" if the file is valid, otherwise a
        user-friendly error message.

    Used by:
        add_document() below.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}."

    if len(file_bytes) == 0:
        return "The uploaded file is empty."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"File is too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB."

    return ""


# ---------------------------------------------------------------------
# ADD DOCUMENT
# ---------------------------------------------------------------------

def add_document(filename: str, file_bytes: bytes) -> Dict:
    """
    Function: add_document()

    Purpose:
        The main "upload a document" workflow:
        validate -> duplicate-check -> extract+chunk -> embed -> index -> save.

    Input:
        filename: original filename.
        file_bytes: raw uploaded bytes.

    Output:
        {
            "success": bool,
            "message": str,        # shown to the user
            "duplicate": bool,
        }

    Used by:
        app.py -> file uploader handler.
    """
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


# ---------------------------------------------------------------------
# CLEAR DOCUMENTS
# ---------------------------------------------------------------------

def clear_all_documents():
    """
    Function: clear_all_documents()

    Purpose:
        Wipes the vector database AND the document registry — used by
        the sidebar's "Clear Documents" button.

    Used by:
        app.py
    """
    store = get_vector_store()
    store.clear_index()

    st.session_state.doc_registry = {}
    if os.path.exists(DOC_REGISTRY_PATH):
        os.remove(DOC_REGISTRY_PATH)


def list_document_names() -> List[str]:
    """Returns just the filenames of all currently indexed documents."""
    return [info["filename"] for info in get_document_registry().values()]
