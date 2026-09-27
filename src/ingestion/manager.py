"""Document management module handling document upload, registry, and indexing."""

import os
from typing import Dict, List, Optional

from src.config.settings import (
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
)
from src.ingestion.parser import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx,
)
from src.ingestion.chunker import split_into_chunks

from src.retrieval.embeddings import generate_embeddings
from src.retrieval.vector_store import VectorStore
from src.utils.helpers import compute_file_hash, format_file_size
from src.utils.logger import get_logger

logger = get_logger("ingestion.manager")
_global_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return session-isolated VectorStore when in Streamlit, or in-memory instance."""
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists() and hasattr(st, "session_state"):
            if "session_vector_store" not in st.session_state:
                st.session_state.session_vector_store = VectorStore()
            return st.session_state.session_vector_store
    except Exception:
        pass

    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = VectorStore()
    return _global_vector_store


def process_document(file_bytes: bytes, filename: str) -> Dict:
    """Validate format, extract text, and chunk document into indexed fragments."""
    lower_name = filename.lower()

    try:
        if lower_name.endswith(".pdf"):
            pages = extract_text_from_pdf(file_bytes)
        elif lower_name.endswith(".txt"):
            pages = extract_text_from_txt(file_bytes)
        elif lower_name.endswith(".docx"):
            pages = extract_text_from_docx(file_bytes)
        else:
            return {"success": False, "error": "Unsupported file type.", "pages": 0,
                    "characters": 0, "chunks": []}
    except Exception as exc:
        return {"success": False, "error": f"Could not read file: {exc}",
                "pages": 0, "characters": 0, "chunks": []}

    if not pages:
        return {
            "success": False,
            "error": "The uploaded document contains no readable text.",
            "pages": 0, "characters": 0, "chunks": [],
        }

    total_characters = sum(len(p["text"]) for p in pages)
    chunks = split_into_chunks(pages, document_name=filename)

    if not chunks:
        return {
            "success": False,
            "error": "The document contains text, but no usable chunks could be created.",
            "pages": len(pages), "characters": total_characters, "chunks": [],
        }

    return {
        "success": True,
        "error": None,
        "pages": len(pages),
        "characters": total_characters,
        "chunks": chunks,
    }


_global_doc_registry: Dict[str, Dict] = {}


def get_document_registry() -> Dict[str, Dict]:
    """Retrieve the session-isolated document registry."""
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists() and hasattr(st, "session_state"):
            if "doc_registry" not in st.session_state:
                st.session_state.doc_registry = {}
            return st.session_state.doc_registry
    except Exception:
        pass
    global _global_doc_registry
    return _global_doc_registry


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


def add_document(filename: str, file_bytes: bytes, store: Optional[VectorStore] = None) -> Dict:
    """Validate, deduplicate, chunk, embed, and index a new uploaded document in-memory."""
    error = validate_file(filename, file_bytes)
    if error:
        return {"success": False, "message": error, "duplicate": False}

    file_hash = compute_file_hash(file_bytes)
    registry = get_document_registry()

    if file_hash in registry or (store and store.chunks_for_document(filename)):
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

    active_store = store or get_vector_store()
    active_store.add_documents(chunks, embeddings_matrix)

    registry[file_hash] = {
        "filename": filename,
        "pages": result["pages"],
        "characters": result["characters"],
        "chunks": len(chunks),
        "size": format_file_size(len(file_bytes)),
    }

    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists() and hasattr(st, "session_state"):
            st.session_state.doc_registry = registry
    except Exception:
        pass

    logger.info("Added '%s' with %d chunks to session vector store", filename, len(chunks))

    return {
        "success": True,
        "message": f"'{filename}' processed successfully — {result['pages']} pages, "
                    f"{len(chunks)} chunks indexed.",
        "duplicate": False,
    }


def clear_all_documents(store: Optional[VectorStore] = None):
    """Clear session vector store and document registry."""
    active_store = store or get_vector_store()
    active_store.clear_index()

    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists() and hasattr(st, "session_state"):
            st.session_state.doc_registry = {}
            if "session_vector_store" in st.session_state:
                st.session_state.session_vector_store = VectorStore()
    except Exception:
        pass

    logger.info("Cleared session indexed documents")


def remove_document(document_name: str, store: Optional[VectorStore] = None) -> bool:
    """Remove a single document from session vector store and document registry."""
    active_store = store or get_vector_store()
    active_store.remove_document(document_name)

    registry = get_document_registry()
    hashes_to_remove = [h for h, info in registry.items() if info.get("filename") == document_name]
    for h in hashes_to_remove:
        del registry[h]

    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists() and hasattr(st, "session_state"):
            st.session_state.doc_registry = registry
    except Exception:
        pass

    logger.info("Removed '%s' from session vector store and registry", document_name)
    return True


def list_document_names() -> List[str]:
    """Return filenames of all currently indexed documents in this session."""
    return [info["filename"] for info in get_document_registry().values()]
