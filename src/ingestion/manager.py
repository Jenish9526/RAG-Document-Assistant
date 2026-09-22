"""Document management module handling document upload, registry, and indexing."""

import os
import pickle
from typing import Dict, List, Optional

from src.config.settings import (
    DOC_REGISTRY_PATH,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    FAISS_INDEX_PATH,
    METADATA_PATH,
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


def get_vector_store(
    index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_PATH
) -> VectorStore:
    """Return the active VectorStore instance, utilizing Streamlit cache if in UI context."""
    global _global_vector_store
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            @st.cache_resource(show_spinner=False)
            def _cached_store():
                s = VectorStore(index_path=index_path, metadata_path=metadata_path)
                s.load_index()
                return s
            return _cached_store()
    except Exception:
        pass

    if _global_vector_store is None:
        _global_vector_store = VectorStore(index_path=index_path, metadata_path=metadata_path)
        _global_vector_store.load_index()
    return _global_vector_store


def process_document(file_bytes: bytes, filename: str) -> Dict:
    """Validate format, extract text, and chunk document into indexed fragments."""
    lower_name = filename.lower()

    try:
        if lower_name.endswith(".pdf"):
            from src.ingestion.parser import extract_text_from_pdf
            pages = extract_text_from_pdf(file_bytes)
        elif lower_name.endswith(".txt"):
            from src.ingestion.parser import extract_text_from_txt
            pages = extract_text_from_txt(file_bytes)
        elif lower_name.endswith(".docx"):
            from src.ingestion.parser import extract_text_from_docx
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


def _load_registry() -> Dict[str, Dict]:
    """Load the document registry from disk if valid index files exist."""
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        if os.path.exists(DOC_REGISTRY_PATH):
            try:
                os.remove(DOC_REGISTRY_PATH)
            except OSError:
                pass
        return {}
    if os.path.exists(DOC_REGISTRY_PATH):
        try:
            with open(DOC_REGISTRY_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, Dict]):
    """Persist the document registry mapping to disk."""
    with open(DOC_REGISTRY_PATH, "wb") as f:
        pickle.dump(registry, f)


def get_document_registry() -> Dict[str, Dict]:
    """Retrieve the in-memory or persisted document registry."""
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "doc_registry" in st.session_state:
            return st.session_state.doc_registry
    except Exception:
        pass
    return _load_registry()


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

    active_store = store or get_vector_store()
    active_store.add_documents(chunks, embeddings_matrix)
    active_store.save_index()

    registry[file_hash] = {
        "filename": filename,
        "pages": result["pages"],
        "characters": result["characters"],
        "chunks": len(chunks),
        "size": format_file_size(len(file_bytes)),
    }

    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            st.session_state.doc_registry = registry
    except Exception:
        pass

    _save_registry(registry)
    logger.info("Added '%s' with %d chunks to vector store", filename, len(chunks))

    return {
        "success": True,
        "message": f"'{filename}' processed successfully — {result['pages']} pages, "
                    f"{len(chunks)} chunks indexed.",
        "duplicate": False,
    }


def clear_all_documents(store: Optional[VectorStore] = None):
    """Clear in-memory and on-disk vector store and document registry."""
    active_store = store or get_vector_store()
    active_store.clear_index()

    try:
        import streamlit as st
        if hasattr(st, "session_state") and "doc_registry" in st.session_state:
            st.session_state.doc_registry = {}
    except Exception:
        pass

    if os.path.exists(DOC_REGISTRY_PATH):
        try:
            os.remove(DOC_REGISTRY_PATH)
        except OSError:
            pass
    logger.info("Cleared all indexed documents")


def list_document_names() -> List[str]:
    """Return filenames of all currently indexed documents."""
    return [info["filename"] for info in get_document_registry().values()]
