"""Embedding generation service using Sentence Transformers."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import EMBEDDING_MODEL

_cached_embedding_model: Optional[SentenceTransformer] = None


def load_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model in memory."""
    global _cached_embedding_model
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            @st.cache_resource(show_spinner=False)
            def _st_model():
                return SentenceTransformer(EMBEDDING_MODEL)
            return _st_model()
    except Exception:
        pass

    if _cached_embedding_model is None:
        _cached_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _cached_embedding_model


def generate_embeddings(chunks: List[dict]) -> np.ndarray:
    """Generate normalized float32 embedding vectors for document chunks."""
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.astype("float32")


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate a single normalized float32 embedding vector for a search query."""
    model = load_embedding_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector.astype("float32")
