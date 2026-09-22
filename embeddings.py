"""Embedding generation service using Sentence Transformers."""

from typing import List
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model in memory."""
    return SentenceTransformer(EMBEDDING_MODEL)


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

