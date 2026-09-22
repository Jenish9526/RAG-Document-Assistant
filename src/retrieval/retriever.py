"""Retrieval module for semantic search filtering and ranking."""

from typing import List, Dict, Tuple
from src.config.settings import TOP_K, SIMILARITY_THRESHOLD
from src.retrieval.embeddings import generate_query_embedding
from src.retrieval.vector_store import VectorStore


def retrieve_relevant_chunks(
    query: str, store: VectorStore, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[Dict, float]]:
    """Retrieve chunks scoring above threshold for a search query, ordered by similarity."""
    query_vector = generate_query_embedding(query)
    raw_results = store.search(query_vector, top_k=top_k)
    return [(chunk, score) for chunk, score in raw_results if score >= threshold]
