"""Retrieval module for semantic search filtering, multi-query support, and ranking."""

import re
from typing import List, Dict, Tuple
from src.config.settings import TOP_K, SIMILARITY_THRESHOLD
from src.retrieval.embeddings import generate_query_embedding
from src.retrieval.vector_store import VectorStore


def retrieve_relevant_chunks(
    query: str, store: VectorStore, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[Dict, float]]:
    """Retrieve chunks scoring above threshold for a search query, ordered by similarity.
    
    If multiple questions are detected in the query, searches across sub-queries to ensure
    all requested topics obtain relevant source excerpts without omission.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # Detect if multiple questions / queries exist (numbered list, multiple '?' marks, bullet points)
    sub_queries = [clean_query]
    
    # Split on question marks followed by space or line breaks, or numbered bullets
    parts = re.split(r"(?<=\?)\s+|\n+(?:\d+[\.\)]\s*|[-•*]\s*|\bQ\d+[:\.]\s*)", clean_query)
    for p in parts:
        cleaned_p = p.strip()
        # Clean leading numbers or bullets like '1. ', '2) '
        cleaned_p = re.sub(r"^(?:\d+[\.\)]\s*|[-•*]\s*|\bQ\d+[:\.]\s*)", "", cleaned_p).strip()
        if len(cleaned_p) >= 12 and cleaned_p not in sub_queries:
            sub_queries.append(cleaned_p)

    # Search vector store for each query component and merge results
    merged_results: Dict[Tuple[str, int, int], Tuple[Dict, float]] = {}

    for sq in sub_queries:
        try:
            query_vector = generate_query_embedding(sq)
            raw_results = store.search(query_vector, top_k=top_k)
            for chunk, score in raw_results:
                if score >= threshold:
                    key = (chunk.get("document", ""), chunk.get("page", 0), chunk.get("chunk_id", -1))
                    if key not in merged_results or score > merged_results[key][1]:
                        merged_results[key] = (chunk, score)
        except Exception:
            continue

    if not merged_results:
        # Fallback to direct search if sub-query parsing didn't match threshold
        query_vector = generate_query_embedding(clean_query)
        raw_results = store.search(query_vector, top_k=top_k)
        return [(chunk, score) for chunk, score in raw_results if score >= threshold]

    # Sort descending by score
    sorted_chunks = sorted(merged_results.values(), key=lambda item: item[1], reverse=True)
    return sorted_chunks[:top_k]

