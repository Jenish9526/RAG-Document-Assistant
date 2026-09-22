"""Sliding-window chunker with word-level boundary handling."""

from typing import List, Dict
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.helpers import clean_text


def split_into_chunks(
    pages: List[Dict],
    document_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Split extracted page text into overlapping word-level chunks with metadata."""
    chunks = []
    chunk_id = 0
    step = max(chunk_size - chunk_overlap, 1)

    for page in pages:
        cleaned = clean_text(page["text"])
        words = cleaned.split(" ")

        if not words:
            continue

        for start in range(0, len(words), step):
            window = words[start:start + chunk_size]
            if not window:
                continue
            chunk_text = " ".join(window).strip()
            if len(chunk_text) < 10:
                continue

            chunks.append({
                "document": document_name,
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk_text,
            })
            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks
