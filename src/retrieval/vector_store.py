"""Vector store wrapper for FAISS similarity search and metadata persistence."""

import os
import pickle
from typing import List, Dict, Tuple
import faiss
import numpy as np

from src.config.settings import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIMENSION,
)


class VectorStore:
    """FAISS IndexFlatIP vector store managing embeddings and associated metadata."""

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict] = []

    def create_index(self):
        """Reset the FAISS index and clear metadata in memory."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add embedding vectors and corresponding chunk metadata to the index."""
        if len(chunks) == 0:
            return
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve top_k most similar chunks by inner product / cosine similarity."""
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save_index(self):
        """Persist FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk if present."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return True
            except Exception:
                pass
        self.create_index()
        return False

    def clear_index(self):
        """Wipe memory index and remove persisted index files from disk."""
        self.create_index()
        for path in (self.index_path, self.metadata_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @property
    def total_chunks(self) -> int:
        """Return the total number of indexed vectors."""
        return self.index.ntotal

    def chunks_for_document(self, document_name: str) -> List[Dict]:
        """Return all metadata chunks for a specific document."""
        return [m for m in self.metadata if m.get("document") == document_name]
