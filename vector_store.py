"""
vector_store.py
================

Purpose
-------
Wraps FAISS (Facebook AI Similarity Search) so the rest of the app never
has to deal with FAISS's low-level API directly.

Simple explanation:
    FAISS is like a super-fast filing cabinet for embedding vectors.
    Instead of searching by keyword, you search by "which vectors are
    closest in meaning to this one?".

Technical explanation:
    FAISS only stores raw float vectors and returns integer positions
    ("vector IDs") — it has no idea what a "document" or "page" is.
    So this module keeps a PARALLEL Python list called `metadata`,
    where metadata[i] describes exactly what vector i represents
    (document name, page, chunk text). Index `i` in the FAISS index
    always lines up with index `i` in the metadata list.

Imported libraries
-------------------
- faiss  : the similarity search library itself.
- pickle : used to save/load the metadata list to disk.
- numpy  : vector math.

Used by
-------
document_manager.py (adding documents) and rag_engine.py (searching).
"""

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIMENSION,
)


class VectorStore:
    """
    A thin, student-friendly wrapper around a FAISS index + its metadata.

    We use IndexFlatIP (Inner Product) together with L2-normalized
    embeddings (see embeddings.py: normalize_embeddings=True). For
    normalized vectors, inner product is mathematically equivalent to
    cosine similarity — this is a common, simple, and exact approach
    (no approximation), which is perfect for a college-project-sized
    document collection.
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata: List[Dict] = []   # metadata[i] <-> vector at index i

    # -------------------------------------------------------------
    # CREATE / ADD
    # -------------------------------------------------------------
    def create_index(self):
        """
        Function: create_index()

        Purpose:
            Resets the FAISS index to a brand-new, empty index.

        Used by:
            clear_index() below, and __init__.
        """
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Function: add_documents()

        Purpose:
            Adds a batch of chunk embeddings (and their metadata) to the
            FAISS index.

        Input:
            chunks:     list of chunk dicts (document, page, chunk_id, text).
            embeddings: numpy array of shape (len(chunks), 384), matching
                        the order of `chunks`.

        Output:
            None (updates the index and metadata list in place).

        Used by:
            document_manager.py -> add_document()
        """
        if len(chunks) == 0:
            return
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    # -------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Function: search()

        Purpose:
            Finds the `top_k` chunks whose embeddings are most similar
            to the query embedding.

        Input:
            query_embedding: numpy array of shape (1, 384).
            top_k: how many results to return.

        Output:
            A list of (chunk_metadata_dict, similarity_score) tuples,
            best match first. similarity_score is in range [-1, 1]
            (cosine similarity); higher = more relevant.

        Used by:
            rag_engine.py -> retrieve_relevant_chunks()
        """
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

    # -------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------
    def save_index(self):
        """
        Function: save_index()

        Purpose:
            Persists the FAISS index and metadata to disk so the
            document database survives an app restart.

        Used by:
            document_manager.py -> add_document(), delete/clear operations.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """
        Function: load_index()

        Purpose:
            Loads a previously saved FAISS index + metadata from disk,
            if one exists. Called once when the app starts.

        Output:
            True if an existing index was loaded, False if none was found
            (in which case an empty index is created instead).

        Used by:
            document_manager.py -> get_vector_store() (app startup).
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        self.create_index()
        return False

    def clear_index(self):
        """
        Function: clear_index()

        Purpose:
            Wipes the index (in memory) AND deletes the saved files on
            disk. Used by the "Clear Documents" button in the sidebar.

        Used by:
            document_manager.py -> clear_all_documents()
        """
        self.create_index()
        for path in (self.index_path, self.metadata_path):
            if os.path.exists(path):
                os.remove(path)

    # -------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------
    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    def chunks_for_document(self, document_name: str) -> List[Dict]:
        """Returns all stored chunks belonging to one document (used for summarization)."""
        return [m for m in self.metadata if m["document"] == document_name]
