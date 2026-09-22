"""Unit tests for embedding generation, FAISS vector store, and semantic retriever."""

import os
import tempfile
import unittest
import numpy as np

from src.retrieval.embeddings import generate_embeddings, generate_query_embedding
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import retrieve_relevant_chunks


class TestRetrieval(unittest.TestCase):
    def test_embeddings_generation(self):
        chunks = [{"text": "Machine learning algorithms learn from data."}]
        vectors = generate_embeddings(chunks)
        self.assertEqual(vectors.shape, (1, 384))
        self.assertEqual(vectors.dtype, np.float32)

        norm = np.linalg.norm(vectors[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_query_embedding_generation(self):
        vector = generate_query_embedding("What is machine learning?")
        self.assertEqual(vector.shape, (1, 384))
        self.assertEqual(vector.dtype, np.float32)

        norm = np.linalg.norm(vector[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_vector_store_add_search_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "index.faiss")
            meta_path = os.path.join(tmpdir, "metadata.pkl")

            store = VectorStore(index_path=index_path, metadata_path=meta_path)
            chunks = [
                {"document": "ai.txt", "page": 1, "chunk_id": 0, "text": "Artificial intelligence and deep learning."},
                {"document": "food.txt", "page": 1, "chunk_id": 1, "text": "Cooking pasta requires boiling water."},
            ]
            embeddings = generate_embeddings(chunks)
            store.add_documents(chunks, embeddings)
            self.assertEqual(store.total_chunks, 2)

            q_vec = generate_query_embedding("neural networks and AI")
            results = store.search(q_vec, top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0]["document"], "ai.txt")

            store.save_index()
            self.assertTrue(os.path.exists(index_path))
            self.assertTrue(os.path.exists(meta_path))

            store2 = VectorStore(index_path=index_path, metadata_path=meta_path)
            loaded = store2.load_index()
            self.assertTrue(loaded)
            self.assertEqual(store2.total_chunks, 2)

            relevant = retrieve_relevant_chunks("neural networks", store2, top_k=2, threshold=0.1)
            self.assertGreaterEqual(len(relevant), 1)

            store2.clear_index()
            self.assertEqual(store2.total_chunks, 0)


if __name__ == "__main__":
    unittest.main()
