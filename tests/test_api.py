"""Integration tests for FastAPI REST endpoints."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.routes import app


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("llm_configured", data)
        self.assertIn("model", data)

    def test_list_documents(self):
        response = self.client.get("/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("count", data)
        self.assertIn("documents", data)

    @patch("src.api.routes.answer_question")
    def test_query_endpoint(self, mock_answer):
        mock_answer.return_value = {
            "answer": "Test answer from model.",
            "found_context": True,
            "sources": [{"document": "test.pdf", "page": 1, "score": 0.85}],
        }
        response = self.client.post("/query", json={"query": "test query", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "test query")
        self.assertEqual(data["answer"], "Test answer from model.")
        self.assertTrue(data["found_context"])
        self.assertEqual(len(data["sources"]), 1)

    @patch("src.api.routes.summarize_document")
    def test_summarize_endpoint(self, mock_sum):
        mock_sum.return_value = "Summary text from mock."
        response = self.client.post("/documents/test.txt/summarize", json={"max_chunks_per_batch": 5})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document"], "test.txt")
        self.assertEqual(data["summary"], "Summary text from mock.")


if __name__ == "__main__":
    unittest.main()

