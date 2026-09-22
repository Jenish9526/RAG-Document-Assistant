"""Unit tests for prompt formatting, LLM error handling, and response synthesis."""

import unittest
from unittest.mock import patch

from src.generation.prompt import build_context, build_prompt
from src.generation.llm import LLMNotConfiguredError
from src.generation.response import answer_question, summarize_document, generate_suggested_questions
from src.retrieval.vector_store import VectorStore


class TestGeneration(unittest.TestCase):
    def test_build_context(self):
        retrieved = [
            ({"document": "doc1.pdf", "page": 2, "text": "Content of chunk 1."}, 0.85),
            ({"document": "doc2.txt", "page": 1, "text": "Content of chunk 2."}, 0.72),
        ]
        context = build_context(retrieved)
        self.assertIn("[Source 1: doc1.pdf, Page 2]", context)
        self.assertIn("Content of chunk 1.", context)
        self.assertIn("[Source 2: doc2.txt, Page 1]", context)

    def test_build_prompt_variants(self):
        context = "[Source 1: algo.txt, Page 1]\nBinary search runs in O(log n)."
        prompt_simple = build_prompt("Explain binary search.", context, answer_style="Simple")
        self.assertIn("easy language", prompt_simple)

        prompt_detailed = build_prompt("Explain binary search.", context, answer_style="Detailed")
        self.assertIn("more thorough explanation", prompt_detailed)

        prompt_exam = build_prompt("Explain binary search.", context, exam_mode=True)
        self.assertIn("Definition, Explanation, Steps", prompt_exam)

    def test_answer_question_validation(self):
        store = VectorStore()
        res_empty = answer_question("", store)
        self.assertIn("Please enter a question", res_empty["answer"])

        res_no_docs = answer_question("What is AI?", store)
        self.assertIn("Please upload a document", res_no_docs["answer"])

    @patch("src.generation.response.generate_answer")
    def test_answer_question_flow(self, mock_generate):
        mock_generate.return_value = "Binary search is an efficient search algorithm."
        store = VectorStore()
        chunks = [{"document": "search.txt", "page": 1, "chunk_id": 0, "text": "Binary search divides the interval in half."}]
        # mock search returning relevant result
        store.search = lambda vec, top_k=5: [(chunks[0], 0.88)]
        store.index.ntotal = 1

        result = answer_question("How does binary search work?", store, top_k=1)
        self.assertTrue(result["found_context"])
        self.assertEqual(result["answer"], "Binary search is an efficient search algorithm.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["document"], "search.txt")

    @patch("src.generation.response.generate_answer")
    def test_summarize_document(self, mock_generate):
        mock_generate.return_value = "Summary: Document covers algorithms."
        store = VectorStore()
        store.chunks_for_document = lambda doc_name: [
            {"document": doc_name, "page": 1, "chunk_id": 0, "text": "Sample text."}
        ]
        summary = summarize_document("doc.txt", store)
        self.assertEqual(summary, "Summary: Document covers algorithms.")

    def test_suggested_questions_fallback(self):
        store = VectorStore()
        questions = generate_suggested_questions("empty.txt", store)
        self.assertIsInstance(questions, list)
        self.assertGreaterEqual(len(questions), 1)


if __name__ == "__main__":
    unittest.main()
