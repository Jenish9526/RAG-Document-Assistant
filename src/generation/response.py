"""Response synthesis module for question answering, summarization, and suggested queries."""

import time
from typing import List, Dict

from src.config.settings import TOP_K
from src.retrieval.retriever import retrieve_relevant_chunks
from src.retrieval.vector_store import VectorStore
from src.generation.prompt import build_context, build_prompt
from src.generation.llm import generate_response, LLMNotConfiguredError, is_configured


def generate_answer(prompt: str) -> str:
    """Invoke LLM service to produce an answer for the given prompt."""
    return generate_response(prompt)


def answer_question(
    query: str,
    store: VectorStore,
    top_k: int = TOP_K,
    answer_style: str = "Simple",
    exam_mode: bool = False,
) -> Dict:
    """Execute end-to-end RAG question answering pipeline."""
    query = query.strip()
    if not query:
        return {
            "answer": "Please enter a question.",
            "sources": [], "chunks": [], "found_context": False,
        }

    if store.total_chunks == 0:
        return {
            "answer": "Please upload a document before asking a question.",
            "sources": [], "chunks": [], "found_context": False,
        }

    retrieved = retrieve_relevant_chunks(query, store, top_k=top_k)

    if not retrieved:
        return {
            "answer": (
                "No sufficiently relevant information was found in the uploaded "
                "documents for this question. Try rephrasing, or upload a document "
                "that covers this topic."
            ),
            "sources": [], "chunks": [], "found_context": False,
        }

    context = build_context(retrieved)
    prompt = build_prompt(query, context, answer_style=answer_style, exam_mode=exam_mode)

    try:
        answer_text = generate_answer(prompt)
    except LLMNotConfiguredError as exc:
        return {"answer": str(exc), "sources": [], "chunks": [], "found_context": False}
    except RuntimeError as exc:
        return {"answer": f"Something went wrong while generating the answer: {exc}",
                "sources": [], "chunks": [], "found_context": False}

    sources = [
        {"document": chunk["document"], "page": chunk["page"], "score": round(score, 3)}
        for chunk, score in retrieved
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": [c for c, _s in retrieved],
        "found_context": True,
    }


def summarize_document(
    document_name: str, store: VectorStore, max_chunks_per_batch: int = 24
) -> str:
    """Generate structured summary of a document using adaptive single-pass or map-reduce aggregation."""
    chunks = store.chunks_for_document(document_name)
    if not chunks:
        return "No content found for this document."

    chunks = sorted(chunks, key=lambda c: (c["page"], c["chunk_id"]))

    # For standard documents, summarize in a single prompt to preserve context and avoid rate-limiting
    if len(chunks) <= max_chunks_per_batch:
        full_text = "\n\n".join(c["text"] for c in chunks)
        prompt = f"""Based on the following content from "{document_name}", provide a comprehensive, well-structured summary formatted with these sections:

1. Main Topic
2. Important Concepts
3. Key Points
4. Important Definitions & Practical Takeaways

Document Content:
---
{full_text}
---

Structured Summary:"""
        try:
            return generate_answer(prompt)
        except (LLMNotConfiguredError, RuntimeError) as exc:
            return f"Could not generate summary: {exc}"

    # For very large documents, aggregate across larger chunks with a safe pacing delay
    partial_summaries = []
    for i in range(0, len(chunks), max_chunks_per_batch):
        if i > 0:
            time.sleep(3.0)
        batch = chunks[i:i + max_chunks_per_batch]
        batch_text = "\n\n".join(c["text"] for c in batch)
        batch_prompt = (
            "Summarize the following document excerpt in concise points, "
            "focusing on the main ideas and any important definitions:\n\n"
            f"{batch_text}\n\nSummary:"
        )
        try:
            partial_summaries.append(generate_answer(batch_prompt))
        except (LLMNotConfiguredError, RuntimeError) as exc:
            return f"Could not generate summary: {exc}"

    time.sleep(2.0)
    combined = "\n\n".join(partial_summaries)
    final_prompt = f"""Based on the following partial summaries of a document called "{document_name}", write one combined, well-structured summary with these sections:

1. Main Topic
2. Important Concepts
3. Key Points
4. Important Definitions

Partial summaries:
---
{combined}
---

Final Summary:"""
    try:
        return generate_answer(final_prompt)
    except (LLMNotConfiguredError, RuntimeError) as exc:
        return f"Could not generate final summary: {exc}"


def generate_suggested_questions(document_name: str, store: VectorStore) -> List[str]:
    """Generate suggested exploration questions for an indexed document with static fallbacks."""
    fallback = [
        "What is the main topic of this document?",
        "Explain the important concepts covered here.",
        "What are the key definitions in this document?",
        "Give me a summary of this document.",
        "What are the key takeaways from this document?",
    ]

    chunks = store.chunks_for_document(document_name)
    if not chunks or not is_configured():
        return fallback

    sample_text = "\n\n".join(c["text"] for c in chunks[:4])
    prompt = (
        "Based on the following document excerpt, suggest exactly 5 short, useful "
        "questions a reader might ask about it. Return ONLY the 5 questions, "
        "one per line, no numbering, no extra text.\n\n"
        f"{sample_text}\n\nQuestions:"
    )
    try:
        raw = generate_answer(prompt)
        questions = [q.strip("-•0123456789. ").strip() for q in raw.split("\n") if q.strip()]
        questions = [q for q in questions if q]
        return questions[:5] if questions else fallback
    except (LLMNotConfiguredError, RuntimeError):
        return fallback
