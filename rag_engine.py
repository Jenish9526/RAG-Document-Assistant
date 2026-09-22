"""RAG execution engine handling semantic retrieval, prompt assembly, and answer synthesis."""

import time
from typing import List, Dict, Tuple

from config import TOP_K, SIMILARITY_THRESHOLD
from embeddings import generate_query_embedding
from vector_store import VectorStore
import llm_service


def retrieve_relevant_chunks(
    query: str, store: VectorStore, top_k: int = TOP_K
) -> List[Tuple[Dict, float]]:
    """Retrieve chunks scoring above SIMILARITY_THRESHOLD for a user query."""
    query_vector = generate_query_embedding(query)
    raw_results = store.search(query_vector, top_k=top_k)
    return [(chunk, score) for chunk, score in raw_results if score >= SIMILARITY_THRESHOLD]


def build_context(retrieved: List[Tuple[Dict, float]]) -> str:
    """Format retrieved document chunks into labeled context blocks."""
    blocks = []
    for i, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[Source {i}: {chunk['document']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str, answer_style: str = "Simple", exam_mode: bool = False) -> str:
    """Assemble the system instructions, context, and user question into an LLM prompt."""
    style_instruction = (
        "Use easy language, short paragraphs, and bullet points where useful. "
        "Avoid unnecessary technical jargon."
        if answer_style == "Simple" else
        "Provide a more thorough explanation, including relevant technical details "
        "and examples if present in the context."
    )


    exam_instruction = ""
    if exam_mode:
        exam_instruction = (
            "\nStructure the answer using ONLY the sections that are relevant to the "
            "question, chosen from: Definition, Explanation, Steps, Example, "
            "Advantages, Disadvantages, Complexity.\n"
        )

    return f"""You are a document assistant. Answer the user's question using ONLY the provided document context below.

Rules:
1. Prefer information from the provided context above your own general knowledge.
2. Do not invent facts that are not supported by the context.
3. If the answer is not present in the context, clearly say: "I couldn't find enough information about this topic in the uploaded documents."
4. {style_instruction}
5. Do not claim information comes from the documents if it does not appear in the context.
{exam_instruction}
Document Context:
---
{context}
---

Question: {query}

Answer:"""


def generate_answer(prompt: str) -> str:
    """Invoke LLM service to produce an answer for the given prompt."""
    return llm_service.generate_response(prompt)


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
    except llm_service.LLMNotConfiguredError as exc:
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


def summarize_document(document_name: str, store: VectorStore, max_chunks_per_batch: int = 6) -> str:
    """Generate structured summary of a document using chunk-based map-reduce aggregation."""
    chunks = store.chunks_for_document(document_name)
    if not chunks:
        return "No content found for this document."

    chunks = sorted(chunks, key=lambda c: (c["page"], c["chunk_id"]))

    partial_summaries = []
    for i in range(0, len(chunks), max_chunks_per_batch):
        if i > 0:
            time.sleep(1.5)
        batch = chunks[i:i + max_chunks_per_batch]
        batch_text = "\n\n".join(c["text"] for c in batch)
        batch_prompt = (
            "Summarize the following document excerpt in 3-5 concise sentences, "
            "focusing on the main ideas and any important definitions:\n\n"
            f"{batch_text}\n\nSummary:"
        )
        try:
            partial_summaries.append(generate_answer(batch_prompt))
        except (llm_service.LLMNotConfiguredError, RuntimeError) as exc:
            return f"Could not generate summary: {exc}"

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
    except (llm_service.LLMNotConfiguredError, RuntimeError) as exc:
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
    if not chunks or not llm_service.is_configured():
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
    except (llm_service.LLMNotConfiguredError, RuntimeError):
        return fallback

