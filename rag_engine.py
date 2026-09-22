"""
rag_engine.py
=============

Purpose
-------
The CENTRAL module of the whole project. It connects every other piece:

    User question
        -> generate_query_embedding()      (embeddings.py)
        -> vector_store.search()           (vector_store.py)
        -> retrieve_relevant_chunks()
        -> build_context()
        -> build_prompt()
        -> llm_service.generate_response() (llm_service.py)
        -> answer_question() returns final answer + sources

This is the file that most clearly demonstrates "Retrieval-Augmented
Generation": RETRIEVAL (finding relevant chunks) followed by
GENERATION (asking the LLM to write an answer using those chunks).

Used by
-------
app.py -> whenever the user submits a question, a document summary
request, or asks for suggested questions.
"""

import time
from typing import List, Dict, Tuple

from config import TOP_K, SIMILARITY_THRESHOLD
from embeddings import generate_query_embedding
from vector_store import VectorStore
import llm_service


# ---------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str, store: VectorStore, top_k: int = TOP_K
) -> List[Tuple[Dict, float]]:
    """
    Function: retrieve_relevant_chunks()

    Purpose:
        Converts the question into an embedding and searches the FAISS
        vector store for the most semantically similar chunks.

    Input:
        query: the user's question.
        store: the active VectorStore instance.
        top_k: how many chunks to retrieve.

    Output:
        A list of (chunk_dict, similarity_score) tuples, sorted best-first.
        Chunks scoring below SIMILARITY_THRESHOLD are filtered out —
        this is a key part of hallucination control (see section below).

    Used by:
        answer_question() in this file.
    """
    query_vector = generate_query_embedding(query)
    raw_results = store.search(query_vector, top_k=top_k)

    # Filter out weakly-related chunks so we don't force the LLM to
    # "make something up" from irrelevant context.
    relevant = [(chunk, score) for chunk, score in raw_results if score >= SIMILARITY_THRESHOLD]
    return relevant


# ---------------------------------------------------------------------
# CONTEXT + PROMPT CONSTRUCTION
# ---------------------------------------------------------------------

def build_context(retrieved: List[Tuple[Dict, float]]) -> str:
    """
    Function: build_context()

    Purpose:
        Formats retrieved chunks into one text block the LLM can read,
        clearly labeling which document/page each piece came from
        (so the LLM can reference sources in its answer if asked to).

    Input:
        retrieved: output of retrieve_relevant_chunks().

    Output:
        A single formatted string, e.g.:
            [Source 1: DAA.pdf, Page 12]
            Dynamic programming is...

            [Source 2: DAA.pdf, Page 13]
            ...

    Used by:
        build_prompt() below.
    """
    blocks = []
    for i, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[Source {i}: {chunk['document']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str, answer_style: str = "Simple", exam_mode: bool = False) -> str:
    """
    Function: build_prompt()

    Purpose:
        Builds the final instruction text sent to the LLM, combining:
        - a strict system-style instruction (rules to reduce hallucination)
        - the retrieved document context
        - the user's actual question
        - the requested answer style (Simple / Detailed) and Exam Mode.

    Input:
        query: user's question.
        context: output of build_context().
        answer_style: "Simple" or "Detailed".
        exam_mode: if True, ask for an exam-ready structured answer.

    Output:
        The complete prompt string ready to send to llm_service.generate_response().

    Used by:
        generate_answer() below.
    """
    style_instruction = (
        "Use easy language, short paragraphs, and bullet points where useful. "
        "Avoid unnecessary technical jargon."
        if answer_style == "Simple" else
        "Provide a more thorough explanation, including relevant technical detail "
        "and examples if present in the context. Make it suitable for exam revision."
    )

    exam_instruction = ""
    if exam_mode:
        exam_instruction = (
            "\nStructure the answer using ONLY the sections that are relevant to the "
            "question, chosen from: Definition, Explanation, Steps, Example, "
            "Advantages, Disadvantages, Complexity.\n"
        )

    prompt = f"""You are a document assistant. Answer the user's question using ONLY the provided document context below.

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
    return prompt


# ---------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    """
    Function: generate_answer()

    Purpose:
        Thin wrapper around llm_service.generate_response(), kept
        separate so rag_engine.py never imports "requests" directly —
        all HTTP/API concerns stay inside llm_service.py.

    Input:
        prompt: full prompt string from build_prompt().

    Output:
        LLM-generated answer text.

    Used by:
        answer_question() below.
    """
    return llm_service.generate_response(prompt)


# ---------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------

def answer_question(
    query: str,
    store: VectorStore,
    top_k: int = TOP_K,
    answer_style: str = "Simple",
    exam_mode: bool = False,
) -> Dict:
    """
    Function: answer_question()

    Purpose:
        The single function app.py calls for every chat message. Runs
        the full RAG pipeline end-to-end.

    Input:
        query: user's question.
        store: active VectorStore.
        top_k: number of chunks to retrieve.
        answer_style: "Simple" or "Detailed".
        exam_mode: bool, whether to use exam-structured answers.

    Output:
        {
            "answer": "<generated text>",
            "sources": [ {"document":.., "page":.., "score":..}, ... ],
            "chunks": [ ...raw retrieved chunk dicts... ],
            "found_context": True/False
        }

    Used by:
        app.py -> chat input handler.
    """
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


# ---------------------------------------------------------------------
# DOCUMENT SUMMARIZATION (chunk-based, "map-reduce" style)
# ---------------------------------------------------------------------

def summarize_document(document_name: str, store: VectorStore, max_chunks_per_batch: int = 6) -> str:
    """
    Function: summarize_document()

    Purpose:
        Generates a structured summary of one document WITHOUT sending
        the entire document to the LLM in one go (which could exceed
        token limits for large documents).

    How it works (map-reduce style):
        Document -> chunks (already stored in FAISS metadata)
                 -> grouped into small batches
                 -> each batch summarized separately ("map")
                 -> partial summaries combined into one final summary ("reduce")

    Input:
        document_name: the filename to summarize.
        store: active VectorStore (chunks are read from its metadata).
        max_chunks_per_batch: how many chunks to summarize per LLM call.

    Output:
        A single formatted summary string covering: Main Topic,
        Important Concepts, Key Points, Important Definitions.

    Used by:
        app.py -> "Summarize Document" button.
    """
    chunks = store.chunks_for_document(document_name)
    if not chunks:
        return "No content found for this document."

    # Sort by page, then chunk_id, so the summary follows reading order.
    chunks = sorted(chunks, key=lambda c: (c["page"], c["chunk_id"]))

    # --- MAP step: summarize each batch of chunks ---
    partial_summaries = []
    for i in range(0, len(chunks), max_chunks_per_batch):
        if i > 0:
            time.sleep(1.5)  # Pace batch calls to stay within free-tier TPM limits
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

    # --- REDUCE step: combine partial summaries into a final structured summary ---
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


# ---------------------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------------------

def generate_suggested_questions(document_name: str, store: VectorStore) -> List[str]:
    """
    Function: generate_suggested_questions()

    Purpose:
        Suggests a handful of questions the user might want to ask
        about a given document, using a sample of its content.
        Falls back to static generic questions if the LLM is not
        configured or the call fails, so this feature never crashes the app.

    Input:
        document_name: filename to generate suggestions for.
        store: active VectorStore.

    Output:
        A list of up to 5 question strings.

    Used by:
        app.py -> "Suggested Questions" button, shown after upload.
    """
    fallback = [
        "What is the main topic of this document?",
        "Explain the important concepts covered here.",
        "What are the key definitions in this document?",
        "Give me a summary of this document.",
        "What are the most important topics for an exam?",
    ]

    chunks = store.chunks_for_document(document_name)
    if not chunks or not llm_service.is_configured():
        return fallback

    sample_text = "\n\n".join(c["text"] for c in chunks[:4])
    prompt = (
        "Based on the following document excerpt, suggest exactly 5 short, useful "
        "questions a student might ask about it. Return ONLY the 5 questions, "
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
