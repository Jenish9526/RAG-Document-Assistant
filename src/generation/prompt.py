"""Prompt templates and context aggregation for generation models."""

from typing import List, Dict, Tuple


def build_context(retrieved: List[Tuple[Dict, float]]) -> str:
    """Format retrieved document chunks into labeled context blocks."""
    blocks = []
    for i, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[Source {i}: {chunk['document']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    query: str, context: str, answer_style: str = "Simple", exam_mode: bool = False
) -> str:
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
