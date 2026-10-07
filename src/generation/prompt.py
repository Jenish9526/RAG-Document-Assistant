from typing import List, Dict, Tuple, Optional


def build_context(retrieved: List[Tuple[Dict, float]]) -> str:
    """Format retrieved document chunks into labeled context blocks."""
    blocks = []
    for i, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[Source {i}: {chunk['document']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    context: str,
    answer_style: str = "Medium",
    exam_mode: bool = False,
    history: Optional[List[Dict]] = None,
) -> str:
    """Assemble the system instructions, context, conversation history, and user question into an LLM prompt."""
    normalized_style = str(answer_style).strip().capitalize()
    if normalized_style in ("Low", "Simple"):
        style_instruction = (
            "Effort Level: LOW. Provide a concise, direct, and straightforward answer addressing "
            "all asked questions without unnecessary fluff or filler, but ensure the answer is 100% complete "
            "with zero cutoffs or omitted points."
        )
    elif normalized_style in ("High", "Detailed", "Academic"):
        style_instruction = (
            "Effort Level: HIGH. Provide an exhaustive, deeply comprehensive, and in-depth explanation. "
            "Thoroughly analyze and unpack all concepts, background, underlying mechanics, step-by-step reasoning, "
            "examples, nuances, and implications supported by the context. Do NOT abbreviate, truncate, or summarize "
            "prematurely; deliver the entire full answer."
        )
    else:  # Medium / default
        style_instruction = (
            "Effort Level: MEDIUM. Provide a balanced, thorough, and well-structured answer with clear explanations, "
            "helpful context, bullet points where useful, and full coverage of all questions asked."
        )

    exam_instruction = ""
    if exam_mode:
        exam_instruction = (
            "\nStructure the answer using ONLY the sections that are relevant to the "
            "question, chosen from: Definition, Explanation, Steps, Example, "
            "Advantages, Disadvantages, Complexity.\n"
        )

    history_section = ""
    if history:
        turns = []
        for msg in history[-4:]:
            role_name = "User" if msg.get("role") == "user" else "Assistant"
            text = msg.get("content", "").strip()
            if not text:
                continue
            # Keep previous turns compact so context focuses on new documents
            if role_name == "Assistant" and len(text) > 400:
                text = text[:400] + "..."
            turns.append(f"{role_name}: {text}")
        if turns:
            history_section = "Recent Conversation Context:\n" + "\n".join(turns) + "\n\n"

    return f"""You are an expert document assistant. Answer the user's question using ONLY the provided document context below.

CRITICAL RULES:
1. COMPLETENESS MANDATE: You MUST provide a complete, fully finished response. Do NOT stop midway, truncate thoughts, or cut off sentences.
2. MULTI-QUESTION COVERAGE: If the user asks multiple questions or requests multiple items, you MUST address and answer EVERY SINGLE ONE thoroughly. Never skip any question.
3. EFFORT LEVEL: Strictly follow the requested effort level below:
   {style_instruction}
4. FACTUALITY: Prefer information from the provided context above your own general knowledge. Do not invent facts that are not supported by the context.
5. If the answer is not present in the context, clearly say: "I couldn't find enough information about this topic in the uploaded documents."
6. Do not claim information comes from the documents if it does not appear in the context.
{exam_instruction}
{history_section}Document Context:
---
{context}
---

Question: {query}

Answer:"""
