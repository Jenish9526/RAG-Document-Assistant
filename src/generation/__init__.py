"""Generation package providing LLM adapters, prompt construction, and response synthesis."""

from src.generation.llm import (
    generate_response,
    get_llm_config,
    is_configured,
    LLMNotConfiguredError,
)
from src.generation.prompt import build_context, build_prompt
from src.generation.response import (
    generate_answer,
    answer_question,
    summarize_document,
    generate_suggested_questions,
    get_effort_parameters,
)

__all__ = [
    "generate_response",
    "get_llm_config",
    "is_configured",
    "LLMNotConfiguredError",
    "build_context",
    "build_prompt",
    "generate_answer",
    "answer_question",
    "summarize_document",
    "generate_suggested_questions",
    "get_effort_parameters",
]
