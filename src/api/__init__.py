"""API package providing FastAPI application and schemas."""

from src.api.routes import app
from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    DocumentInfo,
    DocumentListResponse,
    SummaryRequest,
    SummaryResponse,
    SuggestedQuestionsResponse,
)

__all__ = [
    "app",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "SummaryRequest",
    "SummaryResponse",
    "SuggestedQuestionsResponse",
]
