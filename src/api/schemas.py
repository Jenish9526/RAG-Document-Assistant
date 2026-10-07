"""Pydantic schemas and contract models for the REST API."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_configured: bool
    model: str
    base_url: str
    total_indexed_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query or question about documents")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of chunks to retrieve (defaults dynamically based on effort level)")
    answer_style: Optional[str] = Field("Medium", description="Low, Medium, or High effort level")
    exam_mode: Optional[bool] = Field(False, description="Enable structured output mode")


class SourceReference(BaseModel):
    document: str
    page: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    found_context: bool
    sources: List[SourceReference]


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    characters: int
    chunks: int
    size: str


class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentInfo]


class SummaryRequest(BaseModel):
    max_chunks_per_batch: Optional[int] = Field(6, ge=1, le=20)


class SummaryResponse(BaseModel):
    document: str
    summary: str


class SuggestedQuestionsResponse(BaseModel):
    document: str
    questions: List[str]
