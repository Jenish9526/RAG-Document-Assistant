"""FastAPI REST API routes and application definition."""

from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.config.settings import APP_TITLE, APP_SUBTITLE
from src.generation.llm import get_llm_config, is_configured
from src.ingestion.manager import (
    get_vector_store,
    get_document_registry,
    add_document,
    clear_all_documents,
)
from src.generation.response import (
    answer_question,
    summarize_document,
    generate_suggested_questions,
)
from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    DocumentInfo,
    DocumentListResponse,
    SummaryRequest,
    SummaryResponse,
    SuggestedQuestionsResponse,
)

app = FastAPI(
    title=APP_TITLE,
    description=APP_SUBTITLE,
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health status and current LLM configuration check."""
    _, base_url, model = get_llm_config()
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        llm_configured=is_configured(),
        model=model,
        base_url=base_url,
        total_indexed_chunks=store.total_chunks,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    """Query indexed documents and synthesize an answer using RAG."""
    store = get_vector_store()
    result = answer_question(
        query=req.query,
        store=store,
        top_k=req.top_k,
        answer_style=req.answer_style,
        exam_mode=req.exam_mode,
    )
    sources = [
        SourceReference(document=s["document"], page=s["page"], score=s["score"])
        for s in result.get("sources", [])
    ]
    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        found_context=result.get("found_context", False),
        sources=sources,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all indexed documents with page and chunk metadata."""
    registry = get_document_registry()
    docs = [
        DocumentInfo(
            filename=info["filename"],
            pages=info["pages"],
            characters=info["characters"],
            chunks=info["chunks"],
            size=info["size"],
        )
        for info in registry.values()
    ]
    return DocumentListResponse(count=len(docs), documents=docs)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, TXT, DOCX)."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = add_document(file.filename, file_bytes)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return {"message": result["message"], "filename": file.filename}


@app.post("/documents/{doc_name}/summarize", response_model=SummaryResponse)
def summarize_doc(doc_name: str, req: SummaryRequest = SummaryRequest()):
    """Generate structured summary for an indexed document."""
    store = get_vector_store()
    summary = summarize_document(
        doc_name, store=store, max_chunks_per_batch=req.max_chunks_per_batch
    )
    return SummaryResponse(document=doc_name, summary=summary)


@app.get("/documents/{doc_name}/suggested-questions", response_model=SuggestedQuestionsResponse)
def get_suggested_questions(doc_name: str):
    """Retrieve suggested exploratory questions for an indexed document."""
    store = get_vector_store()
    questions = generate_suggested_questions(doc_name, store=store)
    return SuggestedQuestionsResponse(document=doc_name, questions=questions)


@app.delete("/documents")
def clear_documents():
    """Clear all documents and reset the vector store."""
    clear_all_documents()
    return {"message": "All documents cleared successfully."}
