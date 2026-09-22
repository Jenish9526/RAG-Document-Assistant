"""Ingestion package providing loaders, parsers, chunking, and document registry management."""

from src.ingestion.loader import load_file_from_disk
from src.ingestion.parser import extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx
from src.ingestion.chunker import split_into_chunks
from src.ingestion.manager import (
    process_document,
    validate_file,
    add_document,
    clear_all_documents,
    get_document_registry,
    list_document_names,
    get_vector_store,
)

__all__ = [
    "load_file_from_disk",
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "extract_text_from_docx",
    "split_into_chunks",
    "process_document",
    "validate_file",
    "add_document",
    "clear_all_documents",
    "get_document_registry",
    "list_document_names",
    "get_vector_store",
]
