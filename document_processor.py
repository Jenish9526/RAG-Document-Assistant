"""Document processing pipeline for text extraction and sliding-window chunking."""

from io import BytesIO
from typing import List, Dict
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import clean_text


def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    """Extract page-indexed text blocks from PDF binary data."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if raw_text.strip():
            pages.append({"page": page_number, "text": raw_text})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> List[Dict]:
    """Extract text from UTF-8 plain text binary data."""
    text = file_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"page": 1, "text": text}]


def extract_text_from_docx(file_bytes: bytes) -> List[Dict]:
    """Extract paragraph text from Microsoft Word (.docx) binary data."""
    import docx

    document = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    if not full_text.strip():
        return []
    return [{"page": 1, "text": full_text}]


def split_into_chunks(
    pages: List[Dict],
    document_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Split extracted page text into overlapping word-level chunks with metadata."""
    chunks = []
    chunk_id = 0
    step = max(chunk_size - chunk_overlap, 1)

    for page in pages:
        cleaned = clean_text(page["text"])
        words = cleaned.split(" ")

        if not words:
            continue

        for start in range(0, len(words), step):
            window = words[start:start + chunk_size]
            if not window:
                continue
            chunk_text = " ".join(window).strip()
            if len(chunk_text) < 10:
                continue

            chunks.append({
                "document": document_name,
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk_text,
            })
            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks


def process_document(file_bytes: bytes, filename: str) -> Dict:
    """Validate format, extract text, and chunk document into indexed fragments."""
    lower_name = filename.lower()

    try:
        if lower_name.endswith(".pdf"):
            pages = extract_text_from_pdf(file_bytes)
        elif lower_name.endswith(".txt"):
            pages = extract_text_from_txt(file_bytes)
        elif lower_name.endswith(".docx"):
            pages = extract_text_from_docx(file_bytes)
        else:
            return {"success": False, "error": "Unsupported file type.", "pages": 0,
                    "characters": 0, "chunks": []}
    except Exception as exc:
        return {"success": False, "error": f"Could not read file: {exc}",
                "pages": 0, "characters": 0, "chunks": []}

    if not pages:
        return {
            "success": False,
            "error": "The uploaded document contains no readable text.",
            "pages": 0, "characters": 0, "chunks": [],
        }

    total_characters = sum(len(p["text"]) for p in pages)
    chunks = split_into_chunks(pages, document_name=filename)

    if not chunks:
        return {
            "success": False,
            "error": "The document contains text, but no usable chunks could be created.",
            "pages": len(pages), "characters": total_characters, "chunks": [],
        }

    return {
        "success": True,
        "error": None,
        "pages": len(pages),
        "characters": total_characters,
        "chunks": chunks,
    }

