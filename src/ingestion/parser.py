"""Parsers for PDF, plain text, and DOCX document formats."""

from io import BytesIO
from typing import List, Dict
from pypdf import PdfReader


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
