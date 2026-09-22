"""
document_processor.py
======================

Purpose
-------
This file is responsible for turning a raw uploaded file (PDF / TXT / DOCX)
into a list of clean, metadata-tagged TEXT CHUNKS ready for embedding.

Pipeline handled in this file:

    Raw file bytes
        -> extract_text_from_pdf() / extract_text_from_txt() / extract_text_from_docx()
        -> clean_text()   (from utils.py)
        -> split_into_chunks()

Imported libraries
-------------------
- pypdf        : reads PDF files page by page.
- docx (python-docx) : reads DOCX files paragraph by paragraph. (optional)
- io           : lets us read an in-memory file (Streamlit gives us bytes,
                 not a path on disk).

Used by
-------
document_manager.py calls `process_document()`, the single entry point
of this file, whenever a new file is uploaded.
"""

from io import BytesIO
from typing import List, Dict

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import clean_text


# ---------------------------------------------------------------------
# STEP 1: TEXT EXTRACTION
# ---------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    """
    Function: extract_text_from_pdf()

    Purpose:
        Extracts text from every page of a PDF, keeping track of which
        page each piece of text came from (needed later for source citations).

    Input:
        file_bytes: raw bytes of the uploaded PDF file.

    Output:
        A list of dictionaries, one per page:
            [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        Pages with no extractable text are skipped.

    Used by:
        process_document() in this same file.
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if raw_text.strip():
            pages.append({"page": page_number, "text": raw_text})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> List[Dict]:
    """
    Function: extract_text_from_txt()

    Purpose:
        Extracts text from a plain .txt file. TXT files have no concept
        of "pages", so we treat the whole file as "page 1".

    Input:
        file_bytes: raw bytes of the uploaded TXT file.

    Output:
        A one-item list: [{"page": 1, "text": "..."}]

    Used by:
        process_document() in this same file.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"page": 1, "text": text}]


def extract_text_from_docx(file_bytes: bytes) -> List[Dict]:
    """
    Function: extract_text_from_docx()

    Purpose:
        Extracts text from a .docx file. Word documents don't expose
        page boundaries through python-docx, so (like TXT) the whole
        document is treated as a single "page".

    Input:
        file_bytes: raw bytes of the uploaded DOCX file.

    Output:
        A one-item list: [{"page": 1, "text": "..."}]

    Used by:
        process_document() in this same file.
    """
    import docx  # imported lazily so the app still runs if python-docx
                 # is not installed and the user only ever uploads PDFs.

    document = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    if not full_text.strip():
        return []
    return [{"page": 1, "text": full_text}]


# ---------------------------------------------------------------------
# STEP 2: CHUNKING
# ---------------------------------------------------------------------

def split_into_chunks(
    pages: List[Dict],
    document_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Function: split_into_chunks()

    Purpose:
        Splits cleaned page text into overlapping word-based chunks so
        that each chunk is small enough to embed meaningfully, while
        overlap prevents cutting an idea awkwardly in half between chunks.

    How it works (simple explanation):
        Imagine a page as one long line of words. We take a "window" of
        `chunk_size` words, save it as one chunk, then slide the window
        forward but step back `chunk_overlap` words so the next chunk
        repeats a bit of the previous one for context continuity.

    Input:
        pages: output of an extract_text_from_*() function,
               i.e. [{"page": 1, "text": "..."}, ...]
        document_name: original filename, stored in every chunk's metadata.
        chunk_size: target number of words per chunk.
        chunk_overlap: number of words repeated between consecutive chunks.

    Output:
        A list of chunk dictionaries:
            {
                "document": "DAA_Notes.pdf",
                "page": 5,
                "chunk_id": 12,
                "text": "..."
            }

    Used by:
        process_document() in this same file.
    """
    chunks = []
    chunk_id = 0
    step = max(chunk_size - chunk_overlap, 1)  # avoid infinite loop if overlap >= size

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
                # Skip tiny leftover fragments (e.g. a stray "the end.")
                continue

            chunks.append({
                "document": document_name,
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk_text,
            })
            chunk_id += 1

            # Stop sliding once this window already reached the end of the page.
            if start + chunk_size >= len(words):
                break

    return chunks


# ---------------------------------------------------------------------
# STEP 3: SINGLE ENTRY POINT
# ---------------------------------------------------------------------

def process_document(file_bytes: bytes, filename: str) -> Dict:
    """
    Function: process_document()

    Purpose:
        The ONE function the rest of the app calls. Detects the file type
        from its extension, extracts text, and chunks it. This hides the
        format-specific details (PDF vs TXT vs DOCX) from the caller.

    Input:
        file_bytes: raw bytes of the uploaded file.
        filename:   original filename, e.g. "DAA_Notes.pdf".

    Output:
        {
            "success": True/False,
            "error": None or "error message",
            "pages": <page count>,
            "characters": <total character count>,
            "chunks": [ ...chunk dicts... ],
        }

    Used by:
        document_manager.py -> add_document()
    """
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
    except Exception as exc:  # noqa: BLE001 - we want to catch any parsing failure
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
