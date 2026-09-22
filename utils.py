"""
utils.py
========

Purpose
-------
A small toolbox of generic helper functions that don't belong to any
one specific module (hashing, text cleanup, formatting helpers).

Keeping these here avoids duplicating the same small pieces of code
in document_processor.py, document_manager.py, etc.

Used by
-------
document_processor.py, document_manager.py, app.py
"""

import hashlib
import re


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Purpose:
        Create a unique fingerprint (SHA-256 hash) for a file's content.
        Two files with identical content will always produce the same hash,
        even if their filenames are different.

    Input:
        file_bytes: the raw bytes of the uploaded file.

    Output:
        A 64-character hexadecimal string, e.g. "3f9a1c...".

    Used by:
        document_manager.py -> to detect duplicate uploads.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def clean_text(text: str) -> str:
    """
    Purpose:
        Normalize whitespace in extracted document text WITHOUT
        destroying the actual content (numbers, symbols, formulas).

    What it does:
        - Collapses multiple spaces/tabs into a single space.
        - Collapses 3+ blank lines into a single blank line.
        - Strips leading/trailing whitespace from each line.

    Input:
        text: raw extracted text (may contain messy PDF spacing).

    Output:
        Cleaned text string.

    Used by:
        document_processor.py -> right after text extraction, before chunking.
    """
    if not text:
        return ""

    # Replace tabs and multiple spaces with a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing spaces on each line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines into just 2 (i.e. one blank line).
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_file_size(num_bytes: int) -> str:
    """
    Purpose:
        Convert a raw byte count into a human-readable string.

    Input:
        num_bytes: file size in bytes.

    Output:
        A readable string like "482.3 KB" or "1.2 MB".

    Used by:
        app.py -> document statistics panel.
    """
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def truncate_text(text: str, max_chars: int = 220) -> str:
    """
    Purpose:
        Shorten long text for preview display (e.g. showing a chunk
        preview in the source expander) without cutting a word in half.

    Input:
        text: the full text.
        max_chars: maximum number of characters to keep.

    Output:
        Truncated text, ending with "..." if it was cut short.

    Used by:
        app.py -> displaying retrieved source chunks.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
