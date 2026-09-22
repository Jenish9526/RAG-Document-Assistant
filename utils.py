"""Utility functions for hashing, text sanitization, and formatting."""

import hashlib
import re


def compute_file_hash(file_bytes: bytes) -> str:
    """Generate SHA-256 hex digest for binary content to detect duplicates."""
    return hashlib.sha256(file_bytes).hexdigest()


def clean_text(text: str) -> str:
    """Normalize whitespace and consecutive newlines in extracted document text."""
    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_file_size(num_bytes: int) -> str:
    """Convert a byte count into a formatted human-readable string."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def truncate_text(text: str, max_chars: int = 220) -> str:
    """Truncate text at a word boundary to fit within max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

