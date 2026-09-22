"""Document loader for reading files from local filesystem or binary streams."""

import os
from typing import Tuple


def load_file_from_disk(file_path: str) -> Tuple[str, bytes]:
    """Read file from disk and return its filename and raw bytes."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        content = f.read()
    return filename, content
