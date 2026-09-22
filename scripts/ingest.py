"""CLI script to ingest documents from a directory or path into the vector store."""

import os
import sys
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import RAW_DATA_DIR, ALLOWED_EXTENSIONS
from src.ingestion.manager import add_document, get_vector_store
from src.utils.logger import get_logger

logger = get_logger("scripts.ingest")


def ingest_path(target_path: str):
    """Ingest a single file or an entire directory of supported files."""
    if not os.path.exists(target_path):
        logger.error("Path does not exist: %s", target_path)
        return

    files_to_process = []
    if os.path.isfile(target_path):
        files_to_process.append(target_path)
    else:
        for root, _, files in os.walk(target_path):
            for file in files:
                ext = os.path.splitext(file.lower())[1]
                if ext in ALLOWED_EXTENSIONS:
                    files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        logger.warning("No supported documents (%s) found in %s", ALLOWED_EXTENSIONS, target_path)
        return

    logger.info("Found %d document(s) to process", len(files_to_process))
    store = get_vector_store()

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            result = add_document(filename, content, store=store)
            if result["success"]:
                logger.info("SUCCESS: %s", result["message"])
            elif result.get("duplicate"):
                logger.warning("SKIPPED: %s", result["message"])
            else:
                logger.error("FAILED: %s - %s", filename, result["message"])
        except Exception as exc:
            logger.error("ERROR reading %s: %s", filename, exc)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    parser.add_argument(
        "--path",
        default=RAW_DATA_DIR,
        help=f"File or directory path to ingest (default: {RAW_DATA_DIR})",
    )
    args = parser.parse_args()
    ingest_path(args.path)


if __name__ == "__main__":
    main()
