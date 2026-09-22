"""CLI utility to clear or rebuild the FAISS vector index."""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import RAW_DATA_DIR
from src.ingestion.manager import clear_all_documents, get_vector_store
from scripts.ingest import ingest_path
from src.utils.logger import get_logger

logger = get_logger("scripts.rebuild_index")


def rebuild(reingest: bool = False, source_dir: str = RAW_DATA_DIR):
    """Clear existing index and optionally re-ingest from source directory."""
    store = get_vector_store()
    logger.info("Clearing existing vector store and registry...")
    clear_all_documents(store=store)

    if reingest:
        logger.info("Rebuilding index from: %s", source_dir)
        ingest_path(source_dir)
    else:
        logger.info("Index reset successfully.")


def main():
    parser = argparse.ArgumentParser(description="Reset or rebuild FAISS vector index")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-ingest documents from source directory after clearing",
    )
    parser.add_argument(
        "--source",
        default=RAW_DATA_DIR,
        help=f"Source directory to re-ingest from (default: {RAW_DATA_DIR})",
    )
    args = parser.parse_args()
    rebuild(reingest=args.reingest, source_dir=args.source)


if __name__ == "__main__":
    main()
