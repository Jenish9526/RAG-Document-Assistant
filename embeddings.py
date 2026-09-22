"""
embeddings.py
=============

Purpose
-------
Converts text (document chunks OR a user's question) into numeric
vectors ("embeddings") using a Sentence Transformer model.

Simple explanation:
    An embedding converts text into a list of numbers that represents
    its MEANING. Texts with similar meaning end up as numbers that are
    close together in space, even if they don't share the exact same words.

Technical explanation:
    We use the pretrained "all-MiniLM-L6-v2" Sentence-Transformer model,
    which maps any input sentence/paragraph into a dense 384-dimensional
    vector. Semantic similarity between two texts can then be measured
    with cosine similarity between their vectors.

Imported libraries
-------------------
- sentence_transformers.SentenceTransformer : loads & runs the embedding model.
- streamlit (for @st.cache_resource)         : keeps the model loaded in memory
  across Streamlit re-runs so it is NOT reloaded on every user interaction.
- numpy                                      : embeddings are returned as
  numpy arrays, which FAISS expects.

Used by
-------
rag_engine.py (question embedding) and document_manager.py
(document chunk embeddings, via vector_store.py's add_documents()).
"""

from typing import List
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """
    Function: load_embedding_model()

    Purpose:
        Loads the Sentence Transformer model ONCE and keeps it cached.

    Why caching matters:
        Streamlit re-runs the whole script on every user interaction
        (every button click, every chat message). Without caching, the
        ~90MB model would be reloaded from disk every single time,
        making the app painfully slow. `@st.cache_resource` tells
        Streamlit: "build this object once, then reuse it forever
        (until the app restarts)."

    Input:
        None.

    Output:
        A ready-to-use SentenceTransformer model instance.

    Used by:
        generate_embeddings() and generate_query_embedding() below.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(chunks: List[dict]) -> np.ndarray:
    """
    Function: generate_embeddings()

    Purpose:
        Converts a list of document chunks into a matrix of embedding
        vectors (one row per chunk) so they can be added to FAISS.

    Input:
        chunks: list of chunk dicts, each containing a "text" field
                (as produced by document_processor.split_into_chunks()).

    Output:
        A numpy array of shape (num_chunks, 384), dtype float32.
        float32 is required by FAISS.

    Used by:
        document_manager.py -> add_document(), right before calling
        vector_store.add_documents().
    """
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # pre-normalize so we can use inner product
    )
    return vectors.astype("float32")


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Function: generate_query_embedding()

    Purpose:
        Converts a single user question into one embedding vector,
        using the exact same model as generate_embeddings() so the
        vectors live in the same "meaning space" and can be compared.

    Input:
        query: the user's natural-language question, e.g.
               "What is Dynamic Programming?"

    Output:
        A numpy array of shape (1, 384), dtype float32, ready to be
        passed directly into FAISS's `.search()`.

    Used by:
        rag_engine.py -> retrieve_relevant_chunks()
    """
    model = load_embedding_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector.astype("float32")
