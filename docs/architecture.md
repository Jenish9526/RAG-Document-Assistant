# System Architecture

The **RAG Document Assistant** is designed using an enterprise modular pipeline that strictly separates ingestion, embedding & retrieval, response synthesis, and user-facing presentation layers.

```
                  ┌─────────────────────────────────────┐
                  │    User Interfaces & API Clients    │
                  ├──────────────────┬──────────────────┤
                  │ Streamlit UI     │ FastAPI REST API │
                  │ (app.py)         │ (src.api.routes) │
                  └─────────┬────────┴─────────┬────────┘
                            │                  │
 ┌──────────────────────────┼──────────────────┼─────────────────────────┐
 │ INGESTION (src.ingestion)│                  │ RETRIEVAL (src.retrieval│
 │                          │                  │                         │
 │ ┌──────────────┐         │                  │ ┌─────────────────────┐ │
 │ │ Document     │         ▼                  ▼ │ SentenceTransformer │ │
 │ │ Loaders &    │   ┌──────────────────────────┤ (all-MiniLM-L6-v2)  │ │
 │ │ Parsers      │   │    Manager & Registry    │ └──────────┬──────────┘ │
 │ └──────┬───────┘   │ (src.ingestion.manager)  │            │          │
 │        │           └─────────────┬────────────┘            ▼          │
 │        ▼                         │             ┌────────────────────┐ │
 │ ┌──────────────┐                 ▼             │ FAISS Vector Store │ │
 │ │ Word Chunker │────────▶ Add Chunks & Embeds──┤ (IndexFlatIP)      │ │
 │ └──────────────┘                               └───────────┬────────┘ │
 └────────────────────────────────────────────────────────────┼──────────┘
                                                              │
 ┌────────────────────────────────────────────────────────────┼──────────┐
 │ GENERATION (src.generation)                                │          │
 │                                                            ▼          │
 │ ┌──────────────────────┐                     ┌──────────────────────┐ │
 │ │ Context & Prompt     │◀──Top-K Chunks──────┤ Retriever            │ │
 │ │ Builder              │   (threshold >=0.25)│ (semantic filter)    │ │
 │ └──────────┬───────────┘                     └──────────────────────┘ │
 │            ▼                                                          │
 │ ┌──────────────────────┐                                              │
 │ │ LLM Client Adapter   │──(HTTP Exponential Backoff)──▶ Google Gemini │
 │ │ (OpenAI compatible)  │                                Groq / OpenAI │
 │ └──────────┬───────────┘                                              │
 │            ▼                                                          │
 │ ┌──────────────────────┐                                              │
 │ │ Response Synthesis   │──▶ Verified Answer + Grounded Source Tags    │
 │ └──────────────────────┘                                              │
 └───────────────────────────────────────────────────────────────────────┘
```

---

## 1. Document Ingestion Layer (`src/ingestion`)

1. **Loader (`loader.py`)**:
   Reads binary payloads from disk streams or HTTP upload buffers.
2. **Parser (`parser.py`)**:
   - `extract_text_from_pdf`: Page-by-page extraction preserving document page markers using `pypdf`.
   - `extract_text_from_txt`: Unicode-decoded stream parsing.
   - `extract_text_from_docx`: Paragraph-level extraction via `python-docx`.
3. **Chunker (`chunker.py`)**:
   Sliding-window word-level tokenizer using configurable parameters (`CHUNK_SIZE=350`, `CHUNK_OVERLAP=60`). Ensures boundary preservation and prevents truncated semantics between consecutive fragments.
4. **Manager (`manager.py`)**:
   - Computes SHA-256 fingerprint (`compute_file_hash`) to reject duplicate files before processing.
   - Maintains a persisted registry (`document_registry.pkl`) synchronized with the vector index.

---

## 2. Retrieval Layer (`src/retrieval`)

1. **Embeddings (`embeddings.py`)**:
   Uses `sentence-transformers/all-MiniLM-L6-v2` producing normalized 384-dimensional dense vectors. Normalization maps inner products directly to cosine similarities in `[-1.0, 1.0]`.
2. **Vector Store (`vector_store.py`)**:
   Thin wrapper over `faiss.IndexFlatIP`. Guarantees exact nearest-neighbor search with zero vector quantization distortion. Stores chunk text and page metadata in a parallel serialization list.
3. **Retriever (`retriever.py`)**:
   Embeds input queries and executes top-$k$ nearest neighbor retrieval. Applies a strict similarity threshold (`SIMILARITY_THRESHOLD = 0.25`) to discard irrelevant noise and mitigate model hallucinations.

---

## 3. Generation Layer (`src/generation`)

1. **Prompt Engineering (`prompt.py`)**:
   Formats retrieved context chunks with source numbers and page tags. Injects strict system constraints:
   - Ground truth prioritization over internal weights.
   - Explicit refusal if retrieved context is insufficient.
   - Style adaptations: *Simple* (bulleted, plain) vs. *Detailed* (technical) vs. *Exam Mode* (structured schema).
2. **LLM Adapter (`llm.py`)**:
   Vendor-agnostic REST client compatible with standard OpenAI Chat Completion endpoints. Implements exponential backoff with dynamic rate-limit header inspection for HTTP 429 resiliency.
3. **Synthesis (`response.py`)**:
   - Question Answering: Single-pass retrieval + grounded response generation.
   - Map-Reduce Summarization: Batches document chunks into sub-summaries ("map") and synthesizes a final structured topic summary ("reduce").
   - Suggested Questions: Generates exploratory reader questions based on document excerpts.

---

## 4. API & UI Presentation (`src/api`, `app.py`)

- **FastAPI Backend (`src.api.routes`)**: Exposes REST endpoints for headless server integrations, automated pipelines, and external client frontends.
- **Streamlit Frontend (`app.py`)**: Interactive web workspace with real-time indexing status, chat history, expandable citation blocks, and document management.
