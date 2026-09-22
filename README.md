# RAG Document Assistant

An intelligent, multi-format document question-answering and summarization system built with **Retrieval-Augmented Generation (RAG)**, **SentenceTransformers**, **FAISS Vector Search**, **FastAPI**, and **Streamlit**.

---

## 📌 Project Overview

When dealing with large text documents, lecture notes, or research papers, finding specific facts and generating grounded summaries is time-consuming. Generic LLMs often suffer from knowledge cutoffs and hallucinations when queried on private or domain-specific materials.

**RAG Document Assistant** solves this by implementing an end-to-end Retrieval-Augmented Generation pipeline:
1. **Ingestion & Parsing**: Extracts clean text from PDF, DOCX, and TXT documents.
2. **Chunking**: Splits document pages into overlapping semantic windows with metadata.
3. **Dense Vector Embeddings**: Converts chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
4. **Exact Similarity Search**: Indexes vectors in FAISS (`IndexFlatIP`) for nearest-neighbor cosine similarity retrieval.
5. **Grounded Synthesis**: Builds strict anti-hallucination prompts injecting retrieved excerpts into modern LLMs (Google Gemini, Groq, OpenAI, or local Ollama).
6. **User & API Interfaces**: Provides both an interactive **Streamlit Web UI** and a programmatic **FastAPI REST API**.

---

## 🏗️ Architecture & Data Flow

```text
 ┌──────────────────────────────────────────────────────────────┐
 │                     User & Client Interfaces                 │
 │     ┌────────────────────────────┐  ┌──────────────────────┐ │
 │     │   Streamlit Web Frontend   │  │   FastAPI REST API   │ │
 │     │          (app.py)          │  │  (src.api.routes)    │ │
 │     └─────────────┬──────────────┘  └──────────┬───────────┘ │
 └───────────────────┼────────────────────────────┼─────────────┘
                     │                            │
                     ▼                            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 1. INGESTION LAYER (src.ingestion)                           │
 │    • loader.py   : Reads raw document bytes from disk/stream │
 │    • parser.py   : Extracts page-tagged text (PDF/DOCX/TXT)  │
 │    • chunker.py  : Sliding-window chunker (350w / 60w overlap)│
 │    • manager.py  : SHA-256 deduplication & registry state    │
 └───────────────────┬──────────────────────────────────────────┘
                     │ Chunks
                     ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 2. RETRIEVAL LAYER (src.retrieval)                           │
 │    • embeddings.py   : all-MiniLM-L6-v2 vectorizer (384-dim) │
 │    • vector_store.py : FAISS IndexFlatIP cosine index        │
 │    • retriever.py    : Semantic filter (threshold >= 0.25)   │
 └───────────────────┬──────────────────────────────────────────┘
                     │ Top-K Relevant Context Chunks
                     ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ 3. GENERATION LAYER (src.generation)                         │
 │    • prompt.py   : Context formatting & strict prompt rules  │
 │    • llm.py      : OpenAI-compatible client + retry backoff  │
 │    • response.py : Q&A, Map-Reduce summary & suggestions     │
 └───────────────────┬──────────────────────────────────────────┘
                     │ Verified Answer + Source Citations
                     ▼
```

---

## 🧩 Complete Module & Function Reference

### 1. Ingestion Package (`src.ingestion`)

| File | Function | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **`loader.py`** | `load_file_from_disk` | `file_path: str` | Reads a file from local storage, returning `(filename, bytes)`. |
| **`parser.py`** | `extract_text_from_pdf` | `file_bytes: bytes` | Extracts text page-by-page from PDFs using `pypdf`. |
| | `extract_text_from_txt` | `file_bytes: bytes` | Decodes plain UTF-8 text files. |
| | `extract_text_from_docx`| `file_bytes: bytes` | Extracts paragraphs from Microsoft Word documents. |
| **`chunker.py`**| `split_into_chunks` | `pages, document_name, chunk_size=350, chunk_overlap=60` | Splits text into overlapping word windows with page and chunk IDs. |
| **`manager.py`**| `validate_file` | `filename: str, file_bytes: bytes` | Checks allowed formats (`.pdf`, `.txt`, `.docx`) and size limits (25MB). |
| | `process_document` | `file_bytes: bytes, filename: str` | Dispatches parser and chunker, returning pages, char count, and chunks. |
| | `add_document` | `filename: str, file_bytes: bytes, store=None` | Computes SHA-256 hash, checks duplicates, embeds chunks, and saves index. |
| | `clear_all_documents`| `store=None` | Wipes the vector index and document registry from memory and disk. |
| | `list_document_names`| None | Returns filenames of all currently indexed documents. |
| | `get_document_registry`| None | Retrieves document metadata dictionary (pages, characters, chunks, size). |
| | `get_vector_store` | `index_path, metadata_path` | Returns the singleton `VectorStore` instance. |

---

### 2. Retrieval Package (`src.retrieval`)

| File | Function | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **`embeddings.py`** | `load_embedding_model` | None | Loads and caches the `all-MiniLM-L6-v2` SentenceTransformer. |
| | `generate_embeddings` | `chunks: List[dict]` | Converts document chunks into normalized float32 matrix `(N, 384)`. |
| | `generate_query_embedding` | `query: str` | Converts user question into a normalized float32 vector `(1, 384)`. |
| **`vector_store.py`** | `VectorStore.add_documents` | `chunks: List[Dict], embeddings: np.ndarray` | Adds embeddings and metadata to FAISS `IndexFlatIP`. |
| | `VectorStore.search` | `query_embedding: np.ndarray, top_k: int` | Searches nearest neighbors, returning `[(chunk_metadata, score)]`. |
| | `VectorStore.save_index` | None | Persists FAISS binary (`index.faiss`) and metadata (`metadata.pkl`). |
| | `VectorStore.load_index` | None | Loads persisted FAISS index and metadata from disk. |
| | `VectorStore.clear_index`| None | Clears memory vectors and removes persisted files on disk. |
| | `VectorStore.chunks_for_document` | `document_name: str` | Returns all stored chunks belonging to a specific document. |
| **`retriever.py`** | `retrieve_relevant_chunks` | `query, store, top_k=5, threshold=0.25` | Embeds query, executes search, and filters out chunks below threshold. |

---

### 3. Generation Package (`src.generation`)

| File | Function | Parameters | Description |
| :--- | :--- | :--- | :--- |
| **`llm.py`** | `generate_response` | `prompt: str` | Sends prompt to LLM Chat Completions endpoint with exponential retry. |
| | `get_llm_config` | None | Dynamically reloads `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. |
| | `is_configured` | None | Returns `True` if `LLM_API_KEY` is present. |
| **`prompt.py`** | `build_context` | `retrieved: List[Tuple[Dict, float]]` | Formats chunks into labelled blocks with document and page numbers. |
| | `build_prompt` | `query, context, answer_style, exam_mode` | Assembles strict system anti-hallucination rules, context, and question. |
| **`response.py`**| `answer_question` | `query, store, top_k=5, answer_style="Simple", exam_mode=False` | End-to-end RAG pipeline: retrieval, context assembly, LLM call, citation tags. |
| | `summarize_document` | `document_name, store, max_chunks_per_batch=6` | Map-Reduce summarization: sub-summarizes chunk batches, then combines. |
| | `generate_suggested_questions` | `document_name, store` | Generates 5 exploration questions from document sample with fallbacks. |

---

### 4. REST API Endpoints (`src.api`)

| Method | Endpoint | Request Body / Params | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | Service status, active model, and total indexed chunks. |
| `POST` | `/query` | `{"query": str, "top_k": int, "answer_style": str, "exam_mode": bool}` | Executes RAG question answering with citations. |
| `GET` | `/documents` | None | Lists all indexed documents with page and chunk stats. |
| `POST` | `/documents/upload` | Multipart form (`file`) | Uploads and indexes a PDF, TXT, or DOCX document. |
| `POST` | `/documents/{name}/summarize` | `{"max_chunks_per_batch": int}` | Generates Map-Reduce structured summary. |
| `GET` | `/documents/{name}/suggested-questions` | None | Returns 5 suggested exploratory questions. |
| `DELETE`| `/documents` | None | Clears the vector store and document registry. |

---

### 5. CLI Automation Scripts (`scripts/`)

- **`scripts/ingest.py`**:
  ```powershell
  python scripts/ingest.py --path data/raw
  ```
  Scans directory for supported files, extracts text, chunks, embeds, and indexes them in batch.

- **`scripts/rebuild_index.py`**:
  ```powershell
  # Reset vector index to empty
  python scripts/rebuild_index.py

  # Reset and re-ingest all files from data/raw/
  python scripts/rebuild_index.py --reingest
  ```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Project
```bash
cd RAG_Document_Assistant
```

### 2. Create Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 LLM Configuration (`.env`)

Create a `.env` file in the root folder:

```env
# Default: Google Gemini (Free tier with 1,000,000 Tokens/Minute)
# Get your free key at: https://aistudio.google.com/app/apikey
LLM_API_KEY=your_gemini_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite

# Optional tunables
TOP_K=5
SIMILARITY_THRESHOLD=0.25
CHUNK_SIZE=350
CHUNK_OVERLAP=60
```

> **Provider Flexibility**: You can switch to **Groq Cloud** (`LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_MODEL=openai/gpt-oss-20b`), **OpenAI** (`gpt-4o-mini`), or **Ollama** (`http://localhost:11434/v1`) without changing code.

---

## 🚀 Running the Project

### Option A: Interactive Web Interface (Streamlit)
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- **Upload**: Upload PDF/DOCX/TXT files via the left sidebar.
- **Chat**: Ask natural language questions in the chat input.
- **Inspect**: Expand citation accordions showing exact document name, page, and similarity scores.
- **Summarize**: Click "Summarize Document" for a Map-Reduce overview.

### Option B: Programmatic REST API (FastAPI)
```powershell
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

## 🧪 Automated Testing

The project includes modular unit and integration tests across all components:

```powershell
# Run the complete test suite
python -m unittest discover tests

# Or run individual domain test files:
python -m unittest tests/test_ingestion.py   # Loader, parser, chunker, registry
python -m unittest tests/test_retrieval.py   # Embeddings, FAISS search, persistence
python -m unittest tests/test_generation.py  # Prompting, Q&A synthesis, summarizer
python -m unittest tests/test_api.py         # FastAPI REST endpoints
```

---

## 📁 Clean Repository Structure

```text
RAG_Document_Assistant/
├── docs/
│   ├── architecture.md          # Visual pipeline & architecture guide
│   ├── api.md                   # Complete REST API reference
│   └── setup.md                 # Setup guide for local and Docker
├── src/
│   ├── ingestion/               # Document loaders, parsers, chunking & manager
│   │   ├── loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── manager.py
│   ├── retrieval/               # Embeddings, FAISS vector store & semantic search
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── generation/              # LLM client adapter, prompt compiler & synthesizer
│   │   ├── llm.py
│   │   ├── prompt.py
│   │   └── response.py
│   ├── api/                     # FastAPI application & Pydantic schemas
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── config/                  # Central configuration & paths
│   │   └── settings.py
│   ├── ui/                      # Streamlit session chat state manager
│   │   └── chat_manager.py
│   └── utils/                   # Structured logging & text utilities
│       ├── logger.py
│       └── helpers.py
├── data/
│   ├── raw/                     # Raw input documents
│   ├── processed/               # Processed text artifacts
│   ├── samples/                 # Demonstration documents
│   └── vectorstore/             # FAISS binary index & metadata
├── scripts/
│   ├── ingest.py                # Batch CLI document ingestion script
│   └── rebuild_index.py         # Index reset & rebuild CLI utility
├── tests/
│   ├── test_ingestion.py        # Ingestion & parser tests
│   ├── test_retrieval.py        # Embeddings & FAISS tests
│   ├── test_generation.py       # Prompt & synthesis tests
│   └── test_api.py              # FastAPI endpoint tests
├── app.py                       # Streamlit UI web workspace entrypoint
├── Dockerfile                   # Container deployment definition
├── pyproject.toml               # Packaging configuration
├── requirements.txt             # Project dependencies
├── .env                         # Active configuration & API key
├── .env.example                 # Environment template
└── .gitignore                   # Local secrets & data exclusions
```
