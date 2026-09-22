# RAG Document Assistant

An intelligent, multi-format document question-answering and summarization system built with **Retrieval-Augmented Generation (RAG)**, **SentenceTransformers**, **FAISS Vector Search**, **FastAPI**, and **Streamlit**.

---

## 📌 Project Overview

When querying large documents, textbooks, lecture slides, or research papers, traditional keyword search often misses semantic meaning, while general-purpose Large Language Models (LLMs) can hallucinate or fail to reference specific sections.

**RAG Document Assistant** solves this by implementing an end-to-end, privacy-friendly Retrieval-Augmented Generation pipeline:
1. **Document Ingestion & Parsing**: Ingests PDF, DOCX, and TXT files, extracting page-indexed text content.
2. **Semantic Chunking**: Segments documents into overlapping sliding windows (350 words, 60-word overlap) preserving context boundaries.
3. **Dense Vector Embeddings**: Encodes chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
4. **Fast Similarity Retrieval**: Indexes embeddings with **FAISS** (`IndexFlatIP`) for nearest-neighbor cosine similarity matching.
5. **Grounded Synthesis & Verification**: Injects top-K relevant excerpts into strict, anti-hallucination prompt templates with direct source citations (document name, page number, similarity score).
6. **Dual Presentation**: Delivers an interactive **Streamlit Web UI** for end users and a **FastAPI REST API** for programmatic integration.

---

## 🏗️ System Architecture

```text
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         Client & User Interfaces                         │
 │     ┌───────────────────────────────┐  ┌───────────────────────────────┐ │
 │     │    Streamlit Web Frontend     │  │       FastAPI REST API        │ │
 │     │           (app.py)            │  │      (src.api.routes)         │ │
 │     └───────────────┬───────────────┘  └───────────────┬───────────────┘ │
 └─────────────────────┼──────────────────────────────────┼─────────────────┘
                       │                                  │
                       ▼                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 1. INGESTION LAYER (src.ingestion)                                       │
 │    • loader.py   : Reads file bytes from disk or file streams            │
 │    • parser.py   : Extracts text & page numbers from PDF, DOCX, and TXT  │
 │    • chunker.py  : Splits text into overlapping sliding windows          │
 │    • manager.py  : SHA-256 deduplication, validation, & pipeline orchestrator│
 └─────────────────────┬────────────────────────────────────────────────────┘
                       │ Semantic Chunks with Metadata
                       ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 2. RETRIEVAL LAYER (src.retrieval)                                       │
 │    • embeddings.py   : Generates L2-normalized vectors (all-MiniLM-L6-v2) │
 │    • vector_store.py : FAISS IndexFlatIP (exact cosine similarity)       │
 │    • retriever.py    : Query embedding & relevance score filtering (≥0.25)│
 └─────────────────────┬────────────────────────────────────────────────────┘
                       │ Top-K Relevant Document Excerpts
                       ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 3. GENERATION LAYER (src.generation)                                     │
 │    • prompt.py   : Context construction & anti-hallucination instructions│
 │    • llm.py      : OpenAI-compatible LLM client with exponential retry   │
 │    • response.py : Q&A synthesis, Map-Reduce summarization, suggestions  │
 └─────────────────────┬────────────────────────────────────────────────────┘
                       │ Grounded Answer + Source Citations
                       ▼
```

---

## 🧩 Complete Module & Function Reference

### 1. Ingestion Package (`src.ingestion`)

Handles file loading, format extraction, text chunking, deduplication, and persistence management.

| File | Function / Class | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`loader.py`** | `load_file_from_disk` | `file_path: str` | `Tuple[str, bytes]` | Reads a file from local storage, validates existence, and returns the filename and raw bytes. |
| **`parser.py`** | `extract_text_from_pdf` | `file_bytes: bytes` | `List[Tuple[int, str]]` | Parses PDF files using `pypdf`, extracting page-numbered text blocks while stripping non-printable characters. |
| | `extract_text_from_txt` | `file_bytes: bytes` | `List[Tuple[int, str]]` | Decodes raw text bytes using UTF-8 (with fallback decoding) into a single page entry. |
| | `extract_text_from_docx`| `file_bytes: bytes` | `List[Tuple[int, str]]` | Parses Microsoft Word (`.docx`) documents paragraph by paragraph using `python-docx`. |
| **`chunker.py`** | `split_into_chunks` | `pages: List[Tuple[int, str]]`, `document_name: str`, `chunk_size: int = 350`, `chunk_overlap: int = 60` | `List[Dict]` | Breaks multi-page text into sliding-window word chunks, tagging each chunk with `document_name`, `page_number`, `chunk_id`, and character length. |
| **`manager.py`** | `validate_file` | `filename: str`, `file_bytes: bytes` | `Tuple[bool, str]` | Validates file extension against allowed types (`.pdf`, `.txt`, `.docx`) and verifies file size is within limits (default 25MB). |
| | `process_document` | `file_bytes: bytes`, `filename: str` | `Tuple[List[Tuple[int, str]], int, List[Dict]]` | High-level ingestion coordinator: extracts text pages, calculates character count, and produces chunk objects. |
| | `add_document` | `filename: str`, `file_bytes: bytes`, `store: VectorStore = None` | `Tuple[bool, str, int]` | Computes SHA-256 hash to prevent duplicate ingestion, generates embeddings, adds vectors to FAISS, updates the registry, and saves to disk. |
| | `clear_all_documents` | `store: VectorStore = None` | `bool` | Completely resets the vector index, document registry, and removes persistent index files from disk. |
| | `list_document_names` | None | `List[str]` | Returns a list of filenames for all currently indexed documents. |
| | `get_document_registry` | None | `Dict[str, Dict]` | Returns the document metadata registry containing page counts, character counts, chunk totals, and file sizes. |
| | `get_vector_store` | `index_path: str = None`, `metadata_path: str = None` | `VectorStore` | Returns the singleton instance of `VectorStore`. |

---

### 2. Retrieval Package (`src.retrieval`)

Generates semantic embeddings, manages the FAISS vector index, and performs nearest-neighbor search.

| File | Function / Class | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`embeddings.py`** | `load_embedding_model` | None | `SentenceTransformer` | Loads and caches the `all-MiniLM-L6-v2` transformer model (384-dimensional dense vectors). |
| | `generate_embeddings` | `chunks: List[Dict]` | `np.ndarray` | Encodes chunk text into a normalized `(N, 384)` float32 numpy array. |
| | `generate_query_embedding` | `query: str` | `np.ndarray` | Encodes a user query string into a normalized `(1, 384)` float32 vector for similarity search. |
| **`vector_store.py`** | `VectorStore.__init__` | `dimension: int = 384`, `index_path: str = None`, `metadata_path: str = None` | `None` | Initializes a FAISS `IndexFlatIP` index and loads existing index files if present. |
| | `VectorStore.add_documents` | `chunks: List[Dict]`, `embeddings: np.ndarray` | `None` | Appends vectors and corresponding metadata dicts to the FAISS index. |
| | `VectorStore.search` | `query_embedding: np.ndarray`, `top_k: int = 5` | `List[Tuple[Dict, float]]` | Executes nearest-neighbor search and returns a list of `(chunk_metadata, cosine_score)` tuples. |
| | `VectorStore.save_index` | None | `None` | Persists the FAISS binary index (`index.faiss`) and chunk metadata (`metadata.pkl`) to disk. |
| | `VectorStore.load_index` | None | `bool` | Loads the saved FAISS index and metadata from disk if available. |
| | `VectorStore.clear_index` | None | `None` | Clears all in-memory vectors and deletes the persisted index files. |
| | `VectorStore.chunks_for_document` | `document_name: str` | `List[Dict]` | Retrieves all indexed chunks belonging to a specific document name. |
| **`retriever.py`** | `retrieve_relevant_chunks` | `query: str`, `store: VectorStore`, `top_k: int = 5`, `threshold: float = 0.25` | `List[Tuple[Dict, float]]` | Converts the query to an embedding, queries FAISS, and filters out results below the similarity threshold. |

---

### 3. Generation Package (`src.generation`)

Handles prompt engineering, LLM API communication with exponential backoff, Q&A synthesis, and Map-Reduce summarization.

| File | Function / Class | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`llm.py`** | `get_llm_config` | None | `Tuple[str, str, str]` | Dynamically reads `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` from environment settings. |
| | `is_configured` | None | `bool` | Checks whether an LLM API key is present and configured. |
| | `generate_response` | `prompt: str` | `str` | Submits a prompt to an OpenAI-compatible Chat Completions API with 3 automatic retries and exponential backoff. |
| **`prompt.py`** | `build_context` | `retrieved: List[Tuple[Dict, float]]` | `str` | Formats retrieved chunks into numbered context blocks containing document title and page number. |
| | `build_prompt` | `query: str`, `context: str`, `answer_style: str = "Simple"`, `exam_mode: bool = False` | `str` | Compiles system instructions, style constraints ("Simple", "Detailed", "Academic"), optional exam revision formatting, context, and user question. |
| **`response.py`** | `answer_question` | `query: str`, `store: VectorStore`, `top_k: int = 5`, `answer_style: str = "Simple"`, `exam_mode: bool = False` | `Tuple[str, List[Tuple[Dict, float]]]` | Complete RAG pipeline: retrieves relevant chunks, constructs prompt, invokes LLM, and attaches citations. |
| | `summarize_document` | `document_name: str`, `store: VectorStore`, `max_chunks_per_batch: int = 6` | `str` | Implements Map-Reduce summarization: summarizes chunk batches in parallel/sequence, then combines them into an executive summary. |
| | `generate_suggested_questions` | `document_name: str`, `store: VectorStore` | `List[str]` | Analyzes document chunks and prompts the LLM to generate 5 exploratory study questions with fallback heuristics. |

---

### 4. REST API Endpoints (`src.api`)

FastAPI application providing headless programmatic access to the entire RAG pipeline.

| Method | Endpoint | Request Body / Parameters | Response Schema | Description |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | `HealthResponse` | System health check returning status, active LLM model, embedding dimension, and indexed chunk count. |
| `POST` | `/query` | `QueryRequest` (`query`, `top_k`, `answer_style`, `exam_mode`) | `QueryResponse` | Runs semantic search and LLM synthesis, returning the answer, confidence score, and citation list. |
| `GET` | `/documents` | None | `DocumentListResponse` | Returns a list of all indexed documents with page count, character count, and chunk count. |
| `POST` | `/documents/upload` | Multipart form (`file: UploadFile`) | `UploadResponse` | Uploads and processes a PDF, TXT, or DOCX document into the vector store. |
| `POST` | `/documents/{name}/summarize` | Path: `name`, Body: `SummarizeRequest` (`max_chunks_per_batch`) | `SummarizeResponse` | Generates a structured Map-Reduce summary of the specified document. |
| `GET` | `/documents/{name}/suggested-questions` | Path: `name` | `SuggestedQuestionsResponse` | Returns 5 suggested exploratory questions derived from the document. |
| `DELETE` | `/documents` | None | `Dict[str, Any]` | Resets the vector store and clears all indexed document records. |

---

### 5. CLI Automation Scripts (`scripts/`)

Command-line utilities for batch ingestion and database administration without opening a web browser.

- **`scripts/ingest.py`**:
  ```powershell
  # Ingest all documents from data/raw/
  python scripts/ingest.py

  # Ingest documents from a custom directory or file
  python scripts/ingest.py --path "path/to/my_notes.pdf"
  ```
- **`scripts/rebuild_index.py`**:
  ```powershell
  # Reset vector store to empty state
  python scripts/rebuild_index.py

  # Reset vector store and re-ingest all documents in data/raw/
  python scripts/rebuild_index.py --reingest
  ```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 2. Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
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

## 🔑 LLM Provider Configuration (`.env`)

Create a `.env` file in the project root directory:

```env
# =====================================================================
# Recommended: Google Gemini (Free Tier: 1,000,000 Tokens/Minute)
# Get a free key at: https://aistudio.google.com/app/apikey
# =====================================================================
LLM_API_KEY=your_gemini_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite

# =====================================================================
# Optional Pipeline Parameters
# =====================================================================
TOP_K=5
SIMILARITY_THRESHOLD=0.25
CHUNK_SIZE=350
CHUNK_OVERLAP=60
```

> **Provider Flexibility**: The system uses the OpenAI-compatible Chat Completions format. You can switch to **Groq Cloud** (`LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_MODEL=openai/gpt-oss-20b`), **OpenAI** (`gpt-4o-mini`), or a local **Ollama** instance (`http://localhost:11434/v1`) simply by changing the `.env` settings.

---

## 🚀 Running the System

### 1. Interactive Web Interface (Streamlit)
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser:
- **Upload Documents**: Use the left sidebar to upload PDF, DOCX, or TXT files.
- **Ask Questions**: Type queries into the chat bar; answers appear with expandable citations showing document name, page number, and similarity score.
- **Answer Styles**: Toggle between **Simple** (concise), **Detailed** (comprehensive), and **Academic** (rigorous) modes.
- **Exam Prep Mode**: Activates structured key points, formulas, definitions, and practice questions.
- **Summarize & Explore**: Click "Summarize Document" or generate suggested study questions.

### 2. Programmatic REST API (FastAPI)
```powershell
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 🧪 Automated Testing

The project includes unit and integration tests covering all 4 architectural tiers:

```powershell
# Run the complete test suite
python -m unittest discover tests

# Or run individual domain test modules:
python -m unittest tests/test_ingestion.py   # Loader, parser, chunker, manager
python -m unittest tests/test_retrieval.py   # Embeddings, FAISS indexing, search
python -m unittest tests/test_generation.py  # Prompt building, Q&A synthesis
python -m unittest tests/test_api.py         # FastAPI REST endpoints
```

All 22 unit tests execute in under 10 seconds and validate pipeline integrity.

---

## 📁 Repository Structure

```text
RAG_Document_Assistant/
├── docs/
│   ├── architecture.md          # Technical architecture & pipeline diagrams
│   ├── api.md                   # REST API specification & curl examples
│   └── setup.md                 # Detailed setup & troubleshooting guide
├── src/
│   ├── ingestion/               # Document loaders, parsers, chunking & registry
│   │   ├── loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── manager.py
│   ├── retrieval/               # Embeddings, FAISS vector store & retriever
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── generation/              # LLM client adapter, prompt builder & synthesizer
│   │   ├── llm.py
│   │   ├── prompt.py
│   │   └── response.py
│   ├── api/                     # FastAPI routes & Pydantic request/response schemas
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── config/                  # Global application configuration & path definitions
│   │   └── settings.py
│   ├── ui/                      # Streamlit session chat state manager
│   │   └── chat_manager.py
│   └── utils/                   # Structured logging & text utilities
│       ├── logger.py
│       └── helpers.py
├── data/
│   ├── raw/                     # Raw input documents
│   ├── processed/               # Extracted and processed text artifacts
│   ├── samples/                 # Sample documents for testing
│   └── vectorstore/             # FAISS binary index and metadata pickle files
├── scripts/
│   ├── ingest.py                # Batch CLI document ingestion script
│   └── rebuild_index.py         # Index reset & rebuild CLI utility
├── tests/
│   ├── test_ingestion.py        # Ingestion & parser unit tests
│   ├── test_retrieval.py        # Embeddings & FAISS unit tests
│   ├── test_generation.py       # Prompting & synthesis unit tests
│   └── test_api.py              # FastAPI endpoint integration tests
├── app.py                       # Streamlit web application entrypoint
├── requirements.txt             # Python dependencies
├── .env                         # Active configuration & API key (git-ignored)
├── .env.example                 # Environment template
└── .gitignore                   # Local secrets & data exclusions
```
