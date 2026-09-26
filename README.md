# RAG Document Assistant

An intelligent, multi-format document question-answering and summarization system powered by **Retrieval-Augmented Generation (RAG)**, **SentenceTransformers**, **FAISS Vector Search**, **FastAPI**, and **Streamlit**.

---

## 📌 Features

- **Multi-Format Ingestion**: Ingests and parses **PDF**, **DOCX**, and **TXT** files with automatic page indexing and SHA-256 deduplication.
- **Dense Vector Retrieval**: Uses `all-MiniLM-L6-v2` dense embeddings (384-d) with **FAISS** (`IndexFlatIP`) for exact cosine similarity matching.
- **Claude-Style Interactive UI**:
  - **Left vs. Right Chat Flow**: User questions sent from the far right; assistant responses received on the left.
  - **Instant Rendering on Enter**: Your question appears immediately in the chat when submitted.
  - **Single-Line Thought Pill**: A sleek 1-line progress indicator (`[ ⟳ Searching document index... ▾ ]`) that expands on tap to display real-time FAISS search and synthesis steps.
- **Grounded Anti-Hallucination Answers**: Injects top-K relevant excerpts into strict prompt templates with source citations (document name, page number, confidence score).
- **Dual Presentation**: Interactive Streamlit web interface and headless FastAPI REST API with Swagger docs.
- **Provider Agnostic**: Works seamlessly with Google Gemini, Groq, OpenAI, or local Ollama instances via OpenAI-compatible Chat Completions format.

---

## 🏗️ Architecture

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                  Client & Interfaces                        │
 │   • Streamlit Web UI (app.py)   • FastAPI API (/docs)       │
 └──────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼
 ┌──────────────────────────────┐ ┌────────────────────────────┐
 │     1. Ingestion Layer       │ │    2. Retrieval Layer      │
 │  • Load PDF, DOCX, TXT       │ │  • all-MiniLM-L6-v2 Embed  │
 │  • Sliding window chunking   │ │  • FAISS Vector Store      │
 │  • SHA-256 deduplication     │ │  • Cosine similarity search│
 └──────────────┬───────────────┘ └────────────┬───────────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                    3. Generation Layer                      │
 │  • Context assembly with anti-hallucination guardrails      │
 │  • LLM synthesis (Gemini / OpenAI / Groq / Ollama)          │
 │  • Grounded citations with exact page numbers               │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 2. Clone & Setup Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/Jenish9526/RAG-Document-Assistant.git
cd RAG_Document_Assistant

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate
# If script execution is disabled: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate (Linux / macOS)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (`.env`)
Create a `.env` file in the root directory (or copy `.env.example`):

```env
# LLM Provider (Google Gemini free key: https://aistudio.google.com/app/apikey)
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.7-flash

# Retrieval Pipeline Defaults
TOP_K=5
SIMILARITY_THRESHOLD=0.25
CHUNK_SIZE=350
CHUNK_OVERLAP=60
```

---

## 🖥️ Running the Application

### Streamlit Web Interface (Recommended)
```powershell
streamlit run app.py
```
> **Windows tip**: If your Python launcher has path issues, run directly via:
> ```powershell
> .\venv\Scripts\python.exe -m streamlit run app.py
> ```
* Open **`http://localhost:8501`** in your browser.
* Upload documents via the sidebar, ask questions, and explore citations.

### FastAPI REST API
```powershell
uvicorn src.api.routes:app --reload
```
* **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🛠️ CLI Utilities & Testing

* **Batch Ingest Documents from `data/raw/`**:
  ```powershell
  python scripts/ingest.py
  ```

* **Reset / Rebuild Vector Store**:
  ```powershell
  python scripts/rebuild_index.py --reingest
  ```

* **Run Unit Tests**:
  ```powershell
  python -m unittest discover tests
  ```

---

## 📁 Repository Structure

```text
RAG_Document_Assistant/
├── src/
│   ├── ingestion/       # Document loaders, parsers, sliding-window chunker & registry
│   ├── retrieval/       # SentenceTransformers, FAISS vector store & retriever
│   ├── generation/      # LLM client adapter, prompt builder & synthesis engine
│   ├── api/             # FastAPI routes & Pydantic request/response schemas
│   ├── config/          # Application settings & environment configuration
│   └── ui/              # Session chat state manager
├── data/
│   ├── raw/             # Raw input documents
│   └── vectorstore/     # Persisted FAISS index & metadata files
├── scripts/             # CLI utilities for ingestion & index rebuilding
├── tests/               # Unit and integration test suite
├── app.py               # Streamlit web application
├── requirements.txt     # Python package dependencies
├── .env.example         # Example configuration file
└── README.md            # Project documentation
```

---

## 📄 License
This project is open-source under the MIT License.
