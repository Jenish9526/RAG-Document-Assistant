# Installation & Setup Guide

This guide covers setting up **RAG Document Assistant** in both local development environments and Docker containers.

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- **Git**
- **LLM API Key**: Google Gemini (Recommended, 1M TPM free tier) or Groq / OpenAI

---

## 1. Local Python Setup

### Clone Repository
```bash
git clone https://github.com/Jenish9526/RAG_Document_Assistant.git
cd RAG_Document_Assistant
```

### Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Environment Configuration

Copy the example environment template and add your credentials:

```bash
cp .env.example .env
```

### Recommended: Google Gemini (Free 1M TPM Tier)
```env
LLM_API_KEY=your_google_ai_studio_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
```

### Alternative: Groq Cloud
```env
LLM_API_KEY=your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

### Alternative: Local Ollama
```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

---

## 3. Running Applications

### Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Headless REST API (FastAPI)
```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: `http://localhost:8000/docs`.

---

## 4. CLI Tools

### Ingest Documents in Batch
Place documents in `data/raw/` and run:
```bash
python scripts/ingest.py
```
Or specify a custom file / folder:
```bash
python scripts/ingest.py --path /path/to/documents
```

### Rebuild or Reset Index
```bash
# Clear the index
python scripts/rebuild_index.py

# Clear and re-ingest all files from data/raw/
python scripts/rebuild_index.py --reingest
```

---

## 5. Docker Deployment

### Build Image
```bash
docker build -t rag-document-assistant:latest .
```

### Run Container
```bash
docker run -d \
  --name rag-assistant \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data/vectorstore:/app/data/vectorstore \
  rag-document-assistant:latest
```

---

## 6. Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific domain test suite
python -m unittest tests/test_ingestion.py
python -m unittest tests/test_retrieval.py
python -m unittest tests/test_generation.py
python -m unittest tests/test_api.py
```
