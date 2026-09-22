<div align="center">

# 📚 RAG Document Assistant

**Production-grade, modular Retrieval-Augmented Generation (RAG) assistant for intelligent document search, question-answering, and summarization.**

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-2ea44f?style=for-the-badge&logo=github-actions)](.github/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-00599C?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#supported-llm-providers">Supported LLMs</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#docker-deployment">Docker</a> •
  <a href="#testing">Testing</a>
</p>

</div>

---

## 🌟 Key Features

- 📂 **Multi-Format Ingestion**: Upload PDF, TXT, and DOCX documents with robust text cleaning and chunking.
- 🔍 **Dense Semantic Search**: Fast, exact cosine similarity retrieval powered by **FAISS** (`IndexFlatIP`) and Sentence Transformers (`all-MiniLM-L6-v2`).
- 🤖 **Provider Agnostic**: OpenAI-compatible adapter supporting **Google Gemini**, **Groq**, **OpenAI**, and local **Ollama** models.
- ⚡ **Resilient Networking**: Automatic HTTP 429 rate-limit backoff, request pacing, and detailed provider error diagnostics.
- 📚 **Grounded Citations**: Page-level source attribution with similarity scores to combat hallucination.
- 📝 **Dual Answer Modes**: Toggle between concise bulleted explanations and structured **Exam Mode** (Definition, Steps, Example, Complexity).
- 📑 **Map-Reduce Summarization**: Hierarchical chunk-based document summarization designed to handle documents of any size without exceeding context windows.
- 💡 **Dynamic Question Suggestions**: Automatic exploration prompts generated from document context.
- 💾 **Persistent Vector Storage**: On-disk serialization of FAISS indices and document registries across session restarts.

---

## 🏗️ System Architecture

```text
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│ User Documents  │ ───>  │ Text Extraction │ ───>  │ Overlap Chunking │
│ (PDF, TXT, DOCX)│       │ & Normalization │       │   (350 words)    │
└─────────────────┘       └─────────────────┘       └──────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│  Streamlit UI   │ <───  │ Labeled Context │ <───  │ Sentence Embeds  │
│ (Chat & Sources)│       │  & Prompt Build │       │ (all-MiniLM-L6)  │
└─────────────────┘       └─────────────────┘       └──────────────────┘
         ▲                         ▲                          │
         │                         │                          ▼
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│   OpenAI / LLM  │ <───> │ Top-K Search &  │ <───  │   FAISS Index    │
│  API (Adapter)  │       │ Relevance Filter│       │   (FlatIP / L2)  │
└─────────────────┘       └─────────────────┘       └──────────────────┘
```

---

## 🔄 Supported LLM Providers

The application uses a unified OpenAI-compatible adapter (`llm_service.py`). Swap providers anytime via `.env` without modifying code:

| Provider | Recommended Model | Best For | Typical Free Tier |
| :--- | :--- | :--- | :--- |
| **Google Gemini** *(Default)* | `gemini-3.5-flash-lite` | Ultra-high throughput & long context | 1,000,000 TPM (Free) |
| **Groq** | `openai/gpt-oss-20b` | Near-instant inference speed | Free on-demand tier |
| **Local Ollama** | `llama3.2` | 100% offline, privacy & zero limits | Unlimited / Local |
| **OpenAI** | `gpt-4o-mini` | High-accuracy general tasks | Pay-as-you-go |

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/RAG_Document_Assistant.git
cd RAG_Document_Assistant
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and add your API key:

```env
# Example for Google Gemini (Get free key: https://aistudio.google.com/app/apikey)
LLM_API_KEY=AIzaSy...your_gemini_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
```

### 4. Run the Application

```powershell
# Direct invocation:
.\venv\Scripts\python.exe -m streamlit run app.py

# Or if virtualenv is activated:
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

You can run the assistant in a container without local Python setup:

```bash
# Build Docker image
docker build -t rag-document-assistant .

# Run container with environment variables
docker run -d -p 8501:8501 --env-file .env --name rag-app rag-document-assistant
```

Access the app at `http://localhost:8501`.

---

## 🧪 Testing & Validation

The project includes both isolated unit/integration tests and live API validation scripts:

```bash
# 1. Run all unit and integration tests (mocked & fast, safe for CI)
python -m unittest discover tests

# 2. Run live end-to-end tests against configured LLM API
python tests/live_test_all_functions.py
```

### Test Coverage Highlights
- ✅ **Document Processor**: PDF, TXT, DOCX extraction, boundary chunking & overlap logic.
- ✅ **Vector Store**: FAISS indexing, L2 inner-product search ranking, on-disk persistence.
- ✅ **RAG Pipeline**: Dynamic prompt compilation, hallucination guarding, Map-Reduce summarization.
- ✅ **Resilience**: HTTP 429 rate-limit backoff, token-bucket pacing, and registry synchronization.

---

## 📁 Project Structure

```text
RAG_Document_Assistant/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated CI test pipeline
│   ├── ISSUE_TEMPLATE/          # Structured issue templates
│   └── PULL_REQUEST_TEMPLATE.md # PR standards checklist
├── data/
│   ├── documents/               # Raw uploaded document storage
│   └── processed/               # Intermediate processing artifacts
├── tests/
│   ├── __init__.py
│   ├── test_rag_pipeline.py     # Comprehensive unit/integration test suite
│   └── live_test_all_functions.py # Live API and RAG pipeline verification
├── vector_db/                   # Persisted FAISS vector indices & registries
├── app.py                       # Streamlit UI & interaction orchestrator
├── chat_manager.py              # Session-state chat history manager
├── config.py                    # Centralized hyperparameter configuration
├── document_manager.py          # Validation, deduplication & registry sync
├── document_processor.py        # Text extraction, cleaning & chunking
├── embeddings.py                # Sentence-Transformer cached vectorizer
├── llm_service.py               # Isolated LLM adapter with retry logic
├── rag_engine.py                # Core RAG retrieval, prompt & summarizer
├── utils.py                     # Hashing, formatting & normalization tools
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git ignore policies
├── .dockerignore                # Container build context exclusions
├── Dockerfile                   # Container specification
├── pyproject.toml               # Modern Python packaging configuration
├── requirements.txt             # Direct dependencies
├── CONTRIBUTING.md              # Open-source contribution guidelines
├── CODE_OF_CONDUCT.md           # Community code of conduct
├── SECURITY.md                  # Vulnerability reporting protocol
└── LICENSE                      # MIT Open Source License
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting pull requests.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
