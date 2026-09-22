# REST API Reference

The **RAG Document Assistant** exposes a high-performance REST API built with FastAPI.

## Starting the API Server

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints

### 1. Health Check
`GET /health`

Returns service status, configured LLM model, base URL, and total indexed vector count.

**Response (200 OK):**
```json
{
  "status": "ok",
  "llm_configured": true,
  "model": "gemini-3.6-flash",
  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
  "total_indexed_chunks": 48
}
```

---

### 2. Query Documents
`POST /query`

Performs semantic search across indexed documents and returns an answer with source citations.

**Request Body:**
```json
{
  "query": "What are the core properties of MQTT?",
  "top_k": 5,
  "answer_style": "Simple",
  "exam_mode": false
}
```

**Response (200 OK):**
```json
{
  "query": "What are the core properties of MQTT?",
  "answer": "MQTT is a lightweight publish-subscribe protocol designed for low-bandwidth networks...",
  "found_context": true,
  "sources": [
    {
      "document": "IoT_Protocols.pdf",
      "page": 3,
      "score": 0.812
    }
  ]
}
```

---

### 3. List Documents
`GET /documents`

Retrieves all indexed documents along with page count, character count, and chunk counts.

**Response (200 OK):**
```json
{
  "count": 1,
  "documents": [
    {
      "filename": "IoT_Protocols.pdf",
      "pages": 12,
      "characters": 18450,
      "chunks": 48,
      "size": "450.2 KB"
    }
  ]
}
```

---

### 4. Upload Document
`POST /documents/upload`

Uploads and processes a document (`multipart/form-data`).

**Request Parameters:**
- `file`: Binary file (PDF, TXT, DOCX).

**Response (200 OK):**
```json
{
  "filename": "IoT_Protocols.pdf",
  "message": "'IoT_Protocols.pdf' processed successfully — 12 pages, 48 chunks indexed."
}
```

---

### 5. Summarize Document
`POST /documents/{doc_name}/summarize`

Generates a structured map-reduce summary for an existing indexed document.

**Request Body:**
```json
{
  "max_chunks_per_batch": 6
}
```

**Response (200 OK):**
```json
{
  "document": "IoT_Protocols.pdf",
  "summary": "1. Main Topic\nInternet of Things transport and messaging protocols..."
}
```

---

### 6. Suggested Questions
`GET /documents/{doc_name}/suggested-questions`

Returns up to 5 suggested questions based on document content.

**Response (200 OK):**
```json
{
  "document": "IoT_Protocols.pdf",
  "questions": [
    "What is the difference between MQTT and CoAP?",
    "How does QoS work in MQTT?"
  ]
}
```

---

### 7. Clear Documents
`DELETE /documents`

Resets the FAISS vector store and removes all documents from the registry.

**Response (200 OK):**
```json
{
  "message": "All documents cleared successfully."
}
```
