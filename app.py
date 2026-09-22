"""
app.py
======

Purpose
-------
The Streamlit entry point — the file you run with:

    streamlit run app.py

Purpose of this file:
    Pure UI / orchestration. It does NOT contain RAG logic itself —
    it calls into document_manager.py, chat_manager.py, and
    rag_engine.py, and renders their results.

How this connects to the rest of the project:

    app.py
      ├── document_manager.py  -> upload, dedup, stats, vector store access
      ├── chat_manager.py      -> chat history (session_state)
      ├── rag_engine.py        -> answer_question(), summarize_document(),
      │                          generate_suggested_questions()
      └── llm_service.py       -> (indirectly) checks is_configured()

Run instructions (Windows):
    python -m venv venv
    venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, TOP_K
import document_manager as dm
import chat_manager as cm
import rag_engine
import llm_service


# =======================================================================
# PAGE SETUP
# =======================================================================
st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
cm.init_chat_history()
store = dm.get_vector_store()

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "Simple"
if "exam_mode" not in st.session_state:
    st.session_state.exam_mode = False
if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []


# =======================================================================
# SIDEBAR
# =======================================================================
with st.sidebar:
    st.markdown(f"## 📚 {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    if not llm_service.is_configured():
        st.warning("⚠️ LLM API key is not configured. Set `LLM_API_KEY` in your `.env` file.")

    st.divider()

    # ---------------- UPLOAD ----------------
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, TXT, DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            already_done_key = f"processed_{uploaded.name}_{uploaded.size}"
            if st.session_state.get(already_done_key):
                continue

            with st.status(f"Processing {uploaded.name}...", expanded=False) as status:
                st.write("✓ Reading file")
                file_bytes = uploaded.getvalue()
                st.write("✓ Extracting & chunking text")
                st.write("✓ Generating embeddings")
                result = dm.add_document(uploaded.name, file_bytes)
                st.write("✓ Building FAISS index")

                if result["success"]:
                    status.update(label=f"✅ {uploaded.name} ready", state="complete")
                    st.session_state.suggested_questions = []  # reset, regenerate on demand
                elif result["duplicate"]:
                    status.update(label=f"⚠️ {result['message']}", state="complete")
                else:
                    status.update(label=f"❌ {result['message']}", state="error")

            st.session_state[already_done_key] = True

    # ---------------- DOCUMENT LIST ----------------
    st.markdown("### 📄 Documents")
    registry = dm.get_document_registry()
    if not registry:
        st.caption("No documents uploaded yet.")
    else:
        for info in registry.values():
            with st.expander(f"📄 {info['filename']}"):
                st.write(f"**Pages:** {info['pages']}")
                st.write(f"**Characters:** {info['characters']:,}")
                st.write(f"**Chunks:** {info['chunks']}")
                st.write(f"**Size:** {info['size']}")
                st.write("**Status:** Indexed ✓")

    st.divider()

    # ---------------- SETTINGS ----------------
    st.markdown("### ⚙️ Settings")
    st.session_state.answer_style = st.radio(
        "Answer Style", ["Simple", "Detailed"],
        index=["Simple", "Detailed"].index(st.session_state.answer_style),
        horizontal=True,
    )
    st.session_state.top_k = st.slider("Number of Results (Top K)", min_value=1, max_value=10,
                                        value=st.session_state.top_k)
    st.session_state.exam_mode = st.toggle("🎓 Exam Mode", value=st.session_state.exam_mode)

    st.divider()

    # ---------------- TOOLS ----------------
    st.markdown("### 🛠️ Tools")
    doc_names = dm.list_document_names()

    if doc_names:
        selected_doc = st.selectbox("Select a document", doc_names, key="summarize_target")
        if st.button("📄 Summarize Document", use_container_width=True):
            with st.spinner(f"Summarizing {selected_doc}..."):
                summary = rag_engine.summarize_document(selected_doc, store)
            cm.add_message("assistant", f"**Summary of {selected_doc}:**\n\n{summary}")
            st.rerun()

        if st.button("💡 Suggested Questions", use_container_width=True):
            with st.spinner("Generating suggestions..."):
                st.session_state.suggested_questions = rag_engine.generate_suggested_questions(
                    selected_doc, store
                )
    else:
        st.caption("Upload a document to unlock summarization and suggestions.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            cm.clear_history()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Documents", use_container_width=True):
            dm.clear_all_documents()
            st.session_state.suggested_questions = []
            for key in list(st.session_state.keys()):
                if key.startswith("processed_"):
                    del st.session_state[key]
            st.rerun()


# =======================================================================
# MAIN AREA
# =======================================================================
st.markdown(f"# 📚 {APP_TITLE}")
st.caption(APP_SUBTITLE)

history = cm.get_history()

if not history and store.total_chunks == 0:
    st.info(
        "### Welcome to RAG Document Assistant\n\n"
        "Ask questions about your documents using AI.\n\n"
        "**How it works:**\n"
        "1. Upload a document (PDF, TXT, or DOCX) from the sidebar\n"
        "2. The document is processed and indexed\n"
        "3. Relevant information is retrieved for each question\n"
        "4. Ask questions in the chat box below\n"
        "5. Get AI-generated answers with document sources"
    )

# ---------------- SUGGESTED QUESTIONS (chips) ----------------
if st.session_state.suggested_questions:
    st.markdown("**💡 Suggested Questions:**")
    cols = st.columns(len(st.session_state.suggested_questions))
    for col, question in zip(cols, st.session_state.suggested_questions):
        if col.button(question, use_container_width=True):
            st.session_state["_pending_question"] = question

# ---------------- CHAT HISTORY DISPLAY ----------------
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**{i}. {src['document']} — Page {src['page']}** "
                                f"(relevance: {src['score']})")


def handle_question(question: str):
    """Runs the RAG pipeline for one question and updates chat history."""
    cm.add_message("user", question)
    with st.spinner("Searching documents and generating answer..."):
        result = rag_engine.answer_question(
            question,
            store,
            top_k=st.session_state.top_k,
            answer_style=st.session_state.answer_style,
            exam_mode=st.session_state.exam_mode,
        )
    cm.add_message("assistant", result["answer"], sources=result["sources"])


# Handle a suggested-question chip click (from previous run).
pending = st.session_state.pop("_pending_question", None)
if pending:
    handle_question(pending)
    st.rerun()

# ---------------- CHAT INPUT ----------------
user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    handle_question(user_question)
    st.rerun()
