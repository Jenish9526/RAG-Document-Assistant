"""Modern, clean ChatGPT-style user interface for RAG Document Assistant."""

import base64
import streamlit as st

from src.config.settings import APP_TITLE, TOP_K
import src.ingestion.manager as dm
import src.ui.chat_manager as cm
import src.generation.response as rag_engine
import src.generation.llm as llm_service

# =======================================================================
# SVG ASSETS & ICONS (No Emojis)
# =======================================================================
ICON_SPARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/></svg>"""
ICON_USER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>"""

AVATAR_ASSISTANT = f"data:image/svg+xml;base64,{base64.b64encode(ICON_SPARK_SVG.encode()).decode()}"
AVATAR_USER = f"data:image/svg+xml;base64,{base64.b64encode(ICON_USER_SVG.encode()).decode()}"

# =======================================================================
# PAGE SETUP
# =======================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=AVATAR_ASSISTANT,
    layout="wide",
    initial_sidebar_state="expanded",
)

cm.init_chat_history()
store = dm.get_vector_store()

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# =======================================================================
# PROFESSIONAL CUSTOM STYLES (ChatGPT Aesthetics)
# =======================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Clean subtle scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.22);
    }

    /* Sidebar refinement */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Professional buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .stButton > button:hover {
        border-color: #10a37f;
        color: #10a37f;
    }

    /* New Chat Button */
    .new-chat-btn button {
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #ececf1 !important;
        padding: 0.5rem 0.85rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
    }
    .new-chat-btn button:hover {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Document card styling */
    .doc-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    /* Chat bubble container */
    [data-testid="stChatMessage"] {
        padding: 1.25rem 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Citation expander styling */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        margin-top: 0.75rem !important;
    }

    /* Suggested query pills */
    .suggestion-card {
        padding: 14px 18px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.18s ease;
        margin-bottom: 8px;
    }
    .suggestion-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.18);
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================================================================
# SIDEBAR
# =======================================================================
with st.sidebar:
    # Sleek Brand Header
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:11px; margin: 4px 0 16px 0;">
            <div style="background: linear-gradient(135deg, #10a37f, #0c8a6b); width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 12px rgba(16, 163, 127, 0.25);">
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/>
                </svg>
            </div>
            <div>
                <div style="font-weight:600; font-size:1.02rem; letter-spacing:-0.2px; line-height: 1.2;">Assistant</div>
                <div style="font-size:0.75rem; color:#8e8ea0; font-weight:400;">RAG Document Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New Chat Button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋ New chat", use_container_width=True):
        cm.clear_history()
        st.session_state.suggested_questions = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if not llm_service.is_configured():
        st.warning("LLM API key not configured. Set LLM_API_KEY in your .env file.")

    # ---------------- DOCUMENT UPLOAD ----------------
    st.markdown(
        """
        <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.8px; color:#8e8ea0; font-weight:600; margin-bottom: 8px;">
            Files
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            already_done_key = f"processed_{uploaded.name}_{uploaded.size}"
            if st.session_state.get(already_done_key):
                continue

            with st.status(f"Indexing {uploaded.name}...", expanded=False) as status:
                st.write("Extracting content")
                file_bytes = uploaded.getvalue()
                st.write("Generating vector embeddings")
                result = dm.add_document(uploaded.name, file_bytes)
                st.write("Updating FAISS index")

                if result["success"]:
                    status.update(label=f"{uploaded.name} indexed", state="complete")
                    st.session_state.suggested_questions = []
                elif result["duplicate"]:
                    status.update(label=f"{result['message']}", state="complete")
                else:
                    status.update(label=f"Error: {result['message']}", state="error")

            st.session_state[already_done_key] = True

    # ---------------- DOCUMENT LIST ----------------
    registry = dm.get_document_registry()
    if registry:
        st.markdown(
            f"""
            <div style="font-size:0.75rem; color:#8e8ea0; margin: 12px 0 6px 0; font-weight:500;">
                Indexed Documents ({len(registry)})
            </div>
            """,
            unsafe_allow_html=True,
        )
        for info in registry.values():
            with st.expander(info["filename"]):
                st.caption(
                    f"Pages: {info['pages']}  •  Chunks: {info['chunks']}  •  Size: {info['size']}"
                )
    else:
        st.caption("No files uploaded yet. Add a PDF, TXT, or DOCX above.")

    # ---------------- DOCUMENT ACTIONS ----------------
    doc_names = dm.list_document_names()
    if doc_names:
        st.write("")
        st.markdown(
            """
            <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.8px; color:#8e8ea0; font-weight:600; margin-bottom: 6px;">
                Actions
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_doc = st.selectbox(
            "Target Document",
            doc_names,
            key="summarize_target",
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Summarize", use_container_width=True):
                with st.spinner(f"Summarizing {selected_doc}..."):
                    summary = rag_engine.summarize_document(selected_doc, store)
                cm.add_message("assistant", f"**Executive Summary of {selected_doc}:**\n\n{summary}")
                st.rerun()
        with col_b:
            if st.button("Questions", use_container_width=True):
                with st.spinner("Generating questions..."):
                    st.session_state.suggested_questions = rag_engine.generate_suggested_questions(
                        selected_doc, store
                    )
                st.rerun()

    # ---------------- FOOTER & RESET ----------------
    st.divider()
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("Clear Chat", use_container_width=True):
            cm.clear_history()
            st.rerun()
    with col_clear2:
        if st.button("Reset Store", use_container_width=True):
            dm.clear_all_documents()
            st.session_state.suggested_questions = []
            for key in list(st.session_state.keys()):
                if key.startswith("processed_"):
                    del st.session_state[key]
            st.rerun()

    st.markdown(
        """
        <div style="margin-top: 14px; font-size: 0.72rem; color: #6e6e80; text-align: center;">
            Grounded Semantic Search &bull; FAISS Vector Store
        </div>
        """,
        unsafe_allow_html=True,
    )


# =======================================================================
# MAIN CHAT AREA
# =======================================================================
history = cm.get_history()

def handle_question(question: str):
    """Executes grounded RAG pipeline and records query/response in chat history."""
    cm.add_message("user", question)
    with st.spinner("Searching documents and formulating response..."):
        result = rag_engine.answer_question(
            question,
            store,
            top_k=TOP_K,
            answer_style="Detailed",
            exam_mode=False,
        )
    cm.add_message("assistant", result["answer"], sources=result["sources"])


# Handle suggested-question clicks
pending = st.session_state.pop("_pending_question", None)
if pending:
    handle_question(pending)
    st.rerun()

# ---------------- WELCOME / EMPTY STATE ----------------
if not history:
    st.markdown(
        """
        <div style="text-align: center; max-width: 620px; margin: 48px auto 28px auto;">
            <div style="width: 52px; height: 52px; margin: 0 auto 18px auto; border-radius: 12px; background: rgba(16, 163, 127, 0.1); border: 1px solid rgba(16, 163, 127, 0.25); display: flex; align-items: center; justify-content: center;">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/>
                </svg>
            </div>
            <h2 style="font-weight: 600; font-size: 1.65rem; margin-bottom: 8px; letter-spacing: -0.4px;">
                How can I assist with your documents?
            </h2>
            <p style="color: #8e8ea0; font-size: 0.94rem; line-height: 1.55; margin-bottom: 24px;">
                Upload PDF, Word, or text files in the sidebar to ask questions, explore topics, and extract verified answers with exact page citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Quick Starters
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summarize the key findings and conclusions", use_container_width=True):
            st.session_state["_pending_question"] = "Summarize the key findings and conclusions of the uploaded document."
            st.rerun()
        if st.button("List the core definitions and terms", use_container_width=True):
            st.session_state["_pending_question"] = "What are the core definitions and important concepts introduced in this document?"
            st.rerun()
    with col2:
        if st.button("Explain the main methodology or structure", use_container_width=True):
            st.session_state["_pending_question"] = "Explain the main methodology, architecture, or structure described in this document."
            st.rerun()
        if st.button("What practical recommendations are provided?", use_container_width=True):
            st.session_state["_pending_question"] = "What practical takeaways, solutions, or recommendations does this document offer?"
            st.rerun()

# ---------------- SUGGESTED QUESTIONS (Chips) ----------------
if st.session_state.suggested_questions:
    st.markdown(
        """
        <div style="font-size:0.8rem; font-weight:600; color:#8e8ea0; margin: 12px 0 6px 0;">
            Derived Exploration Questions
        </div>
        """,
        unsafe_allow_html=True,
    )
    for q in st.session_state.suggested_questions:
        if st.button(f"→  {q}", key=f"sug_{q}"):
            st.session_state["_pending_question"] = q
            st.rerun()

# ---------------- CHAT HISTORY DISPLAY ----------------
for message in history:
    avatar = AVATAR_ASSISTANT if message["role"] == "assistant" else AVATAR_USER
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Citations ({len(message['sources'])} references)"):
                for i, src in enumerate(message["sources"], start=1):
                    confidence = int(src["score"] * 100) if src.get("score") else None
                    conf_str = f" • Match confidence: {confidence}%" if confidence else ""
                    st.markdown(
                        f"""
                        <div style="padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.86rem;">
                            <strong>{i}. {src['document']}</strong> — Page {src['page']}<span style="color:#8e8ea0;">{conf_str}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------- CHAT INPUT ----------------
user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    handle_question(user_question)
    st.rerun()
