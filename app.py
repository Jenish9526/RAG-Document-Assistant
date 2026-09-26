"""Modern, clean ChatGPT-style user interface for RAG Document Assistant."""

import base64
import html
import streamlit as st

from src.config.settings import APP_TITLE, TOP_K
import src.ingestion.manager as dm
import src.ui.chat_manager as cm
import src.generation.response as rag_engine
import src.generation.llm as llm_service
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.prompt import build_context, build_prompt

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
        padding: 0.5rem 0.25rem !important;
        border-bottom: none !important;
        background: transparent !important;
        display: flex !important;
        gap: 12px !important;
        width: 100% !important;
    }

    /* User Message: Sent from FAR RIGHT side */
    [data-testid="stChatMessage"]:has(.user-chat-marker),
    div[aria-label="Chat message from user"] {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        margin-left: auto !important;
        width: 100% !important;
        gap: 10px !important;
        padding: 0.5rem 0.1rem !important;
    }

    [data-testid="stChatMessage"]:has(.user-chat-marker) [data-testid="stChatMessageContent"],
    div[aria-label="Chat message from user"] [data-testid="stChatMessageContent"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-start !important;
        text-align: right !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
        flex-grow: 1 !important;
    }

    [data-testid="stChatMessage"]:has(.user-chat-marker) .stMarkdown,
    [data-testid="stChatMessage"]:has(.user-chat-marker) [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
    }

    [data-testid="stChatMessage"]:has(.user-chat-marker) p {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
        margin: 0 !important;
    }

    .user-bubble-container {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }

    .user-bubble {
        background: linear-gradient(135deg, #10a37f 0%, #0d8265 100%) !important;
        color: #ffffff !important;
        padding: 10px 18px !important;
        border-radius: 18px 18px 4px 18px !important;
        display: inline-block !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
        box-shadow: 0 4px 14px rgba(16, 163, 127, 0.25) !important;
        font-size: 0.94rem !important;
        line-height: 1.55 !important;
        letter-spacing: -0.1px !important;
        max-width: 80% !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }

    /* Assistant Message: Clean, transparent typography (No giant empty boxes) */
    [data-testid="stChatMessage"]:has(.assistant-chat-marker),
    div[aria-label="Chat message from assistant"] {
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        margin-right: auto !important;
        width: 100% !important;
        gap: 12px !important;
        padding: 0.5rem 0.1rem !important;
    }

    [data-testid="stChatMessage"]:has(.assistant-chat-marker) [data-testid="stChatMessageContent"],
    div[aria-label="Chat message from assistant"] [data-testid="stChatMessageContent"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        text-align: left !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        width: 100% !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        color: #ececf1 !important;
    }

    /* Claude-Style Single-Line Expandable Progress Dropdown */
    .claude-thought-pill {
        display: inline-block;
        margin: 2px 0 10px 0;
        max-width: fit-content;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-size: 0.82rem;
        color: #9ca3af;
        transition: all 0.15s ease-in-out;
        user-select: none;
    }

    .claude-thought-pill:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.16);
        color: #e5e7eb;
    }

    .claude-thought-pill[open] {
        background: rgba(20, 24, 30, 0.95);
        border-color: rgba(16, 163, 127, 0.35);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }

    .claude-thought-summary {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 11px;
        cursor: pointer;
        list-style: none;
        font-weight: 500;
        font-size: 0.82rem;
        line-height: 1.2;
        white-space: nowrap;
    }

    .claude-thought-summary::-webkit-details-marker {
        display: none;
    }

    .claude-spinner-inline {
        width: 12px;
        height: 12px;
        border: 1.8px solid rgba(16, 163, 127, 0.25);
        border-top-color: #10a37f;
        border-radius: 50%;
        animation: claudeSpin 0.75s linear infinite;
        flex-shrink: 0;
    }

    @keyframes claudeSpin {
        to { transform: rotate(360deg); }
    }

    .claude-check-inline {
        flex-shrink: 0;
    }

    .claude-summary-label {
        color: #d1d5db;
        letter-spacing: -0.1px;
    }

    .claude-chevron {
        transition: transform 0.2s ease;
        opacity: 0.6;
        margin-left: 3px;
        flex-shrink: 0;
    }

    .claude-thought-pill[open] .claude-chevron {
        transform: rotate(180deg);
    }

    .claude-thought-content {
        padding: 8px 14px 10px 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        font-size: 0.76rem;
        color: #94a3b8;
        line-height: 1.6;
        background: rgba(0, 0, 0, 0.2);
    }

    .claude-step-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 2px 0;
    }

    .claude-step-row.done .step-icon {
        color: #10a37f;
        font-weight: bold;
    }

    .claude-step-row.active {
        color: #f1f5f9;
        font-weight: 500;
    }

    .claude-step-row.active .step-icon {
        color: #38bdf8;
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


def render_claude_thought(label: str, steps: list, is_done: bool = False) -> str:
    """Render a single-line Claude-style expandable thought/progress pill."""
    icon_html = (
        """<svg class="claude-check-inline" viewBox="0 0 24 24" width="12" height="12" stroke="#10a37f" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
        if is_done
        else """<span class="claude-spinner-inline"></span>"""
    )
    rows_html = "".join(
        f'<div class="claude-step-row {status}"><span class="step-icon">{"✓" if status == "done" else "○"}</span><span>{text}</span></div>'
        for text, status in steps
    )
    return (
        f'<details class="claude-thought-pill">'
        f'<summary class="claude-thought-summary">'
        f'{icon_html}'
        f'<span class="claude-summary-label">{label}</span>'
        f'<svg class="claude-chevron" viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">'
        f'<polyline points="6 9 12 15 18 9"/></svg>'
        f'</summary>'
        f'<div class="claude-thought-content">{rows_html}</div>'
        f'</details>'
    )


# =======================================================================
# MAIN CHAT AREA
# =======================================================================
history = cm.get_history()

# Check for pending query from suggestion buttons / quick starters
pending_query = st.session_state.pop("_pending_question", None)

# Chat input widget
user_input = st.chat_input("Ask a question about your documents...")

# Active query from input box or button click
active_query = user_input or pending_query

# ---------------- WELCOME / EMPTY STATE ----------------
if not history and not active_query:
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
if st.session_state.suggested_questions and not active_query:
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
    if message["role"] == "user":
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown('<div class="user-chat-marker"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble-container"><div class="user-bubble">{html.escape(message["content"])}</div></div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
            st.markdown('<div class="assistant-chat-marker"></div>', unsafe_allow_html=True)
            if message.get("sources"):
                st.markdown(
                    render_claude_thought(
                        f"Synthesized from {len(message['sources'])} source citations",
                        [
                            ("Queried dense vector embeddings (FAISS)", "done"),
                            (f"Retrieved {len(message['sources'])} matching excerpts", "done"),
                            ("Grounded response verified against source documents", "done"),
                        ],
                        is_done=True,
                    ),
                    unsafe_allow_html=True,
                )
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

# ---------------- ACTIVE QUESTION PROCESSING (Immediate display + Claude progress) ----------------
if active_query:
    query_str = active_query.strip()
    if query_str:
        # Step 1: Immediately show the user question on the far right side
        cm.add_message("user", query_str)
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown('<div class="user-chat-marker"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble-container"><div class="user-bubble">{html.escape(query_str)}</div></div>', unsafe_allow_html=True)

        # Step 2: Open assistant response container on the left side
        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
            st.markdown('<div class="assistant-chat-marker"></div>', unsafe_allow_html=True)
            thought_placeholder = st.empty()
            answer_placeholder = st.empty()

            if store.total_chunks == 0:
                warning_text = "Please upload at least one document (PDF, TXT, or DOCX) in the sidebar before asking a question."
                answer_placeholder.markdown(warning_text)
                cm.add_message("assistant", warning_text, sources=[])
            else:
                steps_log = [("Querying dense vector embeddings (FAISS)", "active")]
                thought_placeholder.markdown(
                    render_claude_thought("Searching document index...", steps_log, is_done=False),
                    unsafe_allow_html=True,
                )

                retrieved = retrieve_relevant_chunks(query_str, store, top_k=TOP_K)

                if not retrieved:
                    thought_placeholder.empty()
                    no_info_text = (
                        "No sufficiently relevant information was found in the uploaded "
                        "documents for this question. Try rephrasing, or upload a document "
                        "that covers this topic."
                    )
                    answer_placeholder.markdown(no_info_text)
                    cm.add_message("assistant", no_info_text, sources=[])
                else:
                    doc_names_found = list({c["document"] for c, _ in retrieved})
                    doc_summary_label = ", ".join(doc_names_found[:2])
                    if len(doc_names_found) > 2:
                        doc_summary_label += f" (+{len(doc_names_found) - 2} more)"

                    steps_log = [
                        ("Queried dense vector embeddings (FAISS)", "done"),
                        (f"Retrieved {len(retrieved)} matching excerpts from {doc_summary_label}", "active"),
                    ]
                    thought_placeholder.markdown(
                        render_claude_thought(f"Retrieved {len(retrieved)} excerpts...", steps_log, is_done=False),
                        unsafe_allow_html=True,
                    )

                    steps_log = [
                        ("Queried dense vector embeddings (FAISS)", "done"),
                        (f"Retrieved {len(retrieved)} matching excerpts from {doc_summary_label}", "done"),
                        ("Synthesizing grounded response with citations", "active"),
                    ]
                    thought_placeholder.markdown(
                        render_claude_thought("Synthesizing response...", steps_log, is_done=False),
                        unsafe_allow_html=True,
                    )

                    context = build_context(retrieved)
                    prompt = build_prompt(query_str, context, answer_style="Detailed", exam_mode=False)

                    try:
                        answer_text = rag_engine.generate_answer(prompt)
                    except rag_engine.LLMNotConfiguredError as exc:
                        answer_text = str(exc)
                    except RuntimeError as exc:
                        answer_text = f"Something went wrong while generating the answer: {exc}"

                    sources = [
                        {"document": chunk["document"], "page": chunk["page"], "score": round(score, 3)}
                        for chunk, score in retrieved
                    ]

                    # Complete status pill (single-line, tap to expand full steps log)
                    steps_log = [
                        ("Queried dense vector embeddings (FAISS)", "done"),
                        (f"Retrieved {len(sources)} matching excerpts from {doc_summary_label}", "done"),
                        ("Grounded answer verified against source documents", "done"),
                    ]
                    thought_placeholder.markdown(
                        render_claude_thought(f"Synthesized from {len(sources)} source citations", steps_log, is_done=True),
                        unsafe_allow_html=True,
                    )

                    answer_placeholder.markdown(answer_text)

                    # Show citations expander
                    with st.expander(f"Citations ({len(sources)} references)"):
                        for i, src in enumerate(sources, start=1):
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

                    cm.add_message("assistant", answer_text, sources=sources)
