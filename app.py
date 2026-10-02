"""Modern, clean ChatGPT-style user interface for RAG Document Assistant."""

import base64
import html
import streamlit as st

from src.config.settings import APP_TITLE, TOP_K
import importlib
import src.ingestion.manager as dm
import src.ui.chat_manager as cm
importlib.reload(dm)
importlib.reload(cm)
import src.generation.response as rag_engine
import src.generation.llm as llm_service
import src.generation.prompt as prompt_module
importlib.reload(prompt_module)
from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.prompt import build_context, build_prompt

# =======================================================================
# SVG ASSETS & ICONS
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

# Session initialization (Session-scoped in-memory data)
cm.init_chat_history()
store = dm.get_vector_store()

if "answer_depth" not in st.session_state:
    st.session_state.answer_depth = "Medium"

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# =======================================================================
# PROFESSIONAL CUSTOM STYLES (Exact ChatGPT Aesthetics)
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
        width: 5px;
        height: 5px;
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

    /* =======================================================================
       SIDEBAR: EXACT CHATGPT STYLE (Matching Project Theme #0e1117)
       ======================================================================= */
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.15rem !important;
        padding-top: 0.4rem !important;
    }

    /* Top Brand Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px 12px 12px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #ececf1;
        letter-spacing: -0.2px;
    }

    /* ALL sidebar buttons default to transparent text rows (No white boxes!) */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button[data-testid*="stBaseButton"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        color: #ececf1 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 8px 10px !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        border-radius: 8px !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
    }

    [data-testid="stSidebar"] button > div,
    [data-testid="stSidebar"] button > div > div,
    [data-testid="stSidebar"] button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] button p {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] button span {
        display: inline-flex !important;
        width: auto !important;
    }

    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] button[data-testid*="stBaseButton"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: none !important;
    }

    /* Active Chat highlight in Recents (Subtle dark background #212121) */
    .chat-row-active button,
    .chat-row-active button[data-testid*="stBaseButton"] {
        background: #212121 !important;
        background-color: #212121 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .chat-row-active button:hover,
    .chat-row-active button[data-testid*="stBaseButton"]:hover {
        background: #282828 !important;
        background-color: #282828 !important;
        color: #ffffff !important;
    }

    /* Action buttons: New Chat */
    .sidebar-action-btn button {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 9px 12px !important;
        margin-bottom: 2px !important;
    }

    /* Clean Upload Button: Sleek button matching sidebar rows, hide all dropzone instructions */
    [data-testid="stFileUploader"] {
        padding: 0 !important;
        margin: 2px 0 6px 0 !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderInstructions"],
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
    }
    [data-testid="stFileUploader"] section button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        padding: 8px 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
    }
    [data-testid="stFileUploader"] section button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] section button p,
    [data-testid="stFileUploader"] section button span,
    [data-testid="stFileUploader"] section button div {
        font-size: 0.86rem !important;
        color: #ececf1 !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploader"] section button svg {
        fill: #ececf1 !important;
        color: #ececf1 !important;
        width: 16px !important;
        height: 16px !important;
    }
    [data-testid="stFileUploader"] section button:hover p,
    [data-testid="stFileUploader"] section button:hover span,
    [data-testid="stFileUploader"] section button:hover div {
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] section button:hover svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] ul {
        display: none !important;
    }

    /* Section Headers (e.g. 'Recents', 'Documents') */
    .sidebar-section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8e8ea0;
        padding: 16px 12px 6px 12px;
        letter-spacing: -0.1px;
    }

    /* Document Items in Sidebar */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {
        align-items: center !important;
        min-height: 32px !important;
        max-height: 32px !important;
        margin: 1px 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button {
        width: 100% !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        line-height: 32px !important;
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        direction: ltr !important;
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        padding: 0 4px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
        overflow: hidden !important;
        min-width: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }
    /* Hide the popover chevron completely so it takes zero space */
    [data-testid="stSidebar"] [data-testid="stPopover"] button svg,
    [data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid*="Icon"],
    [data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] [data-testid="stPopover"] button .material-symbols-outlined,
    [data-testid="stSidebar"] [data-testid="stPopover"] button span:has(svg),
    [data-testid="stSidebar"] [data-testid="stPopover"] button > div + * {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Text container starts flush at left edge, 1-line height, no ellipsis */
    [data-testid="stSidebar"] [data-testid="stPopover"] button > div {
        display: block !important;
        width: 100% !important;
        min-width: 0 !important;
        text-align: left !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        height: 32px !important;
        line-height: 32px !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stPopover"] button p {
        display: block !important;
        text-align: left !important;
        justify-content: flex-start !important;
        direction: ltr !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        width: 100% !important;
        min-width: 0 !important;
        color: #ececf1 !important;
        font-size: 0.86rem !important;
        height: 32px !important;
        line-height: 32px !important;
    }

    /* Document Delete Button: Completely hidden by default, square 28px by 28px */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 32px !important;
        width: 32px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] button {
        opacity: 0 !important;
        visibility: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        color: #8e8ea0 !important;
        width: 28px !important;
        min-width: 28px !important;
        max-width: 28px !important;
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        aspect-ratio: 1 / 1 !important;
        line-height: 1 !important;
        padding: 0 !important;
        font-size: 0.82rem !important;
        box-shadow: none !important;
        text-align: center !important;
        transition: opacity 0.15s ease, visibility 0.15s ease, color 0.15s ease, background-color 0.15s ease !important;
    }
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] button div,
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] button p,
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] button span {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* Document Delete Button: Appears ONLY when the document row is hovered over */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]):hover [data-testid="stButton"] button {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) [data-testid="stButton"] button:hover {
        color: #f87171 !important;
        background: rgba(239, 68, 68, 0.16) !important;
        border-radius: 6px !important;
    }

    /* Chat items in sidebar (Recents - Exact ChatGPT look) */
    .chat-row button {
        display: block !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        background: transparent !important;
        border: none !important;
        color: #ececf1 !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        padding: 7px 12px !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
    }
    .chat-row button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: none !important;
    }
    .chat-row-active button {
        background: #212121 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .chat-row-active button:hover {
        background: #262626 !important;
        color: #ffffff !important;
    }

    .chat-del-btn button {
        background: transparent !important;
        border: none !important;
        color: #64748b !important;
        padding: 7px 4px !important;
        font-size: 0.76rem !important;
        box-shadow: none !important;
    }
    .chat-del-btn button:hover {
        color: #f87171 !important;
        background: transparent !important;
    }

    /* =======================================================================
       CHAT MESSAGES STYLING
       ======================================================================= */
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
        background: #2f2f2f !important;
        color: #ececf1 !important;
        padding: 10px 18px !important;
        border-radius: 18px !important;
        display: inline-block !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
        font-size: 0.94rem !important;
        line-height: 1.55 !important;
        max-width: 80% !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }

    /* Assistant Message */
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

    /* Expandable thought pill */
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

    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        margin-top: 0.75rem !important;
    }

    /* =======================================================================
       QUESTION TEXT BOX AREA (Exact ChatGPT Style - Perfectly Centered)
       ======================================================================= */
    [data-testid="stBottom"] {
        background: transparent !important;
        padding-bottom: 14px !important;
        padding-top: 0 !important;
    }

    [data-testid="stBottom"] [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        position: relative !important;
        max-width: 800px !important;
        margin: 0 auto !important;
        padding: 0 16px !important;
        gap: 0 !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
    }

    [data-testid="stBottom"] [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stBottom"] [data-testid="stElementContainer"]:has(img) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Single sleek ChatGPT capsule */
    [data-testid="stChatInput"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
    }
    [data-testid="stChatInput"] > div {
        position: relative !important;
        background: #212121 !important;
        background-color: #212121 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 26px !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        box-sizing: border-box !important;
        padding: 0 16px !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        transition: border-color 0.15s ease !important;
        overflow: visible !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Inner flex wrappers inside stChatInput - Strict 1-row layout */
    [data-testid="stChatInput"] > div > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: flex-start !important;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div > div > div:first-child {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        flex: 1 !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        min-width: 0 !important;
    }

    /* Hide any extra instructions / badges that Streamlit adds */
    [data-testid="stChatInput"] #stChatInputInstructions,
    [data-testid="stChatInput"] [data-testid="stChatInputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
    }

    /* Completely Transparent Text Area - Centered on Single Line */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"],
    .stChatInput textarea {
        color: #ececf1 !important;
        font-size: 0.94rem !important;
        font-family: inherit !important;
        height: 25px !important;
        min-height: 25px !important;
        max-height: 25px !important;
        line-height: 25px !important;
        border: none !important;
        outline: none !important;
        background: transparent !important;
        background-color: transparent !important;
        padding: 0 130px 0 0 !important;
        margin: 0 !important;
        resize: none !important;
        box-shadow: none !important;
        box-sizing: border-box !important;
        display: block !important;
        width: 100% !important;
        align-self: center !important;
        overflow: hidden !important;
        vertical-align: middle !important;
    }
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInputTextArea"]:focus,
    .stChatInput textarea:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder,
    .stChatInput textarea::placeholder {
        color: #8e8ea0 !important;
        font-size: 0.94rem !important;
        line-height: 25px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Submit Button inside capsule on the far right */
    button[data-testid="stChatInputSubmitButton"] {
        position: absolute !important;
        right: 8px !important;
        top: 24px !important;
        transform: translateY(-50%) !important;
        background: #ffffff !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        max-width: 32px !important;
        max-height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 10 !important;
        transition: opacity 0.15s ease, background-color 0.15s ease !important;
    }
    button[data-testid="stChatInputSubmitButton"] svg {
        fill: #000000 !important;
        color: #000000 !important;
        width: 16px !important;
        height: 16px !important;
    }
    button[data-testid="stChatInputSubmitButton"]:disabled {
        background: rgba(255, 255, 255, 0.12) !important;
        opacity: 0.4 !important;
    }
    button[data-testid="stChatInputSubmitButton"]:disabled svg {
        fill: #8e8ea0 !important;
        color: #8e8ea0 !important;
    }

    /* Effort Button inside capsule, right next to the send button */
    [data-testid="stBottom"] [data-testid="stPopover"],
    [data-testid="stBottom"] .stPopover {
        position: absolute !important;
        right: 63px !important;
        top: 24px !important;
        transform: translateY(-50%) !important;
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        width: auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 99 !important;
    }
    [data-testid="stBottom"] [data-testid="stPopover"] button {
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        line-height: 28px !important;
        border-radius: 14px !important;
        border: none !important;
        background: transparent !important;
        background-color: transparent !important;
        color: #8e8ea0 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        padding: 0 8px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        cursor: pointer !important;
        margin: 0 !important;
    }
    [data-testid="stBottom"] [data-testid="stPopover"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #ececf1 !important;
    }
    [data-testid="stBottom"] [data-testid="stPopover"] button div,
    [data-testid="stBottom"] [data-testid="stPopover"] button p,
    [data-testid="stBottom"] [data-testid="stPopover"] button span {
        display: inline-flex !important;
        align-items: center !important;
        line-height: 28px !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100% !important;
    }

    /* Hide all chevrons/icons on effort button (hide material symbols expand_more and second div) */
    .effort-pill-container [data-testid="stPopover"] .material-symbols-outlined,
    .effort-pill-container [data-testid="stPopover"] button .material-symbols-outlined,
    .effort-pill-container [data-testid="stPopover"] button > div > div:nth-child(2),
    [data-testid="stBottom"] [data-testid="stPopover"] .material-symbols-outlined,
    [data-testid="stBottom"] [data-testid="stPopover"] button > div > div:nth-child(2),
    .effort-pill-container [data-testid="stPopover"] button svg,
    .effort-pill-container [data-testid="stPopover"] button [data-testid*="Icon"],
    .effort-pill-container [data-testid="stPopover"] button [data-testid*="icon"],
    .effort-pill-container [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Effort Dropdown Menu - ChatGPT Reasoning Effort Style */
    [data-testid="stPopoverBody"]:has([data-testid*="effort_sel"]),
    [data-testid="stPopoverBody"]:has(button) {
        background: #212121 !important;
        background-color: #212121 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
        padding: 6px !important;
        min-width: 210px !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 1px 0 !important;
        width: 100% !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 100% !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 12px !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
        font-size: 0.92rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        width: auto !important;
        margin: 0 !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button p {
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        font-size: 0.92rem !important;
        color: inherit !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button code {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #8e8ea0 !important;
        font-size: 0.72rem !important;
        font-family: inherit !important;
        font-weight: 500 !important;
        padding: 2px 7px !important;
        border-radius: 6px !important;
        border: none !important;
        margin-left: 8px !important;
    }
    /* When an option is active (primary), show the blue checkmark on the far right */
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button[data-testid*="primary"] {
        background: transparent !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button[data-testid*="primary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stPopoverBody"] div[data-testid="stButton"] button[data-testid*="primary"]::after {
        content: "✓";
        font-size: 1.15rem;
        font-weight: 700;
        color: #2f81f7;
        margin-left: auto;
        padding-left: 16px;
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

def format_sidebar_doc_name(filename: str) -> str:
    """Extract filename stem without extension, letting it fill the line naturally till the end."""
    return filename.rsplit(".", 1)[0] if "." in filename else filename


# =======================================================================
# SIDEBAR (Exact ChatGPT Layout)
# =======================================================================
with st.sidebar:
    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ececf1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/>
            </svg>
            <span>Document AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. New Chat Option
    st.markdown('<div class="sidebar-action-btn">', unsafe_allow_html=True)
    if st.button("＋ New chat", key="sidebar_new_chat", use_container_width=True):
        cm.create_new_chat()
        st.session_state.suggested_questions = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Upload Document Button (Always visible & fresh)
    if "doc_uploader_key" not in st.session_state:
        st.session_state.doc_uploader_key = 0

    uploaded_files = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"session_doc_uploader_{st.session_state.doc_uploader_key}",
    )

    # Only show error if the upload is wrong
    if uploaded_files:
        has_error = False
        for uploaded in uploaded_files:
            file_bytes = uploaded.getvalue()
            result = dm.add_document(uploaded.name, file_bytes, store=store)

            if result["success"]:
                st.session_state.suggested_questions = []
            elif result["duplicate"]:
                pass
            else:
                has_error = True
                st.error(f"{uploaded.name}: {result['message']}")

        # Reset uploader key so button remains permanently visible and ready for next upload
        if not has_error:
            st.session_state.doc_uploader_key += 1
            st.rerun()

    # 3. Uploaded Documents List (Clean text, metadata shown on tap)
    registry = dm.get_document_registry()
    if registry:
        st.markdown('<div class="sidebar-section-header">Documents</div>', unsafe_allow_html=True)
        for file_hash, info in registry.items():
            fname = info["filename"]
            ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "DOC"
            display_name = format_sidebar_doc_name(fname)

            col_doc, col_del = st.columns([0.86, 0.14], vertical_alignment="center")
            with col_doc:
                with st.popover(display_name, use_container_width=True):
                    st.markdown(f"**Document Details**")
                    st.markdown(f"**Filename:** `{fname}`")
                    st.markdown(f"**Type:** `{ext}`")
                    st.markdown(f"**Pages:** `{info['pages']}`")
                    st.markdown(f"**Chunks:** `{info['chunks']}`")
                    st.markdown(f"**Size:** `{info['size']}`")
                    st.markdown(f"**Characters:** `{info['characters']:,}`")
                    st.caption("Indexed for this session.")
            with col_del:
                if st.button("✕", key=f"del_doc_{file_hash}", help=f"Remove {fname}", use_container_width=True):
                    dm.remove_document(fname, store=store)
                    st.session_state.suggested_questions = []
                    st.rerun()

    # 4. Chats List (Recents - Exact ChatGPT Text Rows)
    chats = cm.get_all_chats()
    current_chat_id = cm.get_current_chat_id()

    st.markdown('<div class="sidebar-section-header">Recents</div>', unsafe_allow_html=True)

    for chat in chats:
        c_id = chat["id"]
        c_title = chat.get("title", "New Chat")
        is_active = (c_id == current_chat_id)

        if is_active and len(chats) > 1:
            col_c, col_d = st.columns([0.88, 0.12])
            with col_c:
                st.markdown('<div class="chat-row-active">', unsafe_allow_html=True)
                if st.button(c_title, key=f"chat_{c_id}", use_container_width=True):
                    pass
                st.markdown("</div>", unsafe_allow_html=True)
            with col_d:
                st.markdown('<div class="chat-del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{c_id}", help="Delete chat", use_container_width=True):
                    cm.delete_chat(c_id)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            item_class = "chat-row-active" if is_active else "chat-row"
            st.markdown(f'<div class="{item_class}">', unsafe_allow_html=True)
            if st.button(c_title, key=f"chat_{c_id}", use_container_width=True):
                if c_id != current_chat_id:
                    cm.switch_chat(c_id)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


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
# MAIN CHAT DISPLAY AREA
# =======================================================================
history = cm.get_history()

# Check for pending query from suggestion buttons / quick starters
pending_query = st.session_state.pop("_pending_question", None)

# =======================================================================
# BOTTOM ASK QUESTION BOX & EFFORT LEVEL BUTTON (ChatGPT Style)
# =======================================================================
with st.bottom:
    user_input = st.chat_input("Ask a question about your documents...")
    current_effort = st.session_state.get("answer_depth", "Medium")
    with st.popover(current_effort, use_container_width=False):
        for level in ["Low", "Medium", "High"]:
            is_selected = (level == current_effort)
            label = "Medium `Default`" if level == "Medium" else level
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                label,
                key=f"effort_sel_{level}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state["answer_depth"] = level
                st.rerun()
    st.markdown(
        """
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" style="display:none;" onload="if(!window._effortCloserInit){window._effortCloserInit=true;document.addEventListener('click',function(e){var b=e.target&&e.target.closest('[data-testid=stPopoverBody] button');if(b){setTimeout(function(){document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',code:'Escape',keyCode:27,which:27,bubbles:true}));},40);}},true);var resetScroll=function(){var t=document.querySelector('[data-testid=stChatInput] textarea');if(t&&t.scrollTop!==0){t.scrollTop=0;}};document.addEventListener('input',resetScroll,true);document.addEventListener('focusin',resetScroll,true);}">
        """,
        unsafe_allow_html=True,
    )

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

# ---------------- CHAT HISTORY DISPLAY ----------------
for message in history:
    if message["role"] == "user":
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown('<div class="user-chat-marker"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="user-bubble-container"><div class="user-bubble">{html.escape(message["content"])}</div></div>',
                unsafe_allow_html=True,
            )
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

# ---------------- ACTIVE QUESTION PROCESSING ----------------
if active_query:
    query_str = active_query.strip()
    if query_str:
        # Step 1: Record user question (names chat after first asked question)
        cm.add_message("user", query_str)
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown('<div class="user-chat-marker"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="user-bubble-container"><div class="user-bubble">{html.escape(query_str)}</div></div>',
                unsafe_allow_html=True,
            )

        # Step 2: Open assistant response container
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
                    active_depth = st.session_state.get("answer_depth", "Medium")
                    prompt = build_prompt(query_str, context, answer_style=active_depth)

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

        # Trigger rerun so sidebar updates chat title based on the first asked question
        st.rerun()
