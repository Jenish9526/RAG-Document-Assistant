"""
chat_manager.py
================

Purpose
-------
Manages the in-session chat history (the back-and-forth between the
user and the assistant). Kept as its own small module so app.py's
UI code doesn't get mixed up with state-management logic.

Storage:
    Chat history lives in `st.session_state`, which Streamlit keeps
    alive for the duration of the browser session (it resets if the
    page is refreshed or the server restarts — this is expected;
    the DOCUMENT database, unlike chat history, is persisted to disk
    separately in vector_store.py).

Used by
-------
app.py -> rendering the chat window and handling "Clear Chat".
"""

import streamlit as st


def init_chat_history():
    """
    Function: init_chat_history()

    Purpose:
        Ensures `st.session_state.chat_history` exists before anything
        tries to read or write it.

    Used by:
        app.py -> once at the top of the script.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role: str, content: str, sources=None):
    """
    Function: add_message()

    Purpose:
        Appends one message (user or assistant) to the chat history.

    Input:
        role: "user" or "assistant".
        content: the message text.
        sources: optional list of source dicts (only for assistant messages).

    Used by:
        app.py -> after the user submits a question and after the
        assistant generates an answer.
    """
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })


def get_history():
    """Returns the full chat history list."""
    return st.session_state.chat_history


def clear_history():
    """
    Function: clear_history()

    Purpose:
        Empties the chat history WITHOUT touching the document
        vector database (documents remain searchable).

    Used by:
        app.py -> "Clear Chat" sidebar button.
    """
    st.session_state.chat_history = []
