"""In-memory chat state management for Streamlit session history."""

import streamlit as st


def init_chat_history():
    """Initialize chat_history list in Streamlit session state if not present."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role: str, content: str, sources=None):
    """Append a user or assistant message with optional source metadata to chat history."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })


def get_history():
    """Retrieve full chat history from session state."""
    return st.session_state.chat_history


def clear_history():
    """Clear chat message history from session state."""
    st.session_state.chat_history = []

