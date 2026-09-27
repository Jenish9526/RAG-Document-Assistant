"""In-memory multi-chat state management for Streamlit session history."""

import time
import uuid
from typing import Dict, List, Optional
import streamlit as st


def init_chat_history():
    """Initialize multi-chat session state if not already present."""
    if "chats" not in st.session_state or not isinstance(st.session_state.chats, dict):
        st.session_state.chats = {}

    if not st.session_state.chats:
        initial_id = str(uuid.uuid4())[:8]
        st.session_state.chats[initial_id] = {
            "id": initial_id,
            "title": "New Chat",
            "messages": [],
            "created_at": time.time(),
        }
        st.session_state.current_chat_id = initial_id

    if (
        "current_chat_id" not in st.session_state
        or st.session_state.current_chat_id not in st.session_state.chats
    ):
        st.session_state.current_chat_id = next(iter(st.session_state.chats))


def create_new_chat() -> str:
    """Create a new chat session, set as active, and return its ID."""
    init_chat_history()
    new_id = str(uuid.uuid4())[:8]
    st.session_state.chats[new_id] = {
        "id": new_id,
        "title": "New Chat",
        "messages": [],
        "created_at": time.time(),
    }
    st.session_state.current_chat_id = new_id
    return new_id


def switch_chat(chat_id: str):
    """Switch active chat to the specified chat_id."""
    init_chat_history()
    if chat_id in st.session_state.chats:
        st.session_state.current_chat_id = chat_id


def get_current_chat_id() -> str:
    """Return the ID of the currently active chat."""
    init_chat_history()
    return st.session_state.current_chat_id


def get_current_chat() -> Optional[Dict]:
    """Return the currently active chat object."""
    init_chat_history()
    return st.session_state.chats.get(st.session_state.current_chat_id)


def get_all_chats() -> List[Dict]:
    """Return all chats sorted by creation time (most recent first)."""
    init_chat_history()
    chats = list(st.session_state.chats.values())
    chats.sort(key=lambda c: c.get("created_at", 0), reverse=True)
    return chats


def add_message(role: str, content: str, sources: Optional[List] = None):
    """Append message to active chat. Set chat title from first user question."""
    init_chat_history()
    curr_id = st.session_state.current_chat_id
    chat = st.session_state.chats.get(curr_id)
    if not chat:
        curr_id = create_new_chat()
        chat = st.session_state.chats[curr_id]

    # If first user question, name the chat after it
    if role == "user":
        has_user_msg = any(m.get("role") == "user" for m in chat["messages"])
        if not has_user_msg or chat.get("title") == "New Chat":
            clean_title = content.strip().split("\n")[0]
            if len(clean_title) > 34:
                clean_title = clean_title[:32].strip() + "..."
            chat["title"] = clean_title

    chat["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": time.time(),
    })


def get_history() -> List[Dict]:
    """Retrieve message history for the currently active chat."""
    init_chat_history()
    chat = st.session_state.chats.get(st.session_state.current_chat_id)
    return chat["messages"] if chat else []


def clear_history():
    """Clear message history of current chat and reset title to 'New Chat'."""
    init_chat_history()
    chat = st.session_state.chats.get(st.session_state.current_chat_id)
    if chat:
        chat["messages"] = []
        chat["title"] = "New Chat"


def delete_chat(chat_id: str):
    """Delete a chat and switch to another available chat or create a fresh one."""
    init_chat_history()
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if not st.session_state.chats:
        create_new_chat()
    elif st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))
