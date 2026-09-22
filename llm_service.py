"""
llm_service.py
===============

Purpose
-------
The ONLY file in the project that talks to the external LLM API.
Every other file calls `generate_response(prompt)` and never needs to
know which provider is behind it. This means you can switch from Groq
to OpenAI, Together.ai, or OpenRouter by editing ONLY your `.env` file
— no code changes required anywhere else in the project.

Design note:
    This is an example of the Adapter pattern in modular design: the
    rest of the RAG pipeline (rag_engine.py) depends only on the
    function signature `generate_response(prompt) -> str`, not on any
    specific vendor's SDK.

Imported libraries
-------------------
- requests      : makes the HTTP call to the LLM's REST API.
- os / dotenv   : loads the API key from the .env file (never hard-coded).

Used by
-------
rag_engine.py -> generate_answer()
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

from config import LLM_BASE_URL_DEFAULT, LLM_MODEL_DEFAULT, LLM_MAX_TOKENS, LLM_TEMPERATURE

# Load variables from a local .env file (if present) into the process
# environment. This never overwrites variables already set by the OS.
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", LLM_BASE_URL_DEFAULT)
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL_DEFAULT)


def get_llm_config():
    """Dynamically loads configuration from .env so modifications take effect without restart."""
    if "LLM_API_KEY" in globals() and globals()["LLM_API_KEY"] == "":
        api_key = ""
    else:
        load_dotenv(override=True)
        api_key = os.getenv("LLM_API_KEY", globals().get("LLM_API_KEY", ""))

    base_url = os.getenv("LLM_BASE_URL", globals().get("LLM_BASE_URL", LLM_BASE_URL_DEFAULT))
    model = os.getenv("LLM_MODEL", globals().get("LLM_MODEL", LLM_MODEL_DEFAULT))
    return api_key, base_url, model


class LLMNotConfiguredError(Exception):
    """Raised when no API key has been set up yet."""
    pass


def generate_response(prompt: str) -> str:
    """
    Function: generate_response()

    Purpose:
        Sends a prompt to the configured LLM (via an OpenAI-compatible
        Chat Completions endpoint) and returns the generated text.

    Input:
        prompt: the full prompt string (built by rag_engine.build_prompt()),
                already containing the retrieved document context + question.

    Output:
        The LLM's generated answer as a plain string.

    Raises:
        LLMNotConfiguredError: if LLM_API_KEY is missing from .env.
        RuntimeError: if the API call fails for any other reason.

    Used by:
        rag_engine.py -> generate_answer()
    """
    api_key, base_url, model = get_llm_config()

    if not api_key:
        raise LLMNotConfiguredError(
            "LLM API key is not configured. Please set LLM_API_KEY in your .env file."
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 429 and attempt < 2:
                wait_sec = 4.0 * (attempt + 1)
                try:
                    err_msg = response.json().get("error", {}).get("message", "")
                    match = re.search(r"try again in ([\d\.]+)s", err_msg, re.IGNORECASE)
                    if match:
                        wait_sec = min(float(match.group(1)) + 0.5, 20.0)
                except Exception:
                    pass
                time.sleep(wait_sec)
                continue

            if not response.ok:
                detail = ""
                try:
                    err_data = response.json()
                    detail = err_data.get("error", {}).get("message", "")
                except Exception:
                    detail = response.text
                msg = f"HTTP {response.status_code}: {detail}" if detail else str(response.status_code)
                raise RuntimeError(f"LLM request failed ({msg})")

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"LLM network request failed: {exc}") from exc
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response format: {exc}") from exc

    if last_error:
        raise RuntimeError(f"LLM request failed: {last_error}")


def is_configured() -> bool:
    """
    Function: is_configured()

    Purpose:
        Lets app.py check (before running a query) whether an API key
        has been set, so it can show a friendly warning instead of
        crashing.

    Used by:
        app.py -> chat input handling.
    """
    api_key, _, _ = get_llm_config()
    return bool(api_key)

