"""LLM communication service with OpenAI-compatible Chat Completions API and backoff retry."""

import os
import re
import time
import requests
from dotenv import load_dotenv

from src.config.settings import (
    LLM_BASE_URL_DEFAULT,
    LLM_MODEL_DEFAULT,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", LLM_BASE_URL_DEFAULT)
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL_DEFAULT)


def get_llm_config():
    """Dynamically load LLM credentials and endpoint configuration from environment."""
    if "LLM_API_KEY" in globals() and globals()["LLM_API_KEY"] == "":
        api_key = ""
    else:
        load_dotenv(override=True)
        api_key = os.getenv("LLM_API_KEY", globals().get("LLM_API_KEY", ""))

    base_url = os.getenv("LLM_BASE_URL", globals().get("LLM_BASE_URL", LLM_BASE_URL_DEFAULT))
    model = os.getenv("LLM_MODEL", globals().get("LLM_MODEL", LLM_MODEL_DEFAULT))
    return api_key, base_url, model


class LLMNotConfiguredError(Exception):
    """Raised when LLM_API_KEY is unset or empty."""
    pass


def generate_response(prompt: str) -> str:
    """Send prompt to the configured LLM endpoint and return the text response with retry backoff."""
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
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                wait_sec = 3.0 * (attempt + 1)
                try:
                    err_json = response.json()
                    err_msg = ""
                    if isinstance(err_json, dict):
                        err_msg = err_json.get("error", {}).get("message", "")
                    elif isinstance(err_json, list) and err_json and isinstance(err_json[0], dict):
                        err_msg = err_json[0].get("error", {}).get("message", "")
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
                    if isinstance(err_data, dict):
                        detail = err_data.get("error", {}).get("message", "")
                    elif isinstance(err_data, list) and err_data and isinstance(err_data[0], dict):
                        detail = err_data[0].get("error", {}).get("message", "")
                    else:
                        detail = str(err_data)
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
    """Return True if an LLM API key is present in configuration."""
    api_key, _, _ = get_llm_config()
    return bool(api_key)
