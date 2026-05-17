"""
tools.py
--------
The "hands" of the agent. Each function is a capability the LLM can invoke.

An agent without tools is just a chatbot. Tools let the model:
    - reach for fresh information (web_search, fetch_url)
    - do precise computation it cannot reliably do in its head (calculator)
    - cause side effects in the real world (omitted here for safety)

Each tool has TWO parts:
    1. A normal Python function that does the work.
    2. A JSON schema describing it to the LLM (see TOOL_SCHEMAS at the bottom).

OpenAI's "function calling" / "tool use" API uses the schema to decide
when and how to call the function. We then execute it and feed the
result back into the conversation.
"""

from __future__ import annotations

import math
import re
import json
from typing import Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Tool 1: Web search
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo (no API key required).

    Returns a compact JSON string with title, url, and snippet for each hit.
    We return JSON because the LLM parses it more reliably than free-form text.
    """
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return json.dumps({"error": f"search failed: {e}"})

    cleaned = [
        {
            "title": h.get("title", ""),
            "url": h.get("href") or h.get("url", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]
    return json.dumps(cleaned, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Fetch a URL and return clean text
# ---------------------------------------------------------------------------
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Download a page and strip it down to readable text.

    We cap output at `max_chars` so we don't blow up the model's context
    window with a 50-page article.
    """
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (LearningAgent/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR fetching {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Drop noisy elements before extracting text.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse runs of whitespace.
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[truncated]"
    return text


# ---------------------------------------------------------------------------
# Tool 3: Calculator
# ---------------------------------------------------------------------------
_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


def calculator(expression: str) -> str:
    """Evaluate a math expression safely.

    LLMs are notoriously bad at arithmetic with many digits. Outsourcing
    math to Python eliminates a whole class of hallucinations.
    """
    # Reject anything that looks like code (imports, attribute access, etc.).
    if re.search(r"[a-zA-Z_]\s*=", expression) or "__" in expression:
        return "ERROR: unsafe expression"
    try:
        # eval is OK here because we restrict the globals/locals dictionaries.
        result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Registry the agent uses to look up tools by name.
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "calculator": calculator,
}


# ---------------------------------------------------------------------------
# JSON schemas — this is what the LLM actually sees.
# Keep descriptions sharp: the model picks tools based on these.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for a query. Returns a JSON list of "
                "search results with title, url, and snippet. Use this to "
                "discover candidate sources before reading them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query, like a real user would type.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Download a URL and return its visible text content (truncated). "
                "Use after web_search to read a promising result in detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL of the page to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return.",
                        "default": 4000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a Python math expression and return the result. "
                "Use this for any non-trivial arithmetic or unit conversion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(1234 * 5.5) / 12'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]
