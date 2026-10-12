"""
Generator for Evidence-Grounded RAG using local Ollama.

Calls Ollama REST API directly via stdlib urllib (no extra deps).
Enforces grounded generation with [Doc N] citation format.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

SYSTEM_PROMPT = """You are a precise, evidence-grounded research assistant for a book catalog.

RULES:
1. Answer the user question ONLY using the facts from the PROVIDED DOCUMENTS below.
2. Every factual statement MUST cite its source using the exact format [Doc N] (e.g. [Doc 1]).
3. If the provided documents DO NOT contain enough information to answer, state clearly: "I cannot find this information in the catalog." Do NOT use outside knowledge or hallucinate.
4. Keep the answer concise, accurate, and faithful to the documents."""


def generate_answer(
    question: str,
    context_text: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.1,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Generate an evidence-grounded answer from context.

    Returns:
        {
            "answer": str,
            "citations": List[str],  # e.g. ["Doc 1", "Doc 2"]
            "model": str,
            "has_answer": bool,
        }
    """
    if not context_text or context_text == "No relevant documents found.":
        return {
            "answer": "I cannot find this information in the catalog.",
            "citations": [],
            "model": model,
            "has_answer": False,
        }

    user_content = f"QUESTION: {question}\n\nPROVIDED DOCUMENTS:\n{context_text}\n\nANSWER:"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": temperature,
        },
        "stream": False,
    }

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer_text = (data.get("message") or {}).get("content", "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {
            "answer": f"Error contacting local LLM: {e}",
            "citations": [],
            "model": model,
            "has_answer": False,
        }

    # Extract citations: handles [Doc 1], [Doc 1, Doc 2], [Doc 1, 2], [Doc 1][Doc 2]
    # ponytail: parse both single and comma-separated Doc refs inside brackets
    doc_nums = set()
    for bracket_content in re.findall(r"\[([^\]]+)\]", answer_text):
        if re.search(r"\bDoc\b", bracket_content, re.IGNORECASE):
            for n in re.findall(r"\b([0-9]+)\b", bracket_content):
                doc_nums.add(int(n))
    doc_ids = sorted([f"Doc {n}" for n in doc_nums], key=lambda x: int(x.split()[1]))

    no_info_markers = [
        "cannot find",
        "could not find",
        "not mentioned",
        "no information",
        "does not contain",
    ]
    has_answer = not any(m in answer_text.lower() for m in no_info_markers) and bool(doc_ids)

    return {
        "answer": answer_text,
        "citations": doc_ids,
        "model": model,
        "has_answer": has_answer,
    }
