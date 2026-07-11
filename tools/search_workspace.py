"""search_workspace: lexical search across the user's workspace."""

import json
import logging
import re
from typing import Any, Dict, List

import urllib.error
import urllib.request

from agents.tool_base import Tool, ToolContext, register
from common.chat.http import BACKEND_URL

logger = logging.getLogger(__name__)

# Stopwords that should never be sent to the per-token fallback search —
# they would either return everything or just add noise. Bilingual on purpose
# because the assistant accepts both languages.
_SEARCH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en",
    "y", "o", "para", "por", "con", "sin", "sobre", "que", "qué", "como",
    "cómo", "es", "ha", "han", "tengo", "tiene", "tienen", "este", "esta",
    "estos", "estas", "mi", "mis", "tu", "tus", "su", "sus", "lo", "le", "se",
    "ya", "muy", "más", "menos", "a", "al", "ni", "no", "si", "sí", "qué",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "but", "is", "are", "was", "were", "be", "have", "has", "had",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "where", "when", "why", "how",
}


def _call_search(term: str) -> List[Dict[str, Any]]:
    """One call to backend POST /search. Returns raw items list (possibly empty)."""
    if not term:
        return []
    body = json.dumps({"term": term}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/search", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8")) or []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("assistant-chat: search term=%r failed: %s", term, e)
        return []


def _post_backend_search(query: str) -> Dict[str, Any]:
    """Hit backend /search with progressively wider queries until we get hits
    or run out of useful terms. The model often writes natural-language queries
    like "documents about research" — the backend does literal ILIKE matching,
    so the full phrase rarely matches but individual content words do. We:

    1. Try the full phrase first (preserves precision if it actually matches).
    2. If empty, retry each non-stopword token, longest first (rarest words
       tend to be most distinctive).
    3. Merge hits dedup'd by (collection, id), trim to 10."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "results": []}

    items = _call_search(query)

    if not items:
        tokens = [
            t for t in re.split(r"[\s,.;:!?¿¡()\"']+", query.lower())
            if t and t not in _SEARCH_STOPWORDS and len(t) >= 3
        ]
        # Longest first — they're typically the most specific.
        tokens.sort(key=len, reverse=True)
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for tok in tokens[:4]:
            for it in _call_search(tok):
                key = (it.get("collection"), it.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)
        items = merged

    # Compact shape for the model — drop scoring noise and highlight HTML.
    trimmed = []
    for it in items[:10]:
        trimmed.append({
            "collection": it.get("collection"),
            "id": it.get("id"),
            "name": it.get("name"),
        })
    return {"query": query, "results": trimmed}


def _execute(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    return _post_backend_search(str(args.get("query") or "").strip())


def _summarize(result: Dict[str, Any]):
    hits = len(result.get("results") or [])
    return f"{hits} results", None


register(Tool(
    schema={
        "type": "function",
        "function": {
            "name": "search_workspace",
            "description": (
                "Search the user's workspace (notes, tasks, calendar events, "
                "indexed resources) and return a compact list of hits "
                "(collection, id, name). Use it for keyword or title lookups, "
                "then read a hit's full text with get_resource_content.\n\n"
                "Pairs well with: get_resource_content (read a hit's full "
                "text by id), list_notes / list_tasks / list_projects (when "
                "you want an enumeration rather than a search)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms in natural language.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    execute=_execute,
    summarize=_summarize,
))
