"""Substring validation tool for agents."""

from typing import Any, Dict, List

from agent.tools.base import tool
from agent.types import ToolContext


def _normalize_texts(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    return str(value or "")


@tool(
    name="substring_check",
    description=(
        "Check whether each candidate string appears literally (case-insensitive) "
        "in a source text or list of texts. Returns per-candidate verdicts."
    ),
    args_schema={
        "candidates": "list of strings to verify",
        "source": "string OR list of strings (defaults to the agent's input payload 'texts')",
    },
)
def substring_check(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    candidates = args.get("candidates")
    if not isinstance(candidates, list):
        return {"error": "'candidates' must be a list of strings"}

    source = args.get("source")
    if source is None:
        source = ctx.payload.get("texts") or ctx.payload.get("text") or ""
    haystack = _normalize_texts(source).lower()

    verdicts: List[Dict[str, Any]] = []
    for c in candidates:
        s = str(c)
        verdicts.append({
            "candidate": s,
            "found": s.lower() in haystack and len(s.strip()) >= 2,
        })

    valid = [v["candidate"] for v in verdicts if v["found"]]
    invalid = [v["candidate"] for v in verdicts if not v["found"]]
    return {
        "verdicts": verdicts,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid": valid,
        "invalid": invalid,
    }
