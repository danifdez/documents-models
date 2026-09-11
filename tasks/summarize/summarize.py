"""Self-contained map and reduce steps for durable summarization workflows."""

import json
import math
import re
from typing import Any, Dict, List

from lib.llm.config import get_llm_params
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service

_SUMMARY_SYSTEM = get_prompt("summarize", "prompts/summary_system.md").strip()
_SUMMARY_USER = get_prompt("summarize", "prompts/summary_user.md")
_MERGE_SYSTEM = get_prompt("summarize", "prompts/merge_system.md").strip()
_MERGE_USER = get_prompt("summarize", "prompts/merge_user.md")
_FINAL_SYSTEM = get_prompt("summarize", "prompts/final_system.md").strip()
_FINAL_USER = get_prompt("summarize", "prompts/final_user.md")

_IDEAS_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["ideas"],
        "additionalProperties": False,
    },
}


def _target_language(payload: Dict[str, Any]) -> str:
    return payload.get("targetLanguage") or "en"


def _dynamic_max_tokens(
    units: int,
    cfg: Dict[str, Any],
    *,
    min_key: str,
    max_key: str,
    per_unit_key: str,
    min_default: int,
    max_default: int,
    per_unit_default: int,
) -> int:
    minimum = int(cfg.get(min_key, min_default))
    maximum = int(cfg.get(max_key, max_default))
    per_unit = int(cfg.get(per_unit_key, per_unit_default))
    if minimum <= 0 or maximum < minimum or per_unit <= 0:
        raise ValueError("dynamic summarization token limits are invalid")
    return min(maximum, max(minimum, 96 + max(1, units) * per_unit))


def _source_information_units(text: str) -> int:
    paragraphs = len([part for part in re.split(r"\n\s*\n", text) if part.strip()])
    clauses = len(re.findall(r"[.!?;:]+(?:\s|$)", text))
    word_blocks = math.ceil(len(text.split()) / 80)
    return max(1, paragraphs, clauses, word_blocks)


def _idea_information_units(ideas: List[str]) -> int:
    word_blocks = math.ceil(sum(len(idea.split()) for idea in ideas) / 25)
    return max(1, len(ideas), word_blocks)


def _parse_ideas(raw: str) -> List[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("summarization returned invalid idea JSON") from error
    ideas = payload.get("ideas") if isinstance(payload, dict) else None
    if (
        not isinstance(ideas, list)
        or not ideas
        or any(not isinstance(idea, str) or not idea.strip() for idea in ideas)
    ):
        raise ValueError("summarization returned invalid ideas")
    unique = []
    seen = set()
    for idea in ideas:
        normalized = idea.strip()
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _extract_ideas(text: str, target_language: str, cfg: Dict[str, Any]) -> List[str]:
    llm = get_llm_service(**get_llm_params("summarize-map"))
    cleaned = strip_dense_blobs(text)
    max_tokens = _dynamic_max_tokens(
        _source_information_units(cleaned),
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="tokens_per_information_unit",
        min_default=384,
        max_default=1200,
        per_unit_default=32,
    )
    safe_text = truncate_for_llm(
        cleaned,
        {**cfg, "dynamic_output_tokens": max_tokens},
        tokens_key="dynamic_output_tokens",
        default_tokens=max_tokens,
    )
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": _SUMMARY_USER.format(
                target_language=target_language,
                safe_text=safe_text,
            ),
        },
    ]
    raw = llm.chat(
        messages,
        max_tokens=max_tokens,
        response_format=_IDEAS_RESPONSE_FORMAT,
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_ideas(raw)


def _merge_idea_lists(
    partials: List[List[str]],
    target_language: str,
    cfg: Dict[str, Any],
) -> List[str]:
    ideas = [idea.strip() for partial in partials for idea in partial if idea.strip()]
    if not ideas:
        raise ValueError("summarize-reduce received no ideas")
    llm = get_llm_service(**get_llm_params("summarize-reduce"))
    max_tokens = _dynamic_max_tokens(
        _idea_information_units(ideas),
        cfg,
        min_key="intermediate_min_tokens",
        max_key="intermediate_max_tokens",
        per_unit_key="intermediate_tokens_per_idea",
        min_default=384,
        max_default=1600,
        per_unit_default=36,
    )
    messages = [
        {"role": "system", "content": _MERGE_SYSTEM},
        {
            "role": "user",
            "content": _MERGE_USER.format(
                target_language=target_language,
                ideas=json.dumps(ideas, ensure_ascii=False),
            ),
        },
    ]
    raw = llm.chat(
        messages,
        max_tokens=max_tokens,
        response_format=_IDEAS_RESPONSE_FORMAT,
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_ideas(raw)


def _write_summary(
    ideas: List[str],
    target_language: str,
    cfg: Dict[str, Any],
) -> str:
    if not ideas:
        raise ValueError("summarize-reduce received no ideas")
    llm = get_llm_service(**get_llm_params("summarize-reduce"))
    max_tokens = _dynamic_max_tokens(
        _idea_information_units(ideas),
        cfg,
        min_key="final_min_tokens",
        max_key="final_max_tokens",
        per_unit_key="final_tokens_per_idea",
        min_default=600,
        max_default=2200,
        per_unit_default=40,
    )
    messages = [
        {"role": "system", "content": _FINAL_SYSTEM},
        {
            "role": "user",
            "content": _FINAL_USER.format(
                target_language=target_language,
                ideas=json.dumps(ideas, ensure_ascii=False),
            ),
        },
    ]
    return llm.chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    ).strip()
