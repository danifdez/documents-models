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


def _dynamic_idea_limit(
    units: int,
    cfg: Dict[str, Any],
    *,
    min_key: str,
    max_key: str,
    units_per_idea_key: str,
    min_default: int,
    max_default: int,
    units_per_idea_default: int,
) -> int:
    minimum = int(cfg.get(min_key, min_default))
    maximum = int(cfg.get(max_key, max_default))
    units_per_idea = int(cfg.get(units_per_idea_key, units_per_idea_default))
    if minimum <= 0 or maximum < minimum or units_per_idea <= 0:
        raise ValueError("dynamic summarization idea limits are invalid")
    estimated = math.ceil(max(1, units) / units_per_idea)
    return min(maximum, max(minimum, estimated))


def _ideas_response_format(max_ideas: int, max_idea_chars: int) -> Dict[str, Any]:
    if max_ideas <= 0 or max_idea_chars <= 0:
        raise ValueError("summarization idea schema limits are invalid")
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "ideas": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_ideas,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": max_idea_chars,
                    },
                },
            },
            "required": ["ideas"],
            "additionalProperties": False,
        },
    }


def _idea_information_units(ideas: List[str]) -> int:
    word_blocks = math.ceil(sum(len(idea.split()) for idea in ideas) / 25)
    return max(1, len(ideas), word_blocks)


def _parse_ideas(
    raw: str,
    *,
    max_ideas: int | None = None,
    max_idea_chars: int | None = None,
) -> List[str]:
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
    if max_ideas is not None and len(unique) > max_ideas:
        raise ValueError("summarization returned too many ideas")
    if max_idea_chars is not None and any(
        len(idea) > max_idea_chars for idea in unique
    ):
        raise ValueError("summarization returned an oversized idea")
    return unique


def _extract_ideas(text: str, target_language: str, cfg: Dict[str, Any]) -> List[str]:
    llm = get_llm_service(**get_llm_params("summarize-map"))
    cleaned = strip_dense_blobs(text)
    information_units = _source_information_units(cleaned)
    max_ideas = _dynamic_idea_limit(
        information_units,
        cfg,
        min_key="output_min_ideas",
        max_key="output_max_ideas",
        units_per_idea_key="information_units_per_idea",
        min_default=4,
        max_default=14,
        units_per_idea_default=3,
    )
    max_idea_chars = int(cfg.get("idea_max_chars", 240))
    max_tokens = _dynamic_max_tokens(
        max_ideas,
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_idea",
        min_default=384,
        max_default=1600,
        per_unit_default=96,
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
                max_ideas=max_ideas,
                max_idea_chars=max_idea_chars,
                safe_text=safe_text,
            ),
        },
    ]
    raw = llm.chat(
        messages,
        max_tokens=max_tokens,
        response_format=_ideas_response_format(max_ideas, max_idea_chars),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_ideas(
        raw,
        max_ideas=max_ideas,
        max_idea_chars=max_idea_chars,
    )


def _merge_idea_lists(
    partials: List[List[str]],
    target_language: str,
    cfg: Dict[str, Any],
) -> List[str]:
    ideas = [idea.strip() for partial in partials for idea in partial if idea.strip()]
    if not ideas:
        raise ValueError("summarize-reduce received no ideas")
    llm = get_llm_service(**get_llm_params("summarize-reduce"))
    max_ideas = len(ideas)
    max_idea_chars = int(cfg.get("idea_max_chars", 240))
    max_tokens = _dynamic_max_tokens(
        max_ideas,
        cfg,
        min_key="intermediate_min_tokens",
        max_key="intermediate_max_tokens",
        per_unit_key="intermediate_tokens_per_idea",
        min_default=512,
        max_default=3600,
        per_unit_default=80,
    )
    messages = [
        {"role": "system", "content": _MERGE_SYSTEM},
        {
            "role": "user",
            "content": _MERGE_USER.format(
                target_language=target_language,
                max_ideas=max_ideas,
                max_idea_chars=max_idea_chars,
                ideas=json.dumps(ideas, ensure_ascii=False),
            ),
        },
    ]
    raw = llm.chat(
        messages,
        max_tokens=max_tokens,
        response_format=_ideas_response_format(max_ideas, max_idea_chars),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_ideas(
        raw,
        max_ideas=max_ideas,
        max_idea_chars=max_idea_chars,
    )


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
