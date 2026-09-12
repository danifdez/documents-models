"""Hierarchical summarization helpers for durable map-compose workflows."""

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
_COMPLETE_ENDINGS = '.!?…"\'»)]'


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


def _dynamic_section_word_limit(units: int, cfg: Dict[str, Any]) -> int:
    floor = int(cfg.get("output_word_limit_floor", 80))
    ceiling = int(cfg.get("output_word_limit_ceiling", 220))
    per_unit = int(cfg.get("output_words_per_information_unit", 3))
    if floor <= 0 or ceiling < floor or per_unit <= 0:
        raise ValueError("section summary word limits are invalid")
    return min(ceiling, max(floor, 32 + max(1, units) * per_unit))


def _section_response_format() -> Dict[str, Any]:
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    }


def _normalize_complete_text(value: Any, error_message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error_message)
    normalized = re.sub(r"\s+", " ", value).strip()
    if normalized[-1] not in _COMPLETE_ENDINGS:
        raise ValueError(error_message)
    return normalized


def _parse_section_summary(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("summarization returned invalid section JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("summarization returned an invalid section summary")
    return _normalize_complete_text(
        payload["summary"],
        "summarization returned an invalid section summary",
    )


def _summarize_chunk(text: str, target_language: str, cfg: Dict[str, Any]) -> str:
    cleaned = strip_dense_blobs(text).strip()
    if not cleaned:
        raise ValueError("summarization received no usable text")
    information_units = _source_information_units(cleaned)
    max_words = _dynamic_section_word_limit(information_units, cfg)
    max_tokens = _dynamic_max_tokens(
        max_words,
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_word",
        min_default=384,
        max_default=1050,
        per_unit_default=4,
    )
    safe_text = truncate_for_llm(
        cleaned,
        {**cfg, "dynamic_output_tokens": max_tokens},
        tokens_key="dynamic_output_tokens",
        default_tokens=max_tokens,
    )
    llm = get_llm_service(**get_llm_params("summarize-map"))
    raw = llm.chat(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": _SUMMARY_USER.format(
                    target_language=target_language,
                    max_output_words=max_words,
                    safe_text=safe_text,
                ),
            },
        ],
        max_tokens=max_tokens,
        response_format=_section_response_format(),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_section_summary(raw)


def _combine_section_summaries(partials: List[Any]) -> List[str]:
    summaries: List[str] = []
    for partial in partials:
        values = partial if isinstance(partial, list) else [partial]
        for value in values:
            summaries.append(
                _normalize_complete_text(
                    value,
                    "summarization received an invalid section summary",
                )
            )
    return summaries


def _dynamic_output_word_limit(summaries: List[str], cfg: Dict[str, Any]) -> int:
    floor = int(cfg.get("output_word_limit_floor", 80))
    ceiling = int(cfg.get("output_word_limit_ceiling", 400))
    input_ratio = float(cfg.get("output_input_word_ratio", 0.55))
    if floor <= 0 or ceiling < floor or not 0 < input_ratio <= 1:
        raise ValueError("global summary word limits are invalid")
    input_words = sum(len(summary.split()) for summary in summaries)
    return min(ceiling, max(floor, math.ceil(input_words * input_ratio)))


def _summary_response_format(
    min_paragraphs: int,
    max_paragraphs: int,
) -> Dict[str, Any]:
    if min_paragraphs <= 0 or max_paragraphs < min_paragraphs:
        raise ValueError("summary paragraph limits are invalid")
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "minItems": min_paragraphs,
                    "maxItems": max_paragraphs,
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": ["paragraphs"],
            "additionalProperties": False,
        },
    }


def _parse_summary(raw: str, min_paragraphs: int, max_paragraphs: int) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("summary returned invalid JSON") from error
    paragraphs = payload.get("paragraphs") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"paragraphs"}
        or not isinstance(paragraphs, list)
        or not min_paragraphs <= len(paragraphs) <= max_paragraphs
    ):
        raise ValueError("summary returned invalid paragraphs")
    return "\n\n".join(
        _normalize_complete_text(
            paragraph,
            "summary returned invalid paragraphs",
        )
        for paragraph in paragraphs
    )


def _write_summary(
    summaries: List[str],
    target_language: str,
    cfg: Dict[str, Any],
) -> str:
    max_sections = int(cfg.get("input_max_sections", 16))
    max_chars = int(cfg.get("input_max_chars", 20000))
    if (
        not summaries
        or len(summaries) > max_sections
        or sum(len(summary) for summary in summaries) > max_chars
    ):
        raise ValueError("summary input exceeds configured limits")
    max_words = _dynamic_output_word_limit(summaries, cfg)
    min_paragraphs = int(cfg.get("output_min_paragraphs", 1))
    max_paragraphs = int(cfg.get("output_max_paragraphs", 4))
    if min_paragraphs <= 0 or max_paragraphs < min_paragraphs:
        raise ValueError("summary paragraph limits are invalid")
    max_tokens = _dynamic_max_tokens(
        max_words,
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_word",
        min_default=512,
        max_default=1600,
        per_unit_default=4,
    )
    llm = get_llm_service(**get_llm_params("summarize-compose"))
    raw = llm.chat(
        [
            {"role": "system", "content": _MERGE_SYSTEM},
            {
                "role": "user",
                "content": _MERGE_USER.format(
                    target_language=target_language,
                    section_count=len(summaries),
                    max_output_words=max_words,
                    min_paragraphs=min_paragraphs,
                    max_paragraphs=max_paragraphs,
                    summaries=json.dumps(
                        summaries,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            },
        ],
        max_tokens=max_tokens,
        response_format=_summary_response_format(min_paragraphs, max_paragraphs),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_summary(raw, min_paragraphs, max_paragraphs)


def _combine_summary_lists(partials: List[List[str]]) -> List[str]:
    summaries = _combine_section_summaries(partials)
    if not summaries:
        raise ValueError("summarize-finalize received no summary sections")
    return summaries
