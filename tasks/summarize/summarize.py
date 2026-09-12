"""Idea extraction and global composition for durable summarization workflows."""

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
    if minimum < 0 or maximum <= 0 or maximum < minimum or units_per_idea <= 0:
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
                "material_idea_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max_ideas,
                },
                "ideas": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": max_ideas,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": max_idea_chars,
                    },
                },
            },
            "required": ["material_idea_count", "ideas"],
            "additionalProperties": False,
        },
    }


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
    declared_count = (
        payload.get("material_idea_count") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(ideas, list)
        or any(not isinstance(idea, str) or not idea.strip() for idea in ideas)
        or not isinstance(declared_count, int)
        or declared_count != len(ideas)
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
    if len(unique) != declared_count:
        raise ValueError("summarization returned duplicate ideas")
    if max_ideas is not None and len(unique) > max_ideas:
        raise ValueError("summarization returned too many ideas")
    if max_idea_chars is not None and any(
        len(idea) > max_idea_chars for idea in unique
    ):
        raise ValueError("summarization returned an oversized idea")
    if any(idea[-1] not in '.!?…"\'»)]' for idea in unique):
        raise ValueError("summarization returned an incomplete idea")
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
        min_default=0,
        max_default=12,
        units_per_idea_default=8,
    )
    target_idea_chars = int(cfg.get("idea_target_chars", 180))
    max_idea_chars = int(cfg.get("idea_max_chars", 240))
    if target_idea_chars <= 0 or target_idea_chars > max_idea_chars:
        raise ValueError("summarization idea length limits are invalid")
    max_tokens = _dynamic_max_tokens(
        max_ideas,
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_idea",
        min_default=384,
        max_default=900,
        per_unit_default=64,
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
                target_idea_chars=target_idea_chars,
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


def _combine_idea_lists(partials: List[List[str]]) -> List[str]:
    ideas = [
        idea.strip()
        for partial in partials
        for idea in partial
        if idea.strip()
    ]
    unique = []
    seen = set()
    for idea in ideas:
        normalized = re.sub(r"\s+", " ", idea).strip()
        key = normalized.casefold().rstrip(".!?")
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _dynamic_output_word_limit(idea_count: int, cfg: Dict[str, Any]) -> int:
    minimum = int(cfg.get("output_min_words", 60))
    maximum = int(cfg.get("output_max_words", 110))
    per_idea = int(cfg.get("output_words_per_idea", 7))
    if minimum <= 0 or maximum < minimum or per_idea <= 0:
        raise ValueError("summary word limits are invalid")
    return min(maximum, max(minimum, idea_count * per_idea))


def _dynamic_paragraph_count(idea_count: int, cfg: Dict[str, Any]) -> int:
    minimum = int(cfg.get("output_min_paragraphs", 1))
    maximum = int(cfg.get("output_max_paragraphs", 6))
    ideas_per_paragraph = int(cfg.get("input_ideas_per_paragraph", 9))
    if (
        idea_count <= 0
        or minimum <= 0
        or maximum < minimum
        or ideas_per_paragraph <= 0
    ):
        raise ValueError("summary paragraph limits are invalid")
    return min(maximum, max(minimum, math.ceil(idea_count / ideas_per_paragraph)))


def _summary_response_format(
    paragraph_count: int,
    max_sentences_per_paragraph: int,
) -> Dict[str, Any]:
    if paragraph_count <= 0 or max_sentences_per_paragraph <= 0:
        raise ValueError("summary schema limits are invalid")
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "minItems": paragraph_count,
                    "maxItems": paragraph_count,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_sentences_per_paragraph,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
            "required": ["paragraphs"],
            "additionalProperties": False,
        },
    }


def _parse_summary(raw: str, paragraph_count: int) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("summary returned invalid JSON") from error
    paragraphs = payload.get("paragraphs") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"paragraphs"}
        or not isinstance(paragraphs, list)
        or len(paragraphs) != paragraph_count
    ):
        raise ValueError("summary returned invalid paragraphs")
    texts: List[str] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, list) or not paragraph:
            raise ValueError("summary returned invalid paragraphs")
        sentences = []
        for text in paragraph:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("summary returned invalid paragraphs")
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized[-1] not in '.!?…"\'»)]':
                raise ValueError("summary returned an incomplete sentence")
            sentences.append(normalized)
        texts.append(" ".join(sentences))
    return "\n\n".join(texts)


def _write_summary(
    ideas: List[str],
    target_language: str,
    cfg: Dict[str, Any],
) -> str:
    max_ideas = int(cfg.get("input_max_ideas", 64))
    max_chars = int(cfg.get("input_max_chars", 20000))
    if (
        not ideas
        or len(ideas) > max_ideas
        or sum(len(idea) for idea in ideas) > max_chars
    ):
        raise ValueError("summary input exceeds configured limits")
    max_words = _dynamic_output_word_limit(len(ideas), cfg)
    paragraph_count = _dynamic_paragraph_count(len(ideas), cfg)
    max_paragraph_words = math.ceil(max_words / paragraph_count)
    max_paragraph_sentences = int(
        cfg.get("output_max_sentences_per_paragraph", 3)
    )
    if max_paragraph_sentences <= 0:
        raise ValueError("summary sentence limits are invalid")
    max_sentence_words = math.ceil(
        max_words / (paragraph_count * max_paragraph_sentences)
    )
    max_tokens = _dynamic_max_tokens(
        max_words,
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_word",
        min_default=384,
        max_default=3000,
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
                    idea_count=len(ideas),
                    paragraph_count=paragraph_count,
                    max_output_words=max_words,
                    max_paragraph_words=max_paragraph_words,
                    max_paragraph_sentences=max_paragraph_sentences,
                    max_sentence_words=max_sentence_words,
                    ideas=json.dumps(
                        ideas,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            },
        ],
        max_tokens=max_tokens,
        response_format=_summary_response_format(
            paragraph_count,
            max_paragraph_sentences,
        ),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    return _parse_summary(raw, paragraph_count)


def _combine_summary_lists(partials: List[List[str]]) -> List[str]:
    summaries = [
        re.sub(r"\s+", " ", summary).strip()
        for partial in partials
        for summary in partial
        if summary.strip()
    ]
    if not summaries:
        raise ValueError("summarize-finalize received no summary sections")
    return summaries
