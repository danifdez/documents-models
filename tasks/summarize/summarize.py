"""Self-contained map and reduce steps for durable summarization workflows."""

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List

from lib.llm.config import get_llm_params
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service

_SUMMARY_SYSTEM = get_prompt("summarize", "prompts/summary_system.md").strip()
_SUMMARY_USER = get_prompt("summarize", "prompts/summary_user.md")
_FINAL_SYSTEM = get_prompt("summarize", "prompts/final_system.md").strip()
_FINAL_USER = get_prompt("summarize", "prompts/final_user.md")

logger = logging.getLogger(__name__)

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
        units_per_idea_default=4,
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


def _combine_idea_lists(partials: List[List[str]]) -> List[str]:
    ideas = [idea.strip() for partial in partials for idea in partial if idea.strip()]
    if not ideas:
        raise ValueError("summarize-reduce received no ideas")
    unique = []
    seen = set()
    for idea in ideas:
        normalized = re.sub(r"\s+", " ", idea).strip()
        key = normalized.casefold().rstrip(".!?")
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _write_summary(
    ideas: List[str],
    cfg: Dict[str, Any],
) -> str:
    if not ideas:
        raise ValueError("summarize-reduce received no ideas")
    ideas_per_paragraph = int(cfg.get("final_ideas_per_paragraph", 4))
    if ideas_per_paragraph <= 0:
        raise ValueError("final summary paragraph size is invalid")
    sentences = []
    for idea in ideas:
        sentence = re.sub(r"\s+", " ", idea).strip()
        sentences.append(
            sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        )
    return "\n\n".join(
        " ".join(sentences[index:index + ideas_per_paragraph])
        for index in range(0, len(sentences), ideas_per_paragraph)
    )


def _idea_batches(ideas: List[str], cfg: Dict[str, Any]) -> List[List[str]]:
    max_ideas = int(cfg.get("input_max_ideas", 24))
    max_chars = int(cfg.get("input_max_chars", 6000))
    if max_ideas <= 0 or max_chars <= 0:
        raise ValueError("final composition input limits are invalid")

    batches: List[List[str]] = []
    current: List[str] = []
    current_chars = 0
    for idea in ideas:
        normalized = re.sub(r"\s+", " ", idea).strip()
        if not normalized:
            continue
        if len(normalized) > max_chars:
            raise ValueError("final composition received an oversized idea")
        if current and (
            len(current) >= max_ideas
            or current_chars + len(normalized) > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(normalized)
        current_chars += len(normalized)
    if current:
        batches.append(current)
    if not batches:
        raise ValueError("final composition received no ideas")
    return batches


def _composition_response_format(
    idea_ids: List[str],
    max_paragraph_chars: int,
) -> Dict[str, Any]:
    if not idea_ids or max_paragraph_chars <= 0:
        raise ValueError("final composition schema limits are invalid")
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": len(idea_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "idea_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": len(idea_ids),
                                "items": {
                                    "type": "string",
                                    "enum": idea_ids,
                                },
                            },
                        },
                        "required": ["text", "idea_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["paragraphs"],
            "additionalProperties": False,
        },
    }


def _parse_composition(
    raw: str,
    idea_ids: List[str],
    *,
    max_paragraph_chars: int,
    max_output_chars: int,
) -> List[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("final composition returned invalid JSON") from error
    paragraphs = payload.get("paragraphs") if isinstance(payload, dict) else None
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("final composition returned invalid paragraphs")

    texts: List[str] = []
    covered_ids: List[str] = []
    expected_ids = set(idea_ids)
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("final composition returned invalid paragraphs")
        text = paragraph.get("text")
        paragraph_ids = paragraph.get("idea_ids")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text.strip()) > max_paragraph_chars
            or not isinstance(paragraph_ids, list)
            or not paragraph_ids
            or any(
                not isinstance(idea_id, str) or idea_id not in expected_ids
                for idea_id in paragraph_ids
            )
        ):
            raise ValueError("final composition returned invalid paragraphs")
        texts.append(re.sub(r"\s+", " ", text).strip())
        covered_ids.extend(paragraph_ids)

    counts = Counter(covered_ids)
    if set(counts) != expected_ids or any(count != 1 for count in counts.values()):
        raise ValueError("final composition did not cover every idea exactly once")
    if sum(len(text) for text in texts) > max_output_chars:
        raise ValueError("final composition expanded beyond its source inventory")
    return texts


def _compose_batch(
    ideas: List[str],
    target_language: str,
    cfg: Dict[str, Any],
    llm,
) -> str:
    idea_ids = [f"idea-{index}" for index in range(1, len(ideas) + 1)]
    max_paragraph_chars = int(cfg.get("paragraph_max_chars", 2400))
    max_char_ratio = float(cfg.get("output_max_char_ratio", 1.1))
    if max_char_ratio <= 0:
        raise ValueError("final composition output ratio is invalid")
    max_output_chars = max(
        80,
        int(sum(len(idea) for idea in ideas) * max_char_ratio),
    )
    max_tokens = _dynamic_max_tokens(
        len(ideas),
        cfg,
        min_key="output_min_tokens",
        max_key="output_max_tokens",
        per_unit_key="output_tokens_per_idea",
        min_default=256,
        max_default=2000,
        per_unit_default=64,
    )
    inventory = [
        {"id": idea_id, "text": idea}
        for idea_id, idea in zip(idea_ids, ideas)
    ]
    raw = llm.chat(
        [
            {"role": "system", "content": _FINAL_SYSTEM},
            {
                "role": "user",
                "content": _FINAL_USER.format(
                    target_language=target_language,
                    max_paragraph_chars=max_paragraph_chars,
                    max_output_chars=max_output_chars,
                    ideas=json.dumps(
                        inventory,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            },
        ],
        max_tokens=max_tokens,
        response_format=_composition_response_format(
            idea_ids,
            max_paragraph_chars,
        ),
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    )
    paragraphs = _parse_composition(
        raw,
        idea_ids,
        max_paragraph_chars=max_paragraph_chars,
        max_output_chars=max_output_chars,
    )
    return "\n\n".join(paragraphs)


def _compose_summary(
    ideas: List[str],
    target_language: str,
    cfg: Dict[str, Any],
) -> str:
    llm = get_llm_service(**get_llm_params("summarize-compose"))
    summaries = []
    for batch in _idea_batches(ideas, cfg):
        try:
            summaries.append(_compose_batch(batch, target_language, cfg, llm))
        except ValueError as error:
            logger.warning(
                "Final composition failed its coverage contract; preserving the "
                "complete batch without rewriting: %s",
                error,
            )
            summaries.append(_write_summary(batch, cfg))
    return "\n\n".join(summaries)


def _combine_summary_lists(partials: List[List[str]]) -> List[str]:
    summaries = [
        summary.strip()
        for partial in partials
        for summary in partial
        if summary.strip()
    ]
    if not summaries:
        raise ValueError("summarize-finalize received no summaries")
    return summaries


def _finish_summary(summaries: List[str]) -> str:
    if not summaries:
        raise ValueError("summarize-finalize received no summaries")
    return "\n\n".join(summary.strip() for summary in summaries if summary.strip())
