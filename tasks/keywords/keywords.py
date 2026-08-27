"""Self-contained steps for the durable keywords workflow."""

import re
from typing import Any, Dict, List

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs
from services.llm_service import get_llm_service


def split_and_clean(generated: str) -> List[str]:
    parts = re.split(r"[\n,]+", generated)
    cleaned = []
    for part in parts:
        item = re.sub(r"^\s*[-\d.\)]+\s*", "", part).strip()
        if item:
            cleaned.append(item)
    return cleaned


def _truncate_words(item: str, max_words: int) -> str:
    return " ".join(item.split()[:max_words]).strip()


def _candidate_statistics(
    candidate_lists: List[List[str]], leaf_start: int, max_words: int
) -> List[Dict[str, Any]]:
    statistics: Dict[str, Dict[str, Any]] = {}
    for partial_index, candidates in enumerate(candidate_lists):
        chunk_seen = set()
        for candidate_index, raw in enumerate(candidates):
            item = _truncate_words(raw, max_words)
            if not item:
                continue
            key = item.casefold()
            if key in chunk_seen:
                continue
            chunk_seen.add(key)
            if key not in statistics:
                statistics[key] = {
                    "value": item,
                    "count": 0,
                    "firstChunk": leaf_start + partial_index,
                    "firstCandidate": candidate_index,
                }
            statistics[key]["count"] += 1
    return list(statistics.values())


def _merge_statistics(
    partials: List[List[Dict[str, Any]]], max_words: int
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for partial in partials:
        for raw in partial:
            statistic = _validate_statistic(raw, max_words)
            key = statistic["value"].casefold()
            if key not in merged:
                merged[key] = statistic
                continue
            merged[key]["count"] += statistic["count"]
            current_order = (
                merged[key]["firstChunk"],
                merged[key]["firstCandidate"],
            )
            candidate_order = (
                statistic["firstChunk"],
                statistic["firstCandidate"],
            )
            if candidate_order < current_order:
                merged[key].update(
                    value=statistic["value"],
                    firstChunk=statistic["firstChunk"],
                    firstCandidate=statistic["firstCandidate"],
                )
    return list(merged.values())


def _validate_statistic(raw: Any, max_words: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("keywords-reduce statistics must be objects")
    value = raw.get("value")
    count = raw.get("count")
    first_chunk = raw.get("firstChunk")
    first_candidate = raw.get("firstCandidate")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("keywords-reduce statistic value is invalid")
    if _truncate_words(value, max_words) != value.strip():
        raise ValueError("keywords-reduce statistic value exceeds its word limit")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("keywords-reduce statistic count is invalid")
    if (
        not isinstance(first_chunk, int)
        or isinstance(first_chunk, bool)
        or first_chunk < 0
        or not isinstance(first_candidate, int)
        or isinstance(first_candidate, bool)
        or first_candidate < 0
    ):
        raise ValueError("keywords-reduce statistic order is invalid")
    return {
        "value": value.strip(),
        "count": count,
        "firstChunk": first_chunk,
        "firstCandidate": first_candidate,
    }


def _ordered_statistics(statistics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        statistics,
        key=lambda item: (item["firstChunk"], item["firstCandidate"]),
    )


def _extract_candidates(
    content: str,
    target_language: str,
    config: Dict[str, Any],
) -> List[str]:
    safe_content = strip_dense_blobs(content).strip()
    if not safe_content:
        raise ValueError("keywords-map requires non-empty content")
    max_input_words = int(config.get("max_input_words", 1500))
    if len(safe_content.split()) > max_input_words:
        raise ValueError("keywords-map content exceeds its word budget")

    prompt_template = get_prompt("keywords-map")
    if not prompt_template:
        raise RuntimeError("keywords-map prompt is unavailable")
    generated = get_llm_service(**get_llm_params("keywords-map")).ask(
        prompt_template.format(
            target_lang=target_language,
            text=safe_content,
        ),
        max_tokens=int(config.get("max_tokens", 500)),
        temperature=0.0,
    )
    candidates = split_and_clean(generated) if generated else []
    if not candidates:
        raise ValueError("keywords-map returned no candidates")
    return candidates


@execution_handler("keywords-map")
def keywords_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("keywords-map content must be a string")
    target_language = payload.get("targetLanguage")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("keywords-map targetLanguage must be a string")
    return {
        "keywords": _extract_candidates(
            content,
            target_language,
            get_task_config("keywords-map"),
        )
    }


@execution_handler("keywords-reduce")
def keywords_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("keywords-reduce requires partials")

    config = get_task_config("keywords-reduce")
    max_words = int(config.get("max_words_per_item", 3))
    input_kind = payload.get("inputKind")
    if input_kind == "candidates":
        candidate_lists: List[List[str]] = []
        for partial in partials:
            if not isinstance(partial, list) or not partial or any(
                not isinstance(candidate, str) for candidate in partial
            ):
                raise ValueError("keywords-reduce partials must be string arrays")
            candidate_lists.append(partial)
        leaf_start = payload.get("leafStartIndex")
        if (
            not isinstance(leaf_start, int)
            or isinstance(leaf_start, bool)
            or leaf_start < 0
        ):
            raise ValueError("keywords-reduce leafStartIndex is invalid")
        statistics = _candidate_statistics(candidate_lists, leaf_start, max_words)
    elif input_kind == "statistics":
        statistic_lists: List[List[Dict[str, Any]]] = []
        for partial in partials:
            if not isinstance(partial, list) or not partial:
                raise ValueError("keywords-reduce partials must be statistic arrays")
            statistic_lists.append(partial)
        statistics = _merge_statistics(statistic_lists, max_words)
    else:
        raise ValueError("keywords-reduce inputKind is invalid")

    statistics = _ordered_statistics(statistics)
    if payload.get("final") is not True:
        return {"keyword_statistics": statistics}

    ranked = sorted(
        statistics,
        key=lambda item: (
            -item["count"],
            item["firstChunk"],
            item["firstCandidate"],
        ),
    )
    keywords = [
        item["value"] for item in ranked[: int(config.get("max_items", 10))]
    ]
    if not keywords:
        raise ValueError("keywords-reduce received no candidates")
    return {"keywords": keywords}
