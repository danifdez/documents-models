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


def _merge_candidates(
    candidate_lists: List[List[str]],
    max_items: int,
    max_words: int,
) -> List[str]:
    counts: Dict[str, int] = {}
    first_form: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    order = 0
    for candidates in candidate_lists:
        chunk_seen = set()
        for raw in candidates:
            item = _truncate_words(raw, max_words)
            if not item:
                continue
            key = item.casefold()
            if key in chunk_seen:
                continue
            chunk_seen.add(key)
            if key not in counts:
                counts[key] = 0
                first_form[key] = item
                first_seen[key] = order
                order += 1
            counts[key] += 1

    ranked = sorted(counts, key=lambda key: (-counts[key], first_seen[key]))
    return [first_form[key] for key in ranked[:max_items]]


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

    candidate_lists: List[List[str]] = []
    for partial in partials:
        if not isinstance(partial, list) or not partial or any(
            not isinstance(candidate, str) for candidate in partial
        ):
            raise ValueError("keywords-reduce partials must be string arrays")
        candidate_lists.append(partial)

    config = get_task_config("keywords-reduce")
    keywords = _merge_candidates(
        candidate_lists,
        max_items=int(config.get("max_items", 10)),
        max_words=int(config.get("max_words_per_item", 3)),
    )
    if not keywords:
        raise ValueError("keywords-reduce received no candidates")
    return {"keywords": keywords}
