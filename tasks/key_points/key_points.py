"""Self-contained inference steps for the durable key-point workflow."""

import re
from typing import Any, Dict, List

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs
from services.llm_service import get_llm_service


def clean_sentence(sentence: str) -> str:
    return re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", sentence).strip()


def word_count(sentence: str) -> int:
    return len(re.findall(r"\w+", sentence))


def _candidates_from_generated(generated: str) -> List[str]:
    if not generated:
        return []
    candidates = [clean_sentence(line) for line in generated.splitlines()]
    candidates = [candidate for candidate in candidates if candidate]
    if candidates:
        return candidates
    return [
        clean_sentence(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", generated)
        if sentence.strip()
    ]


def _bounded_candidates(
    candidates: List[str],
    min_words: int,
    max_words: int,
) -> List[str]:
    seen = set()
    result = []
    for candidate in candidates:
        cleaned = clean_sentence(candidate)
        key = cleaned.casefold()
        if (
            not key
            or key in seen
            or not min_words <= word_count(cleaned) <= max_words
        ):
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _extract_candidates(
    content: str,
    target_language: str,
    config: Dict[str, Any],
) -> List[str]:
    safe_content = strip_dense_blobs(content).strip()
    if not safe_content:
        raise ValueError("key-point-map requires non-empty content")
    max_input_words = int(config.get("max_input_words", 1500))
    if len(safe_content.split()) > max_input_words:
        raise ValueError("key-point-map content exceeds its word budget")

    prompt_template = get_prompt("key-point-map")
    if not prompt_template:
        raise RuntimeError("key-point-map prompt is unavailable")
    generated = get_llm_service(**get_llm_params("key-point-map")).ask(
        prompt_template.format(target_lang=target_language, text=safe_content),
        max_tokens=int(config.get("max_tokens", 1000)),
        temperature=0.0,
    )
    candidates = _bounded_candidates(
        _candidates_from_generated(generated),
        min_words=int(config.get("min_words", 3)),
        max_words=int(config.get("max_words", 10)),
    )
    if not candidates:
        raise ValueError("key-point-map returned no valid candidates")
    return candidates


@execution_handler("key-point-map")
def key_point_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("key-point-map content must be a string")
    target_language = payload.get("targetLanguage")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("key-point-map targetLanguage must be a string")
    return {
        "key_points": _extract_candidates(
            content,
            target_language,
            get_task_config("key-point-map"),
        )
    }


@execution_handler("key-point-reduce")
def key_point_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("key-point-reduce requires partials")
    target_language = payload.get("targetLanguage")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("key-point-reduce targetLanguage must be a string")

    candidates: List[str] = []
    for partial in partials:
        if not isinstance(partial, list) or not partial or any(
            not isinstance(candidate, str) for candidate in partial
        ):
            raise ValueError("key-point-reduce partials must be string arrays")
        candidates.extend(partial)

    config = get_task_config("key-point-reduce")
    candidates = _bounded_candidates(
        candidates,
        min_words=int(config.get("min_words", 3)),
        max_words=int(config.get("max_words", 10)),
    )
    if not candidates:
        raise ValueError("key-point-reduce received no valid candidates")

    prompt_template = get_prompt("key-point-reduce", "refine_prompt.md")
    if not prompt_template:
        raise RuntimeError("key-point-reduce prompt is unavailable")
    max_items = int(config.get("max_items", 5))
    generated = get_llm_service(**get_llm_params("key-point-reduce")).ask(
        prompt_template.format(
            target_lang=target_language,
            candidates="\n".join(f"- {candidate}" for candidate in candidates),
            max_items=max_items,
        ),
        max_tokens=int(config.get("max_tokens", 800)),
        temperature=0.0,
    )
    selected = _bounded_candidates(
        _candidates_from_generated(generated),
        min_words=int(config.get("min_words", 3)),
        max_words=int(config.get("max_words", 10)),
    )[:max_items]
    if not selected:
        raise ValueError("key-point-reduce returned no valid key points")
    return {"key_points": selected}
