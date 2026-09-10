"""Self-contained map and reduce steps for durable summarization workflows."""

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


def _output_limits(
    cfg: Dict[str, Any], *, target_key: str, max_key: str,
    target_default: int, max_default: int,
) -> tuple[int, int]:
    target_tokens = int(cfg.get(target_key, target_default))
    max_tokens = int(cfg.get(max_key, max_default))
    if target_tokens <= 0 or max_tokens <= target_tokens:
        raise ValueError(f"{target_key} must be positive and lower than {max_key}")
    return target_tokens, max_tokens


def _summarize_chunk(text: str, target_language: str, cfg: Dict[str, Any]) -> str:
    llm = get_llm_service(**get_llm_params("summarize-map"))
    target_tokens, max_tokens = _output_limits(
        cfg,
        target_key="chunk_target_tokens",
        max_key="chunk_max_tokens",
        target_default=250,
        max_default=400,
    )
    safe_text = truncate_for_llm(strip_dense_blobs(text), cfg)
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": _SUMMARY_USER.format(
                target_language=target_language,
                max_tokens=target_tokens,
                safe_text=safe_text,
            ),
        },
    ]
    return llm.chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    ).strip()


def _merge_summaries(partials: List[str], target_language: str, cfg: Dict[str, Any]) -> str:
    llm = get_llm_service(**get_llm_params("summarize-reduce"))
    target_tokens, max_tokens = _output_limits(
        cfg,
        target_key="merge_target_tokens",
        max_key="merge_max_tokens",
        target_default=600,
        max_default=800,
    )
    joined = "\n\n---\n\n".join(
        f"[part {i + 1}]\n{p}" for i, p in enumerate(partials) if p
    )
    joined = truncate_for_llm(
        joined,
        cfg,
        tokens_key="merge_max_tokens",
        default_tokens=800,
    )
    messages = [
        {"role": "system", "content": _MERGE_SYSTEM},
        {
            "role": "user",
            "content": _MERGE_USER.format(
                target_language=target_language,
                max_tokens=target_tokens,
                joined=joined,
            ),
        },
    ]
    return llm.chat(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=int(cfg.get("seed", 0)),
    ).strip()
