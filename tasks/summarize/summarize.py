"""Agentic summarization task.
"""

from typing import Any, Dict, List, Optional

from common.job_registry import job_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.map_reduce import MapReduceSpec, run_map_reduce
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service

_SUMMARY_SYSTEM = get_prompt("summarize", "prompts/summary_system.md").strip()
_SUMMARY_USER = get_prompt("summarize", "prompts/summary_user.md")
_MERGE_SYSTEM = get_prompt("summarize", "prompts/merge_system.md").strip()
_MERGE_USER = get_prompt("summarize", "prompts/merge_user.md")


def _target_language(payload: Dict[str, Any]) -> str:
    return payload.get("targetLanguage") or "en"


def _summarize_chunk(text: str, target_language: str, cfg: Dict[str, Any]) -> str:
    llm = get_llm_service(**get_llm_params("summarize"))
    max_tokens = int(cfg.get("chunk_max_tokens", 400))
    safe_text = truncate_for_llm(strip_dense_blobs(text), cfg)
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": _SUMMARY_USER.format(
                target_language=target_language,
                max_tokens=max_tokens,
                safe_text=safe_text,
            ),
        },
    ]
    return llm.chat(messages, max_tokens=max_tokens, temperature=0.0).strip()


def _merge_summaries(partials: List[str], target_language: str, cfg: Dict[str, Any]) -> str:
    llm = get_llm_service(**get_llm_params("summarize"))
    max_tokens = int(cfg.get("merge_max_tokens", 800))
    joined = "\n\n---\n\n".join(
        f"[part {i + 1}]\n{p}" for i, p in enumerate(partials) if p
    )
    joined = truncate_for_llm(joined, cfg)
    messages = [
        {"role": "system", "content": _MERGE_SYSTEM},
        {
            "role": "user",
            "content": _MERGE_USER.format(
                target_language=target_language,
                max_tokens=max_tokens,
                joined=joined,
            ),
        },
    ]
    return llm.chat(messages, max_tokens=max_tokens, temperature=0.0).strip()


def _leaf(chunk: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    return _summarize_chunk(chunk, _target_language(payload), cfg)


def _reduce(partials: List[str], payload: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    return _merge_summaries(partials, _target_language(payload), cfg)


_SPEC = MapReduceSpec(
    task_name="summarize",
    leaf_fn=_leaf,
    reduce_fn=_reduce,
    carry_fields=("targetLanguage", "sourceLanguage"),
    chunk_field="content",
    units_filters=["relevance"],
    recursive_merge=True,
)


@job_handler("summarize")
def summarize_text(payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None, ctx=None) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    cfg = get_task_config("summarize")
    return run_map_reduce(payload, state, ctx, spec=_SPEC, cfg=cfg)
