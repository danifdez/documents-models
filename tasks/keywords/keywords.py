"""Agentic keyword extraction.

Runs on the shared map-reduce state machine (`lib.llm.map_reduce`), like
`summarize`, `key-point` and `date-extraction`:

- Top-level invocation cleans the text (HTML → markdown, strip dense blobs),
  runs the relevance filter to drop bibliography/appendix-like sections, and
  chunks the survivors. Single chunk → run extraction inline; N chunks →
  fan-out one `keywords` child per chunk and wait.
- Each child detects `_chunk_idx` in payload and returns the *raw*
  per-chunk candidate list. The parent merges them with the existing
  frequency-then-first-appearance ranking.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; the reduce step produces the final ranked keyword list.

Defends against pathological inputs (data URIs, long base64 blobs) and
truncates per-chunk LLM input to a safe character budget so a degenerate
chunk can't blow Phi's context window.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from services.llm_service import get_llm_service
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.map_reduce import MapReduceSpec, run_map_reduce
from lib.llm.prompts import get_prompt
from lib.llm.text import truncate_for_llm
from services.relevance import select_relevant_units
from services.text import (
    chunk_units,
    extract_section_units,
    html_to_markdown,
    normalize_text,
    strip_dense_blobs,
)
from common.execution_registry import execution_handler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (preserved from the previous one-shot version)
# ─────────────────────────────────────────────────────────────────────────────


def split_and_clean(generated: str) -> List[str]:
    parts = re.split(r'[\n,]+', generated)
    cleaned = []
    for p in parts:
        it = re.sub(r'^\s*[-\d\.\)]+\s*', '', p).strip()
        if it:
            cleaned.append(it)
    return cleaned


def _truncate_words(item: str, max_words: int) -> str:
    return ' '.join(item.split()[:max_words]).strip()


def _merge_candidates(
    candidate_lists: List[List[str]],
    max_items: int,
    max_words: int,
) -> List[str]:
    """Merge per-chunk candidate lists, ranking by frequency across chunks then by first appearance."""
    counts: Dict[str, int] = {}
    first_form: Dict[str, str] = {}
    first_seen: Dict[str, int] = {}
    order = 0
    for cands in candidate_lists:
        chunk_seen = set()
        for raw in cands:
            item = _truncate_words(raw, max_words)
            if not item:
                continue
            key = item.lower()
            if key in chunk_seen:
                continue
            chunk_seen.add(key)
            if key not in counts:
                counts[key] = 0
                first_form[key] = item
                first_seen[key] = order
                order += 1
            counts[key] += 1

    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k]))
    return [first_form[k] for k in ranked[:max_items]]


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk LLM extraction (used by both leaf and child paths)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_chunk_candidates(chunk: str, target_lang: str, cfg: Dict[str, Any]) -> List[str]:
    if not chunk or not chunk.strip():
        return []
    safe = truncate_for_llm(strip_dense_blobs(chunk), cfg,
                            tokens_key="max_tokens", default_tokens=500)
    try:
        params = get_llm_params("keywords")
        llm_service = get_llm_service(**params)
    except Exception as e:
        logger.warning("LLM service unavailable for keywords extraction: %s", e)
        return []
    if llm_service is None:
        return []
    prompt_template = get_prompt("keywords")
    max_tokens = int(cfg.get("max_tokens", 500))
    try:
        prompt = prompt_template.format(target_lang=target_lang, text=safe)
        generated = llm_service.ask(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as e:
        logger.warning("keywords chunk extraction failed: %s", e)
        return []
    return split_and_clean(generated) if generated else []


# ─────────────────────────────────────────────────────────────────────────────
# Cross-chunk merge (with fallback when LLM produced nothing)
# ─────────────────────────────────────────────────────────────────────────────


def _merge_pipeline(
    per_chunk_lists: List[List[str]],
    raw_content: str,
    cfg: Dict[str, Any],
) -> List[str]:
    max_items = int(cfg.get("max_items", 10))
    max_words = int(cfg.get("max_words_per_item", 3))

    keywords_list = _merge_candidates(per_chunk_lists, max_items=max_items, max_words=max_words)

    if not keywords_list and raw_content:
        # Fallback: pull short sentences from the raw text.
        full_text = normalize_text(str(raw_content))
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
        heuristic = [_truncate_words(s, max_words) for s in sentences]
        seen = set()
        for item in heuristic:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            keywords_list.append(item)
            if len(keywords_list) >= max_items:
                break

    return keywords_list


# ─────────────────────────────────────────────────────────────────────────────
# Chunking helper
# ─────────────────────────────────────────────────────────────────────────────


def _build_chunks(
    content: str,
    chunk_word_budget: int,
    *,
    units_filter=None,
) -> List[str]:
    cleaned = strip_dense_blobs(html_to_markdown(content or ""))
    units = extract_section_units(cleaned)
    if not units:
        return []
    if units_filter is not None:
        units = units_filter(units) or units
    return chunk_units(units, chunk_word_budget, joiner="\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# Map-reduce spec
# ─────────────────────────────────────────────────────────────────────────────


def _target_lang(payload: Dict[str, Any]) -> str:
    return (
        payload.get("targetLanguage")
        or payload.get("target_language")
        or "auto"
    )


def _chunks(payload: Dict[str, Any], cfg: Dict[str, Any], is_child: bool) -> List[str]:
    raw_content = payload.get("content", "") or ""
    if not str(raw_content).strip():
        return []
    units_filter = None
    if not is_child:
        target_lang = _target_lang(payload)
        units_filter = lambda us: select_relevant_units(
            us, cfg, task_label="keyword extraction", target_lang=target_lang,
        )
    return _build_chunks(
        raw_content, int(cfg.get("chunk_word_budget", 1500)), units_filter=units_filter
    )


def _leaf(chunk: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    return _extract_chunk_candidates(chunk, _target_lang(payload), cfg)


def _reduce(partials: List[Any], payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    per_chunk_lists: List[List[str]] = []
    for kw in partials:
        kw = kw or []
        per_chunk_lists.append([str(x) for x in kw] if isinstance(kw, list) else [])
    return _merge_pipeline(per_chunk_lists, payload.get("content", "") or "", cfg)


def _child_static(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"targetLanguage": _target_lang(payload)}


def _fanout_extras(chunks: List[str], payload: Dict[str, Any], cfg: Dict[str, Any]):
    return None, {
        "targetLanguage": _target_lang(payload),
        "raw_content": payload.get("content", "") or "",
    }


def _merge_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"content": state.get("raw_content", "")}


_SPEC = MapReduceSpec(
    task_name="keywords",
    leaf_fn=_leaf,
    reduce_fn=_reduce,
    chunk_field="content",
    recursive_merge=False,
    result_key="keywords",
    empty_value=[],
    list_results=True,
    chunks_fn=_chunks,
    child_static_fn=_child_static,
    fanout_extras_fn=_fanout_extras,
    merge_payload_fn=_merge_payload,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@execution_handler("keywords")
def keywords(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    try:
        cfg = get_task_config("keywords")
        return run_map_reduce(payload, state, ctx, spec=_SPEC, cfg=cfg)
    except Exception as e:
        logger.exception("Error extracting keywords")
        return {"error": f"Error extracting keywords: {e}"}
