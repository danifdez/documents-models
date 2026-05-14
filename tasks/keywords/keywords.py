"""Agentic keyword extraction.

Mirrors the state-machine used by `summarize`, `key-point` and
`date-extraction`:

- Top-level invocation cleans the text (HTML → markdown, strip dense blobs),
  runs the relevance filter to drop bibliography/appendix-like sections, and
  chunks the survivors. Single chunk → run extraction inline; N chunks →
  fan-out one `keywords` child per chunk and wait.
- Each child detects `_chunk_idx` in payload and returns the *raw*
  per-chunk candidate list. The parent merges them with the existing
  frequency-then-first-appearance ranking.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; `_phase_merge` produces the final ranked keyword list.

Defends against pathological inputs (data URIs, long base64 blobs) and
truncates per-chunk LLM input to a safe character budget so a degenerate
chunk can't blow Phi's context window.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from services.llm_service import get_llm_service
from services.model_config import get_llm_defaults, get_llm_params, get_task_config
from services.prompts import get_prompt
from services.relevance import select_relevant_units
from services.text import (
    chunk_units,
    extract_section_units,
    html_to_markdown,
    normalize_text,
    strip_dense_blobs,
)
from utils.job_registry import job_handler

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
# Defensive truncation against degenerate chunks
# ─────────────────────────────────────────────────────────────────────────────


def _char_budget(cfg: Dict[str, Any]) -> int:
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get("max_tokens", 500))
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def _truncate_for_llm(text: str, cfg: Dict[str, Any]) -> str:
    cap = _char_budget(cfg)
    if len(text) <= cap:
        return text
    return text[:cap]


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk LLM extraction (used by both leaf and child paths)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_chunk_candidates(chunk: str, target_lang: str, cfg: Dict[str, Any]) -> List[str]:
    if not chunk or not chunk.strip():
        return []
    safe = _truncate_for_llm(strip_dense_blobs(chunk), cfg)
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
        try:
            generated = llm_service.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception:
            generated = llm_service.generate(prompt, max_tokens=max_tokens)
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
# Phases
# ─────────────────────────────────────────────────────────────────────────────


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    target_lang = (
        payload.get("targetLanguage")
        or payload.get("target_language")
        or "auto"
    )
    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    is_child = "_chunk_idx" in payload

    raw_content = payload.get("content", "") or ""
    if not str(raw_content).strip():
        return {"keywords": []}

    units_filter = None
    if not is_child:
        units_filter = lambda us: select_relevant_units(
            us, cfg, task_label="keyword extraction", target_lang=target_lang,
        )

    chunks = _build_chunks(raw_content, chunk_word_budget, units_filter=units_filter)
    if not chunks:
        return {"keywords": []}

    # CHILD: return raw per-chunk candidates; the parent does the cross-chunk merge.
    if is_child:
        candidates: List[str] = []
        for c in chunks:
            candidates.extend(_extract_chunk_candidates(c, target_lang, cfg))
        return {"keywords": candidates}

    # TOP-LEVEL with a single chunk: run the full pipeline inline.
    if len(chunks) == 1:
        per_chunk = [_extract_chunk_candidates(chunks[0], target_lang, cfg)]
        return {"keywords": _merge_pipeline(per_chunk, raw_content, cfg)}

    # No DB context (unit tests / fallback): in-process serial.
    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        per_chunk_lists = [_extract_chunk_candidates(c, target_lang, cfg) for c in chunks]
        return {"keywords": _merge_pipeline(per_chunk_lists, raw_content, cfg)}

    # FAN-OUT: one child per chunk.
    pending: Dict[str, int] = {}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        child_payload = {
            "content": chunk,
            "targetLanguage": target_lang,
            "_chunk_idx": i,
        }
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id,
            "keywords",
            payload=child_payload,
            agent_max_steps=1,
        )
        if child_id is None:
            return {"error": f"failed to enqueue child for chunk {i}"}
        pending[str(child_id)] = i
        results[str(i)] = None
        retries[str(i)] = 0

    state = {
        "phase": "merging",
        "chunks_count": len(chunks),
        "pending": pending,
        "results": results,
        "retries": retries,
        "chunks": chunks,
        "targetLanguage": target_lang,
        "raw_content": raw_content,
        "chunk_field": "content",
        "chunk_payload_template": {"targetLanguage": target_lang},
    }

    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _phase_merge(state: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return {
            "error": (
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            )
        }

    n = int(state.get("chunks_count", 0))
    raw_content = state.get("raw_content", "")
    results = state.get("results") or {}

    per_chunk_lists: List[List[str]] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            kw = r.get("keywords") or []
            if isinstance(kw, list):
                per_chunk_lists.append([str(x) for x in kw])
            else:
                per_chunk_lists.append([])
        else:
            per_chunk_lists.append([])

    return {"keywords": _merge_pipeline(per_chunk_lists, raw_content, cfg)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@job_handler("keywords")
def keywords(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    try:
        cfg = get_task_config("keywords")
        if state and state.get("phase") == "merging":
            return _phase_merge(state, cfg, ctx)
        return _phase_plan_or_leaf(payload, cfg, ctx)
    except Exception as e:
        logger.exception("Error extracting keywords")
        return {"error": f"Error extracting keywords: {e}"}
