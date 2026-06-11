"""Agentic key-point extraction.

Mirrors the agentic state-machine used by `summarize`:

- Top-level invocation chunks the content. Single chunk → run the full
  extraction + dedup + refine + rank pipeline inline. Multiple chunks → fan
  out one `key-point` child job per chunk and wait.
- Each child detects it's running on a single chunk (via the `_chunk_idx` marker
  in payload) and returns only the *raw* per-chunk candidates so the parent can
  do cross-chunk semantic dedup, refine and ranking.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; `_phase_merge` runs the cross-chunk pipeline and returns
  the final `key_points` list.

Defends against pathological inputs (data URIs, base64 blobs) the same way as
`summarize` so a single inline blob can't blow Phi's context window.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.types import ModelSpec
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
# Defensive content sanitization is shared with summarize via services.text.
# ─────────────────────────────────────────────────────────────────────────────


def _char_budget(cfg: Dict[str, Any]) -> int:
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get("max_tokens", 1000))
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def _truncate_for_llm(text: str, cfg: Dict[str, Any]) -> str:
    cap = _char_budget(cfg)
    if len(text) <= cap:
        return text
    return text[:cap]


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
# Existing helpers (preserved from the seq2seq version)
# ─────────────────────────────────────────────────────────────────────────────


def clean_sentence(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\d+\.|^-|^\*', '', s).strip()
    return s


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


def _candidates_from_generated(generated: str) -> List[str]:
    if not generated:
        return []
    candidates = [clean_sentence(line) for line in generated.splitlines()]
    candidates = [c for c in candidates if c]
    if not candidates:
        candidates = [clean_sentence(s) for s in re.split(r'(?<=[.!?])\s+', generated) if s.strip()]
    return candidates


def _read_local_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _embed(texts: List[str]):
    if not texts:
        return None
    try:
        from services.embedding_service import get_embedding_service
        emb = get_embedding_service().encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        logger.warning("Embedding service unavailable, skipping semantic step: %s", e)
        return None


def _semantic_dedupe(candidates: List[str], threshold: float) -> Tuple[List[str], Optional[np.ndarray]]:
    if len(candidates) <= 1:
        return candidates, None
    emb = _embed(candidates)
    if emb is None:
        return candidates, None
    kept_idx: List[int] = []
    kept_emb: List[np.ndarray] = []
    for i, vec in enumerate(emb):
        if not kept_emb:
            kept_idx.append(i)
            kept_emb.append(vec)
            continue
        sims = np.array([float(np.dot(vec, k)) for k in kept_emb])
        if sims.max() < threshold:
            kept_idx.append(i)
            kept_emb.append(vec)
    deduped = [candidates[i] for i in kept_idx]
    return deduped, np.stack(kept_emb) if kept_emb else None


def _rank_by_centrality(
    candidates: List[str],
    cand_emb: Optional[np.ndarray],
    doc_centroid: Optional[np.ndarray],
) -> List[str]:
    if cand_emb is None or doc_centroid is None or len(candidates) != len(cand_emb):
        return candidates
    scores = cand_emb @ doc_centroid
    order = np.argsort(-scores)
    return [candidates[i] for i in order]


def _refine_chunk(
    items: List[str],
    target_lang: str,
    max_items: int,
    prompt_template: str,
    llm_service,
    max_tokens: int,
) -> List[str]:
    if not items or llm_service is None:
        return items
    candidates_block = "\n".join(f"- {c}" for c in items)
    try:
        prompt = prompt_template.format(
            target_lang=target_lang,
            candidates=candidates_block,
            max_items=max_items,
        )
    except (KeyError, IndexError):
        return items
    try:
        generated = llm_service.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as e:
        logger.warning("Refine LLM call failed: %s", e)
        return items
    refined = _candidates_from_generated(generated)
    return refined or items


def _refine_chunked(
    candidates: List[str],
    target_lang: str,
    max_items: int,
    prompt_template: str,
    llm_service,
    max_tokens: int,
    chunk_size: int,
    overselect: int,
    threshold: float,
) -> List[str]:
    if not candidates:
        return candidates
    if len(candidates) <= chunk_size:
        return _refine_chunk(candidates, target_lang, max_items, prompt_template, llm_service, max_tokens)

    per_chunk_target = max(max_items, max_items * overselect)
    partials: List[str] = []
    for start in range(0, len(candidates), chunk_size):
        piece = candidates[start:start + chunk_size]
        partials.extend(
            _refine_chunk(piece, target_lang, per_chunk_target, prompt_template, llm_service, max_tokens)
        )

    seen = set()
    flat: List[str] = []
    for p in partials:
        k = p.lower().strip()
        if k and k not in seen:
            seen.add(k)
            flat.append(p)
    flat, _ = _semantic_dedupe(flat, threshold)

    if len(flat) > chunk_size:
        return _refine_chunked(
            flat, target_lang, max_items, prompt_template,
            llm_service, max_tokens, chunk_size, overselect, threshold,
        )
    return _refine_chunk(flat, target_lang, max_items, prompt_template, llm_service, max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk LLM extraction (used by both leaf and child paths)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_chunk_candidates(chunk: str, target_lang: str, cfg: Dict[str, Any]) -> List[str]:
    if not chunk or not chunk.strip():
        return []
    safe = _truncate_for_llm(strip_dense_blobs(chunk), cfg)
    try:
        params = get_llm_params("key-point")
        llm_service = get_llm_service(**params)
    except Exception as e:
        logger.warning("LLM service unavailable for key-point extraction: %s", e)
        return []
    if llm_service is None:
        return []
    prompt_template = get_prompt("key-point")
    max_tokens = int(cfg.get("max_tokens", 1000))
    try:
        prompt = prompt_template.format(target_lang=target_lang, text=safe)
        generated = llm_service.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as e:
        logger.warning("key-point chunk extraction failed: %s", e)
        return []
    return _candidates_from_generated(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-chunk merge pipeline (dedup + refine + rank + fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _merge_pipeline(
    per_chunk_lists: List[List[str]],
    chunks: List[str],
    raw_content: str,
    target_lang: str,
    cfg: Dict[str, Any],
) -> List[str]:
    min_words = int(cfg.get("min_words", 3))
    max_words = int(cfg.get("max_words", 10))
    max_items = int(cfg.get("max_items", 5))
    threshold = float(cfg.get("dedupe_similarity_threshold", 0.85))

    seen = set()
    candidates: List[str] = []
    for lst in per_chunk_lists:
        for c in lst or []:
            k = c.lower().strip()
            if k and k not in seen:
                seen.add(k)
                candidates.append(c)

    candidates = [c for c in candidates if min_words <= word_count(c) <= max_words]
    deduped, deduped_emb = _semantic_dedupe(candidates, threshold)

    refine_enabled = bool(cfg.get("refine_enabled", True))
    if refine_enabled and len(deduped) > max_items:
        try:
            params = get_llm_params("key-point")
            llm_service = get_llm_service(**params)
        except Exception:
            llm_service = None
        if llm_service is not None:
            refine_template = _read_local_prompt("refine_prompt.md")
            if refine_template:
                refine_max_tokens = int(cfg.get("refine_max_tokens", cfg.get("max_tokens", 1000)))
                refine_chunk_size = int(cfg.get("refine_chunk_size", 30))
                overselect = int(cfg.get("refine_overselect", 3))
                refined = _refine_chunked(
                    deduped, target_lang, max_items, refine_template,
                    llm_service, refine_max_tokens, refine_chunk_size, overselect, threshold,
                )
                refined = [c for c in refined if min_words <= word_count(c) <= max_words]
                refined, deduped_emb = _semantic_dedupe(refined, threshold)
                deduped = refined

    chunk_emb = _embed(chunks) if deduped_emb is not None else None
    doc_centroid = None
    if chunk_emb is not None and len(chunk_emb) > 0:
        centroid = chunk_emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            doc_centroid = centroid / norm

    ranked = _rank_by_centrality(deduped, deduped_emb, doc_centroid)

    selected: List[str] = []
    selected_keys: set = set()
    for s in ranked:
        key = s.lower()
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(s)
        if len(selected) >= max_items:
            break

    if len(selected) < max_items:
        full_text = normalize_text(str(raw_content))
        for s in re.split(r'(?<=[.!?])\s+', full_text):
            cs = clean_sentence(s)
            if not cs:
                continue
            wc = word_count(cs)
            if min_words <= wc <= max_words and cs.lower() not in selected_keys:
                selected_keys.add(cs.lower())
                selected.append(cs)
            if len(selected) >= max_items:
                break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────────────────────


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    target_lang = payload.get("targetLanguage") or payload.get("target_language") or "en"
    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    is_child = "_chunk_idx" in payload

    raw_content = payload.get("content", "") or ""
    if not str(raw_content).strip():
        return {"key_points": []}

    units_filter = None
    if not is_child:
        units_filter = lambda us: select_relevant_units(
            us, cfg, task_label="key-point extraction", target_lang=target_lang,
        )

    chunks = _build_chunks(raw_content, chunk_word_budget, units_filter=units_filter)
    if not chunks:
        return {"key_points": []}

    # CHILD invocation (one chunk per child by construction). Return raw
    # per-chunk candidates; the parent does cross-chunk dedup/refine/rank.
    if is_child:
        candidates: List[str] = []
        for c in chunks:
            candidates.extend(_extract_chunk_candidates(c, target_lang, cfg))
        return {"key_points": candidates}

    # TOP-LEVEL with a single chunk: run the full pipeline inline.
    if len(chunks) == 1:
        per_chunk = [_extract_chunk_candidates(chunks[0], target_lang, cfg)]
        return {"key_points": _merge_pipeline(per_chunk, chunks, raw_content, target_lang, cfg)}

    # No DB context (e.g. unit tests): run sequentially in-process.
    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        per_chunk_lists = [_extract_chunk_candidates(c, target_lang, cfg) for c in chunks]
        return {"key_points": _merge_pipeline(per_chunk_lists, chunks, raw_content, target_lang, cfg)}

    # TOP-LEVEL with N chunks: fan out one child per chunk.
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
            "key-point",
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
    target_lang = state.get("targetLanguage") or "en"
    chunks = state.get("chunks") or []
    raw_content = state.get("raw_content", "")
    results = state.get("results") or {}

    per_chunk_lists: List[List[str]] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            kp = r.get("key_points") or []
            if isinstance(kp, list):
                per_chunk_lists.append([str(x) for x in kp])
            else:
                per_chunk_lists.append([])
        else:
            per_chunk_lists.append([])

    final = _merge_pipeline(per_chunk_lists, chunks, raw_content, target_lang, cfg)
    return {"key_points": final}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@job_handler("key-point")
def key_points(payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None, ctx=None) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    try:
        cfg = get_task_config("key-point")
        if state and state.get("phase") == "merging":
            return _phase_merge(state, cfg, ctx)
        return _phase_plan_or_leaf(payload, cfg, ctx)
    except Exception as e:
        logger.exception("Error extracting key points")
        return {"error": f"Error extracting key points: {e}"}
