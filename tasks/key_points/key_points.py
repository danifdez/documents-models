"""Agentic key-point extraction.

Runs on the shared map-reduce state machine (`lib.llm.map_reduce`), like
`summarize`:

- Top-level invocation chunks the content. Single chunk → run the full
  extraction + dedup + refine + rank pipeline inline. Multiple chunks → fan
  out one `key-point` child job per chunk and wait.
- Each child detects it's running on a single chunk (via the `_chunk_idx` marker
  in payload) and returns only the *raw* per-chunk candidates so the parent can
  do cross-chunk semantic dedup, refine and ranking.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; the reduce step runs the cross-chunk pipeline and returns
  the final `key_points` list.

Defends against pathological inputs (data URIs, base64 blobs) the same way as
`summarize` so a single inline blob can't blow Phi's context window.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.types import ModelSpec
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
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Defensive content sanitization is shared with summarize via services.text.
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
        generated = llm_service.ask(prompt, max_tokens=max_tokens, temperature=0.0)
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
    safe = truncate_for_llm(strip_dense_blobs(chunk), cfg,
                            tokens_key="max_tokens", default_tokens=1000)
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
        generated = llm_service.ask(prompt, max_tokens=max_tokens, temperature=0.0)
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
            refine_template = get_prompt("key-point", "refine_prompt.md")
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
# Map-reduce spec
# ─────────────────────────────────────────────────────────────────────────────


def _target_lang(payload: Dict[str, Any]) -> str:
    return payload.get("targetLanguage") or payload.get("target_language") or "en"


def _chunks(payload: Dict[str, Any], cfg: Dict[str, Any], is_child: bool) -> List[str]:
    raw_content = payload.get("content", "") or ""
    if not str(raw_content).strip():
        return []
    units_filter = None
    if not is_child:
        target_lang = _target_lang(payload)
        units_filter = lambda us: select_relevant_units(
            us, cfg, task_label="key-point extraction", target_lang=target_lang,
        )
    return _build_chunks(
        raw_content, int(cfg.get("chunk_word_budget", 1500)), units_filter=units_filter
    )


def _leaf(chunk: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    return _extract_chunk_candidates(chunk, _target_lang(payload), cfg)


def _reduce(partials: List[Any], payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    per_chunk_lists: List[List[str]] = []
    for kp in partials:
        kp = kp or []
        per_chunk_lists.append([str(x) for x in kp] if isinstance(kp, list) else [])
    return _merge_pipeline(
        per_chunk_lists,
        payload.get("_chunks") or [],
        payload.get("content", "") or "",
        _target_lang(payload),
        cfg,
    )


def _child_static(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"targetLanguage": _target_lang(payload)}


def _fanout_extras(chunks: List[str], payload: Dict[str, Any], cfg: Dict[str, Any]):
    return None, {
        "targetLanguage": _target_lang(payload),
        "raw_content": payload.get("content", "") or "",
    }


def _merge_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": state.get("raw_content", ""),
        "targetLanguage": state.get("targetLanguage"),
    }


_SPEC = MapReduceSpec(
    task_name="key-point",
    leaf_fn=_leaf,
    reduce_fn=_reduce,
    chunk_field="content",
    recursive_merge=False,
    result_key="key_points",
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


@job_handler("key-point")
def key_points(payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None, ctx=None) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    try:
        cfg = get_task_config("key-point")
        return run_map_reduce(payload, state, ctx, spec=_SPEC, cfg=cfg)
    except Exception as e:
        logger.exception("Error extracting key points")
        return {"error": f"Error extracting key points: {e}"}
