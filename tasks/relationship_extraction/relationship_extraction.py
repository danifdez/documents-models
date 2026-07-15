"""Agentic relationship extraction.

Mirrors the state-machine used by `summarize`, `key-point`, `keywords`, and
`date-extraction`:

- Top-level cleans the text (HTML → markdown, strip dense blobs), drops
  auxiliary sections via the relevance filter, and chunks the survivors
  with `semantic_chunk_text` (overlap-aware so cross-chunk relationships at
  boundaries still have a chance). Single chunk → run inline; N chunks →
  fan-out one `relationship-extraction` child per chunk.
- Each child detects `_chunk_idx` in payload, runs the LLM on its single
  chunk, validates relationships against the entity list (provided in the
  payload) and returns the validated raw list.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; `_phase_merge` deduplicates across chunks and persists to
  the graph. **Graph writes only happen at merge time** so a partially failed
  job never leaves an inconsistent graph.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from database.graph_db import get_graph
from lib.llm.grammars import RELATIONSHIPS_GBNF
from services.llm_service import get_llm_service
from lib.llm.config import get_llm_defaults, get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from services.relevance import select_relevant_units
from services.text import (
    extract_section_units,
    html_to_markdown,
    semantic_chunk_text,
    strip_dense_blobs,
)
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (preserved from the previous one-shot version)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM output, handling markdown fences."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _validate_relationships(relationships: list, entity_names: Set[str]) -> list:
    """Filter relationships to only those referencing known entities."""
    valid = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        subject = rel.get("subject", "")
        obj = rel.get("object", "")
        predicate = rel.get("predicate", "")
        if not subject or not obj or not predicate:
            continue
        if subject not in entity_names or obj not in entity_names:
            continue
        if subject == obj:
            continue
        valid.append({
            "subject": subject,
            "predicate": str(predicate).lower().replace(" ", "_"),
            "object": obj,
            "confidence": float(rel.get("confidence", 0.5)),
            "context": str(rel.get("context", ""))[:500],
        })
    return valid


def _deduplicate(relationships: list) -> list:
    """Remove duplicate relationships keeping the highest confidence."""
    best: Dict[tuple, dict] = {}
    for rel in relationships:
        key = (rel["subject"], rel["predicate"], rel["object"])
        if key not in best or rel["confidence"] > best[key]["confidence"]:
            best[key] = rel
    return list(best.values())


def _extract_from_chunk(chunk: str, entities_str: str, prompt_template: str,
                         llm_service, max_tokens: int) -> str:
    """Run LLM on a single chunk and return the raw response text.

    Output is grammar-constrained to a JSON array of triples and sampled at
    temperature 0: extraction should be deterministic and always parseable.
    """
    prompt = prompt_template.format(entities=entities_str, text=chunk)
    try:
        return llm_service.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            grammar=RELATIONSHIPS_GBNF,
            temperature=0.0,
        )
    except Exception:
        return llm_service.generate(
            prompt,
            max_tokens=max_tokens,
            grammar=RELATIONSHIPS_GBNF,
            temperature=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Defensive truncation against degenerate chunks
# ─────────────────────────────────────────────────────────────────────────────


def _char_budget(cfg: Dict[str, Any]) -> int:
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get("max_tokens", 2000))
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def _truncate_for_llm(text: str, cfg: Dict[str, Any]) -> str:
    cap = _char_budget(cfg)
    if len(text) <= cap:
        return text
    return text[:cap]


# ─────────────────────────────────────────────────────────────────────────────
# Graph persistence (called only from merge phase)
# ─────────────────────────────────────────────────────────────────────────────


def _persist_to_graph(
    relationships: List[Dict[str, Any]],
    entity_map: Dict[str, Dict[str, Any]],
    resource_id,
    project_id,
) -> Optional[str]:
    """Write entities and relationships to the graph. Returns an error string on
    failure, None on success or no-op."""
    if not relationships:
        return None
    graph = get_graph()
    if not graph:
        return None
    try:
        seen_entities = set()
        for rel in relationships:
            for name in (rel["subject"], rel["object"]):
                if name in seen_entities:
                    continue
                seen_entities.add(name)
                e = entity_map.get(name)
                if not e:
                    continue
                graph.upsert_entity(
                    entity_id=e["id"],
                    name=e["name"],
                    entity_type=e.get("type", "UNKNOWN"),
                    project_id=int(project_id) if project_id else None,
                    resource_id=int(resource_id) if resource_id else None,
                )

        for rel in relationships:
            subject = entity_map.get(rel["subject"])
            obj = entity_map.get(rel["object"])
            if not subject or not obj:
                continue
            graph.upsert_relationship(
                subject_id=subject["id"],
                predicate=rel["predicate"],
                object_id=obj["id"],
                resource_id=int(resource_id) if resource_id else 0,
                project_id=int(project_id) if project_id else None,
                confidence=rel["confidence"],
                context=rel.get("context", ""),
            )

        logger.info(
            "Stored %d relationships for resource %s in the graph",
            len(relationships), resource_id,
        )
        return None
    except Exception as e:
        logger.error("Failed to store relationships in the graph: %s", e)
        return f"Graph storage failed: {e}"


def _final_result(relationships: List[Dict[str, Any]], resource_id, error: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "relationships": [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
            }
            for r in relationships
        ],
        "resourceId": resource_id,
    }
    if error:
        out["error"] = error
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────────────────────


def _run_chunk_llm(
    chunk: str,
    entities_str: str,
    entity_names: Set[str],
    prompt_template: str,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Single LLM call against one chunk, validated against entity_names."""
    if not chunk or not chunk.strip():
        return []
    safe = _truncate_for_llm(strip_dense_blobs(chunk), cfg)
    try:
        params = get_llm_params("relationship-extraction")
        llm_service = get_llm_service(**params)
    except Exception as e:
        logger.error("LLM error: %s", e)
        return []
    if llm_service is None:
        return []
    max_tokens = int(cfg.get("max_tokens", 2000))
    try:
        generated = _extract_from_chunk(safe, entities_str, prompt_template, llm_service, max_tokens)
    except Exception as e:
        logger.warning("relationship-extraction chunk LLM failed: %s", e)
        return []
    raw = _parse_json_array(generated or "")
    return _validate_relationships(raw, entity_names)


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    text = payload.get("text", "")
    entities = payload.get("entities", [])
    resource_id = payload.get("resource_id") or payload.get("resourceId")
    project_id = payload.get("project_id") or payload.get("projectId")
    is_child = "_chunk_idx" in payload

    if not text or not entities:
        return _final_result([], resource_id)

    entity_names = {e["name"] for e in entities}
    entity_map = {e["name"]: e for e in entities}
    entities_str = "\n".join(
        f"- {e['name']} ({e.get('type', 'UNKNOWN')})" for e in entities
    )
    prompt_template = get_prompt("relationship-extraction")
    if not prompt_template:
        return _final_result([], resource_id, error="Prompt template not found")

    # CHILD: skip chunking; run LLM directly on the received chunk text.
    if is_child:
        valid = _run_chunk_llm(text, entities_str, entity_names, prompt_template, cfg)
        return {"relationships": valid}

    # TOP-LEVEL: clean → units → relevance → semantic chunks.
    units = extract_section_units(strip_dense_blobs(html_to_markdown(str(text))))
    if not units:
        return _final_result([], resource_id)

    if cfg.get("relevance_filter_enabled", True):
        units = select_relevant_units(
            units, cfg, task_label="relationship extraction",
        ) or units

    chunk_words = int(cfg.get("chunk_words", 600))
    chunk_overlap = int(cfg.get("chunk_overlap", 100))
    max_words_per_chunk = int(cfg.get("max_words_per_chunk", chunk_words + 100))
    chunks = semantic_chunk_text(
        units,
        target_words=chunk_words,
        max_words=max_words_per_chunk,
        overlap_words=chunk_overlap,
    )
    if not chunks:
        return _final_result([], resource_id)

    logger.info(
        "Processing %d chunk(s) for relationship extraction (%d entities)",
        len(chunks), len(entities),
    )

    # TOP-LEVEL with 1 chunk: run inline + persist.
    if len(chunks) == 1:
        valid = _run_chunk_llm(chunks[0], entities_str, entity_names, prompt_template, cfg)
        deduped = _deduplicate(valid)
        err = _persist_to_graph(deduped, entity_map, resource_id, project_id)
        return _final_result(deduped, resource_id, error=err)

    # No DB ctx (unit tests / fallback): run all chunks in-process.
    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        all_valid: List[Dict[str, Any]] = []
        for c in chunks:
            all_valid.extend(_run_chunk_llm(c, entities_str, entity_names, prompt_template, cfg))
        deduped = _deduplicate(all_valid)
        err = _persist_to_graph(deduped, entity_map, resource_id, project_id)
        return _final_result(deduped, resource_id, error=err)

    # FAN-OUT: one child per chunk.
    pending: Dict[str, int] = {}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        child_payload = {
            "text": chunk,
            "entities": entities,
            "resource_id": resource_id,
            "project_id": project_id,
            "_chunk_idx": i,
        }
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id,
            "relationship-extraction",
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
        "entities": entities,
        "resource_id": resource_id,
        "project_id": project_id,
        "chunk_field": "text",
        "chunk_payload_template": {
            "entities": entities,
            "resource_id": resource_id,
            "project_id": project_id,
        },
    }
    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _phase_merge(state: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return _final_result(
            [],
            state.get("resource_id"),
            error=(
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            ),
        )

    n = int(state.get("chunks_count", 0))
    entities = state.get("entities") or []
    resource_id = state.get("resource_id")
    project_id = state.get("project_id")
    entity_map = {e["name"]: e for e in entities}
    results = state.get("results") or {}

    all_rels: List[Dict[str, Any]] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            for rel in (r.get("relationships") or []):
                if isinstance(rel, dict):
                    all_rels.append(rel)

    deduped = _deduplicate(all_rels)
    err = _persist_to_graph(deduped, entity_map, resource_id, project_id)
    return _final_result(deduped, resource_id, error=err)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@job_handler("relationship-extraction")
def extract_relationships(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    try:
        cfg = get_task_config("relationship-extraction")
        if state and state.get("phase") == "merging":
            return _phase_merge(state, cfg, ctx)
        return _phase_plan_or_leaf(payload, cfg, ctx)
    except Exception as e:
        logger.exception("relationship-extraction failed")
        resource_id = (
            (state or {}).get("resource_id")
            or payload.get("resource_id")
            or payload.get("resourceId")
        )
        return _final_result([], resource_id, error=f"relationship-extraction failed: {e}")
