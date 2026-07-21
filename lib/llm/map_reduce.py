"""Task-agnostic reentrant map-reduce orchestrator.

Extracts the skeleton that used to live inside `tasks/summarize/summarize.py`
so any content task (summarize, keywords, key-point, …) can reuse the same
pattern: chunk the input, process each chunk in parallel through the job system
(fan-out) and merge the partial results (reduce), with optional recursion when
the merge is still too large.

A task is defined by declaring a `MapReduceSpec` with two functions —`leaf_fn`
(how to process one chunk) and `reduce_fn` (how to merge partials)— and calling
`run_map_reduce(payload, state, ctx, spec=..., cfg=...)` from its `@job_handler`.

Dispatcher contract (shared with documents-dev via `job_mock`/`process_job`):
the fan-out returns
`{"_sub_agent_pending_many": True, "_state": {...}, "pending_children": {...}}`
and the `_state` keeps the shape `resume_parent_with_child` knows how to
reconstruct (`phase`, `chunks`, `results`, `retries`, `chunk_field`,
`chunk_payload_template`, plus the `carry_fields`). Do not change that shape
without also touching the dispatcher.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.llm.text import build_chunks, word_count
from lib.llm.unit_filters import build_units_filter

# (chunk_text, payload, cfg) -> processed text for that chunk.
LeafFn = Callable[[str, Dict[str, Any], Dict[str, Any]], str]
# (partials, payload, cfg) -> merged text.
ReduceFn = Callable[[List[str], Dict[str, Any], Dict[str, Any]], str]


@dataclass(frozen=True)
class MapReduceSpec:
    """Everything task-specific; the orchestrator is the generic part.

    - `task_name`: job type used to enqueue the children (== handler name).
    - `leaf_fn` / `reduce_fn`: process one chunk / merge partials.
    - `carry_fields`: payload keys that travel to the children, into the `state`
      and into the merge (e.g. `targetLanguage`, `sourceLanguage`). Copied only
      when their value is not `None`.
    - `chunk_field`: payload key holding the text to chunk.
    - `units_filters`: names of the pre-chunking filters (or bundles) to run in
      order, only at the root (never on children, which already receive a single
      chunk). Resolved by `lib.llm.unit_filters`; `cfg["units_filters"]` overrides
      this list, and a filter returning an empty list keeps its input (fail-open).
    - `recursive_merge`: if the merge exceeds `chunk_word_budget *
      merge_recursion_factor` words, fan out over it again.
    - `child_max_steps`: `agent_max_steps` of each child job (leaves are
      single-step).
    """

    task_name: str
    leaf_fn: LeafFn
    reduce_fn: ReduceFn
    carry_fields: Tuple[str, ...] = ()
    chunk_field: str = "content"
    units_filters: Sequence[str] = ()
    recursive_merge: bool = True
    child_max_steps: int = 1


def run_map_reduce(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]],
    ctx,
    *,
    spec: MapReduceSpec,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Entry point of the reentrant handler.

    `state` is `None` on the first invocation (plan/leaf phase) and populated by
    the dispatcher with `phase == "merging"` when the parent is woken after its
    children finish.
    """
    if state and state.get("phase") == "merging":
        return _merge(state, ctx, spec=spec, cfg=cfg)
    return _plan_or_leaf(payload, ctx, spec=spec, cfg=cfg)


def _carry(source: Dict[str, Any], spec: MapReduceSpec) -> Dict[str, Any]:
    return {k: source[k] for k in spec.carry_fields if source.get(k) is not None}


def _plan_or_leaf(
    payload: Dict[str, Any], ctx, *, spec: MapReduceSpec, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    is_child = "_chunk_idx" in payload
    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))

    units_filter = None
    if not is_child:
        units_filter = build_units_filter(spec.units_filters, payload, cfg)

    chunks = build_chunks(
        payload.get(spec.chunk_field, ""), chunk_word_budget, units_filter=units_filter
    )
    if not chunks:
        return {"response": ""}

    if len(chunks) == 1:
        return {"response": spec.leaf_fn(chunks[0], payload, cfg)}

    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        # No job queue (e.g. unit tests): process the chunks in-process and
        # merge, without fan-out.
        partials = [spec.leaf_fn(c, payload, cfg) for c in chunks]
        return {"response": spec.reduce_fn(partials, payload, cfg)}

    carry = _carry(payload, spec)

    pending: Dict[str, int] = {}
    results: Dict[str, Optional[str]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        child_payload = {spec.chunk_field: chunk, "_chunk_idx": i, **carry}
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id, spec.task_name,
            payload=child_payload, agent_max_steps=spec.child_max_steps,
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
        "chunk_field": spec.chunk_field,
        "chunk_payload_template": dict(carry),
        **carry,
    }

    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _merge(
    state: Dict[str, Any], ctx, *, spec: MapReduceSpec, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    n = int(state.get("chunks_count", 0))
    results = state.get("results") or {}

    partials: List[str] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            partials.append(r.get("response") or "")
        elif isinstance(r, str):
            partials.append(r)
        else:
            partials.append("")

    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return {
            "error": (
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            )
        }

    valid_partials = [p for p in partials if p]
    if not valid_partials:
        return {"error": "no chunk results available to merge"}

    ctx_payload = _carry(state, spec)
    merged = spec.reduce_fn(valid_partials, ctx_payload, cfg)

    if spec.recursive_merge:
        chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
        factor = float(cfg.get("merge_recursion_factor", 2))
        if word_count(merged) > chunk_word_budget * factor:
            return _plan_or_leaf(
                {**ctx_payload, spec.chunk_field: merged}, ctx, spec=spec, cfg=cfg,
            )

    return {"response": merged}
