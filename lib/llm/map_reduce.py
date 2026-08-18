"""Task-agnostic reentrant map-reduce orchestrator.

Extracts the skeleton that used to live inside `tasks/summarize/summarize.py`
so any content task (summarize, keywords, key-point, …) can reuse the same
pattern: chunk the input, process each chunk in parallel through the job system
(fan-out) and merge the partial results (reduce), with optional recursion when
the merge is still too large.

A task is defined by declaring a `MapReduceSpec` with two functions —`leaf_fn`
(how to process one chunk) and `reduce_fn` (how to merge partials)— and calling
`run_map_reduce(payload, state, ctx, spec=..., cfg=...)` from its `@job_handler`.

Two result models coexist:

- Default (string) mode: `leaf_fn`/`reduce_fn` produce text and every result is
  `{"response": <str>}` — the original summarize behaviour.
- `list_results` mode: `leaf_fn` returns a list, results are wrapped under
  `result_key` (`{"dates": [...]}`, `{"keywords": [...]}`, …), a single-chunk
  root still runs `reduce_fn` (so the task's cross-chunk pipeline also applies
  to one chunk), children never fan out again, and the merge hands `reduce_fn`
  the raw per-chunk lists without any validity filtering.

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

# (chunk_text, payload, cfg) -> processed result for that chunk: text in the
# default mode, a list in `list_results` mode.
LeafFn = Callable[[str, Dict[str, Any], Dict[str, Any]], Any]
# (partials, payload, cfg) -> merged result.
ReduceFn = Callable[[List[Any], Dict[str, Any], Dict[str, Any]], Any]
# (payload, cfg, is_child) -> chunks.
ChunksFn = Callable[[Dict[str, Any], Dict[str, Any], bool], List[str]]
# (payload, cfg) -> static fields for every child payload (and the template).
ChildStaticFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
# (chunks, payload, cfg) -> (per-chunk extra payloads or None, extra state keys).
FanoutExtrasFn = Callable[
    [List[str], Dict[str, Any], Dict[str, Any]],
    Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]],
]
# (state) -> context payload handed to reduce_fn in the merge phase.
MergePayloadFn = Callable[[Dict[str, Any]], Dict[str, Any]]


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
      Ignored when `chunks_fn` is set (the custom pipeline owns its filtering).
    - `recursive_merge`: if the merge exceeds `chunk_word_budget *
      merge_recursion_factor` words, fan out over it again. Only honoured in the
      default string mode; `list_results` tasks never recurse.
    - `child_max_steps`: `agent_max_steps` of each child job (leaves are
      single-step).

    Optional extensions (defaults preserve the original string behaviour):

    - `result_key`: key wrapping every result (`"response"` by default). Also
      the key the merge phase reads from each stored child result.
    - `empty_value`: value returned under `result_key` when there is nothing to
      process.
    - `list_results`: switch to list semantics (see the module docstring).
    - `chunks_fn`: replaces the default `build_chunks` pipeline for tasks with
      their own cleaning/chunking. Called exactly once per plan invocation.
    - `child_static_fn`: static fields every child receives, also stored as the
      `chunk_payload_template`. Unlike `carry_fields`, the values are kept even
      when `None` and are placed before `_chunk_idx` in the child payload.
    - `fanout_extras_fn`: per-chunk extra payload entries appended after
      `_chunk_idx` (e.g. date-extraction's `_chunk_offset`) plus extra state
      keys inserted between `chunks` and `chunk_field`. Also applied to the
      in-process fallback so its leaves see the same extras.
    - `merge_payload_fn`: context payload handed to `reduce_fn` in the merge
      phase; defaults to the `carry_fields` extraction from the state. In
      `list_results` mode the payload `reduce_fn` receives always carries the
      chunk list under `"_chunks"`.
    """

    task_name: str
    leaf_fn: LeafFn
    reduce_fn: ReduceFn
    carry_fields: Tuple[str, ...] = ()
    chunk_field: str = "content"
    units_filters: Sequence[str] = ()
    recursive_merge: bool = True
    child_max_steps: int = 1
    result_key: str = "response"
    empty_value: Any = ""
    list_results: bool = False
    chunks_fn: Optional[ChunksFn] = None
    child_static_fn: Optional[ChildStaticFn] = None
    fanout_extras_fn: Optional[FanoutExtrasFn] = None
    merge_payload_fn: Optional[MergePayloadFn] = None


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

    if spec.chunks_fn is not None:
        chunks = spec.chunks_fn(payload, cfg, is_child)
    else:
        units_filter = None
        if not is_child:
            units_filter = build_units_filter(spec.units_filters, payload, cfg)
        chunks = build_chunks(
            payload.get(spec.chunk_field, ""), chunk_word_budget, units_filter=units_filter
        )
    if not chunks:
        return {spec.result_key: spec.empty_value}

    if spec.list_results and is_child:
        # Children never fan out again: when re-cleaning splits the received
        # chunk further (rare), process every piece against the payload the
        # child got and concatenate the per-piece lists.
        collected: List[Any] = []
        for chunk in chunks:
            collected.extend(spec.leaf_fn(chunk, payload, cfg))
        return {spec.result_key: collected}

    if len(chunks) == 1:
        if spec.list_results:
            # A single-chunk root still runs the task's full merge pipeline.
            merged = spec.reduce_fn(
                [spec.leaf_fn(chunks[0], payload, cfg)],
                {**payload, "_chunks": chunks},
                cfg,
            )
            return {spec.result_key: merged}
        return {spec.result_key: spec.leaf_fn(chunks[0], payload, cfg)}

    # Computed before the queue check: the in-process fallback needs the
    # per-chunk extras (e.g. chunk offsets) too.
    chunk_extras: Optional[List[Dict[str, Any]]] = None
    state_extras: Dict[str, Any] = {}
    if spec.fanout_extras_fn is not None:
        chunk_extras, state_extras = spec.fanout_extras_fn(chunks, payload, cfg)

    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        # No job queue (e.g. unit tests): process the chunks in-process and
        # merge, without fan-out.
        if spec.list_results:
            partials: List[Any] = []
            for i, chunk in enumerate(chunks):
                leaf_payload = {**payload, **chunk_extras[i]} if chunk_extras else payload
                partials.append(spec.leaf_fn(chunk, leaf_payload, cfg))
            merged = spec.reduce_fn(partials, {**payload, "_chunks": chunks}, cfg)
            return {spec.result_key: merged}
        partials = [spec.leaf_fn(c, payload, cfg) for c in chunks]
        return {spec.result_key: spec.reduce_fn(partials, payload, cfg)}

    carry = _carry(payload, spec)
    static = spec.child_static_fn(payload, cfg) if spec.child_static_fn is not None else None

    pending: Dict[str, int] = {}
    results: Dict[str, Optional[Any]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        if static is not None:
            child_payload = {spec.chunk_field: chunk, **static, "_chunk_idx": i}
        else:
            child_payload = {spec.chunk_field: chunk, "_chunk_idx": i, **carry}
        if chunk_extras is not None:
            child_payload.update(chunk_extras[i])
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
        **state_extras,
        "chunk_field": spec.chunk_field,
        "chunk_payload_template": dict(static) if static is not None else dict(carry),
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

    if spec.list_results:
        failed_idx = state.get("failed_idx")
        if failed_idx is not None:
            return {
                "error": (
                    f"chunk {failed_idx} failed after retries: "
                    f"{state.get('failed_error') or 'unknown error'}"
                )
            }
        partials: List[Any] = []
        for i in range(n):
            r = results.get(str(i))
            partials.append(r.get(spec.result_key) if isinstance(r, dict) else None)
        if spec.merge_payload_fn is not None:
            base = spec.merge_payload_fn(state)
        else:
            base = _carry(state, spec)
        ctx_payload = {**base, "_chunks": state.get("chunks") or []}
        return {spec.result_key: spec.reduce_fn(partials, ctx_payload, cfg)}

    partials = []
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

    return {spec.result_key: merged}
