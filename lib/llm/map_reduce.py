"""In-process map-reduce helpers for content tasks.

Durable fan-out belongs to the Backend coordinator. A Models step receives
self-contained work and returns one result without creating executions or
writing execution state in PostgreSQL.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.llm.text import build_chunks, word_count
from lib.llm.unit_filters import build_units_filter

LeafFn = Callable[[str, Dict[str, Any], Dict[str, Any]], Any]
ReduceFn = Callable[[List[Any], Dict[str, Any], Dict[str, Any]], Any]
ChunksFn = Callable[[Dict[str, Any], Dict[str, Any], bool], List[str]]
ChildStaticFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
FanoutExtrasFn = Callable[
    [List[str], Dict[str, Any], Dict[str, Any]],
    Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]],
]
MergePayloadFn = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class MapReduceSpec:
    task_name: str
    leaf_fn: LeafFn
    reduce_fn: ReduceFn
    carry_fields: Tuple[str, ...] = ()
    chunk_field: str = "content"
    units_filters: Sequence[str] = ()
    recursive_merge: bool = True
    result_key: str = "response"
    empty_value: Any = ""
    list_results: bool = False
    chunks_fn: Optional[ChunksFn] = None
    child_static_fn: Optional[ChildStaticFn] = None
    fanout_extras_fn: Optional[FanoutExtrasFn] = None
    merge_payload_fn: Optional[MergePayloadFn] = None


def run_map_reduce(
    payload: Dict[str, Any],
    _state: Optional[Dict[str, Any]],
    _ctx,
    *,
    spec: MapReduceSpec,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return _run_inline(payload, spec=spec, cfg=cfg)


def _run_inline(
    payload: Dict[str, Any],
    *,
    spec: MapReduceSpec,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    is_chunk = "_chunk_idx" in payload
    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    if spec.chunks_fn is not None:
        chunks = spec.chunks_fn(payload, cfg, is_chunk)
    else:
        units_filter = None
        if not is_chunk:
            units_filter = build_units_filter(spec.units_filters, payload, cfg)
        chunks = build_chunks(
            payload.get(spec.chunk_field, ""),
            chunk_word_budget,
            units_filter=units_filter,
        )
    if not chunks:
        return {spec.result_key: spec.empty_value}

    extras = None
    if spec.fanout_extras_fn is not None:
        extras, _ = spec.fanout_extras_fn(chunks, payload, cfg)

    partials: List[Any] = []
    for index, chunk in enumerate(chunks):
        leaf_payload = dict(payload)
        if extras is not None:
            leaf_payload.update(extras[index])
        partials.append(spec.leaf_fn(chunk, leaf_payload, cfg))

    if spec.list_results:
        if is_chunk:
            return {
                spec.result_key: [
                    item for partial in partials for item in partial
                ]
            }
        merge_payload = {**payload, "_chunks": chunks}
        return {
            spec.result_key: spec.reduce_fn(partials, merge_payload, cfg)
        }

    if len(partials) == 1:
        return {spec.result_key: partials[0]}

    merged = spec.reduce_fn(partials, payload, cfg)
    if spec.recursive_merge:
        factor = float(cfg.get("merge_recursion_factor", 2))
        if word_count(merged) > chunk_word_budget * factor:
            return _run_inline(
                {**payload, spec.chunk_field: merged}, spec=spec, cfg=cfg
            )
    return {spec.result_key: merged}
