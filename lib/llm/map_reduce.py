"""In-process map-reduce helpers for content tasks.

Durable fan-out belongs to the Backend coordinator. A Models step receives
self-contained work and returns one result without creating executions or
writing execution state in PostgreSQL.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

LeafFn = Callable[[str, Dict[str, Any], Dict[str, Any]], Any]
ReduceFn = Callable[[List[Any], Dict[str, Any], Dict[str, Any]], Any]
ChunksFn = Callable[[Dict[str, Any], Dict[str, Any], bool], List[str]]
LeafPayloadExtrasFn = Callable[
    [List[str], Dict[str, Any], Dict[str, Any]],
    Optional[List[Dict[str, Any]]],
]


@dataclass(frozen=True)
class InlineListMapReduceSpec:
    leaf_fn: LeafFn
    reduce_fn: ReduceFn
    chunks_fn: ChunksFn
    result_key: str
    leaf_payload_extras_fn: Optional[LeafPayloadExtrasFn] = None


def run_inline_list_map_reduce(
    payload: Dict[str, Any],
    *,
    spec: InlineListMapReduceSpec,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Map chunks to list results and reduce them within one worker step."""
    is_chunk = "_chunk_idx" in payload
    chunks = spec.chunks_fn(payload, cfg, is_chunk)
    if not chunks:
        return {spec.result_key: []}

    extras = None
    if spec.leaf_payload_extras_fn is not None:
        extras = spec.leaf_payload_extras_fn(chunks, payload, cfg)

    partials: List[Any] = []
    for index, chunk in enumerate(chunks):
        leaf_payload = dict(payload)
        if extras is not None:
            leaf_payload.update(extras[index])
        partials.append(spec.leaf_fn(chunk, leaf_payload, cfg))

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
