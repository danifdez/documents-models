"""Dataset extraction worker.

Execution types: `dataset.extract-row-map` and `dataset.extract-row-reduce`.
`dataset.extract-row` remains available for direct, bounded assignments.

Payload (set by backend `DatasetExtractionService`):
    {
      "datasetId":         int,
      "recordId":          int,
      "resourceId":        int,
      "projectId":         int,
      "schema":            list[DatasetField],   # already filtered to fields with description
      "columnsToExtract":  list[str],            # subset of schema keys; [] means "all"
      "documentText":      str,                  # bounded document chunk
      "chunkIndex":        int,                  # map assignments only
      "sourceTitle":       str,
      "isAudio":           bool,                 # mimeType startswith "audio/"
      "model":             str | None            # optional model override
    }

Result (consumed by backend `DatasetExtractionProcessor`):
    {
      "data":         dict[str, Any],                # fieldKey -> value (or null)
      "cellMetadata": dict[str, CellAnchor],         # fieldKey -> anchor; omitted when value is null
      "model":        str,                           # model used (echoed back for persistence)
      "promptVersion": str
    }

The backend (T04) is responsible for writing `data` + `cellMetadata` onto the
DatasetRecord and flipping `extraction_status` to 'extracted' or 'failed'.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from lib.llm.json import parse_json
from lib.llm.config import get_llm_defaults, get_llm_params, get_task_config
from services.llm_service import get_llm_service
from common.execution_registry import execution_handler

from .grammar import build_grammar
from .prompt import PROMPT_VERSION, build_prompt

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _filter_fields(schema: List[Dict[str, Any]], columns_to_extract: List[str]) -> List[Dict[str, Any]]:
    """Keep only fields that (a) have a non-empty description and (b) are in the requested subset.

    `columns_to_extract` empty means "all fields with a description".
    """
    wanted = set(columns_to_extract or [])
    out: List[Dict[str, Any]] = []
    for field in schema:
        if not field.get("description"):
            continue
        if wanted and field["key"] not in wanted:
            continue
        out.append(field)
    return out


def _char_budget(cfg: Dict[str, Any], max_tokens: int) -> int:
    n_ctx = int(cfg.get("n_ctx", get_llm_defaults().get("n_ctx", 32768)))
    # Reserve room for the prompt boilerplate (~1500 tokens) and the response.
    available_tokens = max(1024, n_ctx - max_tokens - 1500)
    return available_tokens * 4


def _truncate(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    return text[:budget] + "\n\n[...document truncated for length...]\n", True


def _coerce_value(value: Any, ftype: str) -> Any:
    if value is None:
        return None
    if ftype == "number":
        try:
            n = float(value)
            return int(n) if n.is_integer() else n
        except (TypeError, ValueError):
            return None
    if ftype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "true":
                return True
            if v == "false":
                return False
        return None
    return value


def _build_result(
    fields: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    resource_id: int,
    model_name: str,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    cell_metadata: Dict[str, Any] = {}
    extracted_at = _now_iso()

    for field in fields:
        key = field["key"]
        entry = parsed.get(key) or {}
        raw_value = entry.get("value") if isinstance(entry, dict) else None
        value = _coerce_value(raw_value, field.get("type", "text"))
        data[key] = value

        if value is None:
            # Hard rule: no anchor for null values.
            continue

        quote = entry.get("_quote") if isinstance(entry, dict) else ""
        page = entry.get("_page") if isinstance(entry, dict) else None
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                page = None
        cell_metadata[key] = {
            "sourceResourceId": resource_id,
            "page": page if isinstance(page, int) else None,
            "quote": quote if isinstance(quote, str) else "",
            "extractedAt": extracted_at,
            "model": model_name,
            "promptVersion": PROMPT_VERSION,
            "editedByUser": False,
        }

    return {"data": data, "cellMetadata": cell_metadata}


def _extract_dataset_row(
    payload: Dict[str, Any], task_type: str
) -> Dict[str, Any]:
    cfg = (
        get_task_config(task_type)
        or get_task_config("dataset.extract-row")
        or {}
    )
    schema: List[Dict[str, Any]] = payload.get("schema") or []
    columns_to_extract: List[str] = payload.get("columnsToExtract") or []
    document_text: str = payload.get("documentText") or ""
    source_title: str = payload.get("sourceTitle") or ""
    resource_id = int(payload.get("resourceId") or 0)
    is_audio: bool = bool(payload.get("isAudio") or False)
    model_override: Optional[str] = payload.get("model")

    if not document_text.strip():
        raise ValueError("Resource has no extracted content")

    fields = _filter_fields(schema, columns_to_extract)
    if not fields:
        return {
            "data": {},
            "cellMetadata": {},
            "model": model_override or cfg.get("model") or "",
            "promptVersion": PROMPT_VERSION,
        }

    model_name = model_override or cfg.get(
        "model") or get_llm_defaults().get("model")
    if not model_name:
        raise RuntimeError("No model configured for dataset extraction")

    max_tokens = int(cfg.get("max_tokens", 2048))
    temperature = float(cfg.get("temperature", 0.1))
    seed = cfg.get("seed", 42)
    seed = int(seed) if seed is not None else None

    budget = _char_budget(cfg, max_tokens)
    safe_text, truncated = _truncate(document_text, budget)
    if truncated:
        logger.warning(
            "%s: truncated document at %d chars (resourceId=%s)",
            task_type,
            budget,
            resource_id,
        )

    grammar = build_grammar(fields)
    prompt = build_prompt(fields, safe_text, source_title, is_audio=is_audio)

    llm = get_llm_service(**get_llm_params(task_type, model_name))

    raw = ""
    parsed: Optional[Dict[str, Any]] = None
    try:
        raw = llm.generate(
            prompt,
            max_tokens=max_tokens,
            grammar=grammar,
            temperature=temperature,
            seed=seed,
        )
        parsed = parse_json(raw, default=None)
    except Exception:  # noqa: BLE001
        logger.exception("dataset.extract-row: LLM generation failed")
        parsed = None

    if not isinstance(parsed, dict):
        # One defensive retry without grammar — surfaces grammar bugs without
        # silently failing entire extractions when the grammar is wrong.
        try:
            raw_retry = llm.generate(
                prompt
                + "\n\nReturn only the JSON object described above, no markdown.",
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            parsed = parse_json(raw_retry, default=None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dataset.extract-row: retry failed")
            raise RuntimeError(f"LLM retry failed: {exc}") from exc
        if not isinstance(parsed, dict):
            logger.error(
                "dataset.extract-row: retry without grammar returned invalid JSON"
            )
            raise RuntimeError("LLM returned invalid JSON after retry")

    built = _build_result(fields, parsed, resource_id, model_name)
    logger.debug(
        "dataset.extract-row: extracted %d fields (resourceId=%s, recordId=%s)",
        len([v for v in built["data"].values() if v is not None]),
        resource_id,
        payload.get("recordId"),
    )

    return {
        "data": built["data"],
        "cellMetadata": built["cellMetadata"],
        "model": model_name,
        "promptVersion": PROMPT_VERSION,
    }


@execution_handler("dataset.extract-row")
def extract_dataset_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _extract_dataset_row(payload, "dataset.extract-row")


@execution_handler("dataset.extract-row-map")
def extract_dataset_row_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    chunk_index = payload.get("chunkIndex")
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("dataset.extract-row-map chunkIndex must be non-negative")
    result = _extract_dataset_row(payload, "dataset.extract-row-map")
    return {"candidates": [{"chunkIndex": chunk_index, **result}]}


def _candidate(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("dataset.extract-row-reduce candidate must be an object")
    chunk_index = raw.get("chunkIndex")
    data = raw.get("data")
    metadata = raw.get("cellMetadata")
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("dataset.extract-row-reduce candidate index is invalid")
    if not isinstance(data, dict) or not isinstance(metadata, dict):
        raise ValueError("dataset.extract-row-reduce candidate data is invalid")
    return {
        "chunkIndex": chunk_index,
        "data": data,
        "cellMetadata": metadata,
        "model": str(raw.get("model") or ""),
        "promptVersion": str(raw.get("promptVersion") or PROMPT_VERSION),
    }


def _grounded(candidate: Dict[str, Any], key: str) -> bool:
    metadata = candidate["cellMetadata"].get(key)
    return isinstance(metadata, dict) and bool(
        str(metadata.get("quote") or "").strip()
    )


def _merge_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(candidates, key=lambda item: item["chunkIndex"])
    keys: List[str] = []
    for candidate in ordered:
        for key in candidate["data"]:
            if key not in keys:
                keys.append(key)

    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key in keys:
        matches = [item for item in ordered if item["data"].get(key) is not None]
        grounded = [item for item in matches if _grounded(item, key)]
        selected = grounded or matches
        if not selected:
            data[key] = None
            continue
        winner = selected[0]
        data[key] = winner["data"][key]
        if key in winner["cellMetadata"]:
            metadata[key] = winner["cellMetadata"][key]

    first = ordered[0]
    return {
        "chunkIndex": first["chunkIndex"],
        "data": data,
        "cellMetadata": metadata,
        "model": first["model"],
        "promptVersion": first["promptVersion"],
    }


@execution_handler("dataset.extract-row-reduce")
def extract_dataset_row_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    final = payload.get("final")
    if not isinstance(partials, list) or not partials:
        raise ValueError("dataset.extract-row-reduce requires partials")
    if not isinstance(final, bool):
        raise ValueError("dataset.extract-row-reduce final must be a boolean")

    candidates: List[Dict[str, Any]] = []
    for partial in partials:
        if not isinstance(partial, list) or not partial:
            raise ValueError(
                "dataset.extract-row-reduce partials must be candidate arrays"
            )
        candidates.extend(_candidate(item) for item in partial)

    merged = _merge_candidates(candidates)
    if not final:
        return {"candidates": [merged]}
    return {
        "data": merged["data"],
        "cellMetadata": merged["cellMetadata"],
        "model": merged["model"],
        "promptVersion": merged["promptVersion"],
    }
