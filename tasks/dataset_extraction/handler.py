"""Dataset extraction worker.

Job type: `dataset.extract-row`.

Payload (set by backend `DatasetExtractionService` in T04):
    {
      "datasetId":         int,
      "recordId":          int,
      "resourceId":        int,
      "projectId":         int,
      "schema":            list[DatasetField],   # already filtered to fields with description
      "columnsToExtract":  list[str],            # subset of schema keys; [] means "all"
      "documentText":      str,                  # Resource.content
      "sourceTitle":       str,
      "isAudio":           bool,                 # mimeType startswith "audio/"
      "model":             str | None            # optional model override
    }

Result (consumed by backend job processor in T04):
    {
      "data":         dict[str, Any] | None,         # fieldKey -> value (or null)
      "cellMetadata": dict[str, CellAnchor] | None,  # fieldKey -> anchor; omitted when value is null
      "error":        str | None,                    # set when extraction failed
      "model":        str,                           # model used (echoed back for persistence)
      "promptVersion": str
    }

The backend (T04) is responsible for writing `data` + `cellMetadata` onto the
DatasetRecord and flipping `extraction_status` to 'extracted' or 'failed'.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.llm import get_llm_for_spec
from agent.types import ModelSpec
from lib.llm.json import parse_json
from lib.llm.config import get_llm_defaults, get_task_config
from utils.job_registry import job_handler

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


@job_handler("dataset.extract-row")
def extract_dataset_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_task_config("dataset.extract-row") or {}
    schema: List[Dict[str, Any]] = payload.get("schema") or []
    columns_to_extract: List[str] = payload.get("columnsToExtract") or []
    document_text: str = payload.get("documentText") or ""
    source_title: str = payload.get("sourceTitle") or ""
    resource_id = int(payload.get("resourceId") or 0)
    is_audio: bool = bool(payload.get("isAudio") or False)
    model_override: Optional[str] = payload.get("model")

    if not document_text.strip():
        return {
            "data": None,
            "cellMetadata": None,
            "error": "Resource has no extracted content",
            "model": model_override or cfg.get("model") or "",
            "promptVersion": PROMPT_VERSION,
        }

    fields = _filter_fields(schema, columns_to_extract)
    if not fields:
        return {
            "data": {},
            "cellMetadata": {},
            "error": None,
            "model": model_override or cfg.get("model") or "",
            "promptVersion": PROMPT_VERSION,
        }

    model_name = model_override or cfg.get(
        "model") or get_llm_defaults().get("model")
    if not model_name:
        return {
            "data": None,
            "cellMetadata": None,
            "error": "No model configured for dataset-extraction",
            "model": "",
            "promptVersion": PROMPT_VERSION,
        }

    max_tokens = int(cfg.get("max_tokens", 2048))
    temperature = float(cfg.get("temperature", 0.1))
    seed = cfg.get("seed", 42)
    seed = int(seed) if seed is not None else None

    budget = _char_budget(cfg, max_tokens)
    safe_text, truncated = _truncate(document_text, budget)
    if truncated:
        logger.warning(
            "dataset.extract-row: truncated document at %d chars (resourceId=%s)",
            budget, resource_id,
        )

    grammar = build_grammar(fields)
    prompt = build_prompt(fields, safe_text, source_title, is_audio=is_audio)

    spec = ModelSpec(path=model_name)
    llm = get_llm_for_spec(spec)

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
    except Exception as exc:  # noqa: BLE001 — surface as job failure, not crash
        logger.exception("dataset.extract-row: LLM generation failed")
        raw = f"<llm error: {exc}>"
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
            if not isinstance(parsed, dict):
                logger.error(
                    "dataset.extract-row: retry without grammar also failed. raw=%r",
                    (raw_retry or "")[:200],
                )
                return {
                    "data": None,
                    "cellMetadata": None,
                    "error": "LLM returned invalid JSON after retry",
                    "model": model_name,
                    "promptVersion": PROMPT_VERSION,
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception("dataset.extract-row: retry failed")
            return {
                "data": None,
                "cellMetadata": None,
                "error": f"LLM retry failed: {exc}",
                "model": model_name,
                "promptVersion": PROMPT_VERSION,
            }

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
        "error": None,
        "model": model_name,
        "promptVersion": PROMPT_VERSION,
    }
