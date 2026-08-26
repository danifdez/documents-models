import importlib
import inspect
import logging
from typing import Any, Callable, Dict, Optional

from common.execution_registry import TASK_HANDLERS

logger = logging.getLogger(__name__)

TASK_MODULES = {
    "date-extraction-map": "tasks.dates.dates",
    "date-extraction-reduce": "tasks.dates.dates",
    "document-extraction": "tasks.extraction.extractor",
    "entity-extraction-map": "tasks.entities.entities",
    "entity-extraction-reduce": "tasks.entities.entities",
    "keywords-map": "tasks.keywords.keywords",
    "keywords-reduce": "tasks.keywords.keywords",
    "indexed-file-extraction": "tasks.extraction.extractor",
    "indexed-file-ingest": "tasks.indexed_file.indexed_file",
    "indexed-file-search": "tasks.indexed_file.indexed_file",
    "ingest-content": "tasks.ingest.ingest",
    "key-point-map": "tasks.key_points.key_points",
    "key-point-reduce": "tasks.key_points.key_points",
    "assistant-chat": "tasks.assistant_chat.assistant_chat",
    "agent-chat": "tasks.assistant_chat.assistant_chat",
    "context-input-map": "tasks.context_input_map.context_input_map",
    "context-input-reduce": "tasks.context_input_reduce.context_input_reduce",
    "dataset.extract-row": "tasks.dataset_extraction.handler",
    "dataset.propose-columns": "tasks.dataset_extraction.propose_columns",
}


def call_handler(
    handler: Callable,
    payload: Dict[str, Any],
    *,
    state: Optional[Dict[str, Any]] = None,
):
    try:
        parameters = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return handler(payload)

    kwargs = {"state": state} if "state" in parameters else {}
    return handler(payload, **kwargs)


def ensure_task_handler(task_type: str) -> bool:
    if not task_type:
        return False
    if task_type in TASK_HANDLERS:
        return True

    module_name = task_type.replace("-", "_").replace(".", "_")
    candidates = (
        *([TASK_MODULES[task_type]] if task_type in TASK_MODULES else []),
        f"tasks.{module_name}.{module_name}",
        f"tasks.{module_name}.handler",
        f"tasks.{module_name}",
    )
    for candidate in candidates:
        try:
            importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            if error.name == candidate or candidate.startswith(f"{error.name}."):
                continue
            logger.exception("Task %s has a missing dependency", task_type)
            return False
        except Exception:
            logger.exception("Failed to load task %s from %s", task_type, candidate)
            return False
        if task_type in TASK_HANDLERS:
            return True
    return False
