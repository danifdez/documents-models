from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import _summarize_chunk, _target_language


@execution_handler("summarize-map")
def summarize_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("summarize-map requires non-empty content")
    return {
        "response": _summarize_chunk(
            content,
            _target_language(payload),
            get_task_config("summarize-map"),
        )
    }
