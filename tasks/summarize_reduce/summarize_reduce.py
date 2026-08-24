from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import _merge_summaries, _target_language


@execution_handler("summarize-reduce")
def summarize_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if (
        not isinstance(partials, list)
        or not partials
        or any(not isinstance(partial, str) for partial in partials)
    ):
        raise ValueError("summarize-reduce requires string partials")
    return {
        "response": _merge_summaries(
            partials,
            _target_language(payload),
            get_task_config("summarize-reduce"),
        )
    }
