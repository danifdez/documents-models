from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import (
    _combine_section_summaries,
    _target_language,
    _write_summary,
)


@execution_handler("summarize-compose")
def summarize_compose(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-compose requires section summaries")
    summaries = _combine_section_summaries(partials)
    if not summaries:
        raise ValueError("summarize-compose received no section summaries")
    return {
        "response": _write_summary(
            summaries,
            _target_language(payload),
            get_task_config("summarize-compose"),
        )
    }
