from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import (
    _combine_idea_lists,
    _target_language,
    _write_summary,
)


@execution_handler("summarize-compose")
def summarize_compose(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-compose requires idea inventories")
    if (
        any(not isinstance(partial, list) for partial in partials)
        or any(
            not isinstance(idea, str) or not idea.strip()
            for partial in partials
            for idea in partial
        )
    ):
        raise ValueError("summarize-compose requires valid idea inventories")
    ideas = _combine_idea_lists(partials)
    if not ideas:
        raise ValueError("summarize-compose received no material ideas")
    return {
        "response": _write_summary(
            ideas,
            _target_language(payload),
            get_task_config("summarize-compose"),
        )
    }
