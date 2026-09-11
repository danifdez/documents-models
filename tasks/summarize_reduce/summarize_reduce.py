from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import (
    _combine_idea_lists,
    _write_summary,
)


@execution_handler("summarize-reduce")
def summarize_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-reduce requires idea partials")
    if (
        any(not isinstance(partial, list) or not partial for partial in partials)
        or any(
            not isinstance(idea, str) or not idea.strip()
            for partial in partials
            for idea in partial
        )
    ):
        raise ValueError("summarize-reduce requires non-empty idea lists")
    config = get_task_config("summarize-reduce")
    ideas = _combine_idea_lists(partials)
    if payload.get("final", True) is True:
        return {
            "response": _write_summary(
                ideas,
                config,
            )
        }
    return {"ideas": ideas}
