from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from tasks.summarize.summarize import (
    _merge_idea_lists,
    _target_language,
    _write_summary,
)


@execution_handler("summarize-reduce")
def summarize_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-reduce requires idea partials")
    final = payload.get("final", True) is True
    if final:
        if all(isinstance(partial, list) for partial in partials):
            ideas = [idea for partial in partials for idea in partial]
        elif all(isinstance(partial, str) for partial in partials):
            ideas = partials
        else:
            raise ValueError("summarize-reduce requires compatible idea partials")
        if any(not isinstance(idea, str) or not idea.strip() for idea in ideas):
            raise ValueError("summarize-reduce requires non-empty ideas")
        return {
            "response": _write_summary(
                ideas,
                _target_language(payload),
                get_task_config("summarize-reduce"),
            )
        }
    if (
        any(not isinstance(partial, list) or not partial for partial in partials)
        or any(
            not isinstance(idea, str) or not idea.strip()
            for partial in partials
            for idea in partial
        )
    ):
        raise ValueError("summarize-reduce requires non-empty idea lists")
    return {
        "ideas": _merge_idea_lists(
            partials,
            _target_language(payload),
            get_task_config("summarize-reduce"),
        )
    }
