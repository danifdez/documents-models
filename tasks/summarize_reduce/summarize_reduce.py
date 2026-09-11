from typing import Any, Dict

from common.execution_registry import execution_handler
from tasks.summarize.summarize import _combine_idea_lists


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
    ideas = _combine_idea_lists(partials)
    return {"ideas": ideas}
