from typing import Any, Dict

from common.execution_registry import execution_handler
from tasks.summarize.summarize import _combine_summary_lists


@execution_handler("summarize-finalize")
def summarize_finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-finalize requires summary partials")
    if (
        any(not isinstance(partial, list) for partial in partials)
        or any(
            not isinstance(summary, str) or not summary.strip()
            for partial in partials
            for summary in partial
        )
    ):
        raise ValueError("summarize-finalize requires non-empty summary lists")
    summaries = _combine_summary_lists(partials)
    if payload.get("final", False) is True:
        return {"response": "\n\n".join(summaries)}
    return {"responses": summaries}
