from typing import Any, Dict

from common.execution_registry import execution_handler
from tasks.summarize.summarize import _combine_section_summaries


@execution_handler("summarize-reduce")
def summarize_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("summarize-reduce requires summary partials")
    summaries = _combine_section_summaries(partials)
    if not summaries:
        raise ValueError("summarize-reduce received no section summaries")
    return {"summaries": summaries}
