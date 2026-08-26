from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from services.llm_service import get_llm_service


@execution_handler("context-input-reduce")
def context_input_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    config = get_task_config("context-input-reduce")
    max_partials = int(config.get("max_partials", 8))
    max_partial_chars = int(config.get("max_partial_chars", 8000))
    if (
        not isinstance(partials, list)
        or not 1 <= len(partials) <= max_partials
        or any(
            not isinstance(partial, str)
            or not partial.strip()
            or len(partial) > max_partial_chars
            for partial in partials
        )
    ):
        raise ValueError("Invalid context input partials")

    ordered = "\n\n".join(
        f"[Part {index + 1}]\n{partial.strip()}"
        for index, partial in enumerate(partials)
    )
    llm = get_llm_service(**get_llm_params("context-input-reduce"))
    digest = llm.chat(
        [
            {"role": "system", "content": get_prompt("context-input-reduce")},
            {"role": "user", "content": ordered},
        ],
        max_tokens=int(config.get("max_tokens", 900)),
        inference_name="context-input-reduce",
    )
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Context input reduce returned an empty digest")
    return {"digest": digest.strip()}
