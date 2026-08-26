import hashlib
from typing import Any, Dict

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from services.llm_service import get_llm_service


@execution_handler("context-input-map")
def context_input_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    expected_hash = payload.get("contentHash")
    chunk_index = payload.get("chunkIndex")
    config = get_task_config("context-input-map")
    if (
        not isinstance(content, str)
        or not content
        or len(content) > int(config.get("max_input_chars", 12000))
        or not isinstance(chunk_index, int)
        or chunk_index < 0
        or expected_hash != _content_hash(content)
    ):
        raise ValueError("Invalid context input chunk")

    llm = get_llm_service(**get_llm_params("context-input-map"))
    digest = llm.chat(
        [
            {"role": "system", "content": get_prompt("context-input-map")},
            {"role": "user", "content": content},
        ],
        max_tokens=int(config.get("max_tokens", 700)),
        inference_name="context-input-map",
    )
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Context input map returned an empty digest")
    return {"digest": digest.strip()}


def _content_hash(content: str) -> str:
    value = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{value}"
