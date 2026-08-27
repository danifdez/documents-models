import json
from typing import Any, Dict, List

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params
from services.llm_service import get_llm_service

_ALLOWED_ROLES = {"system", "user", "assistant"}
_SAMPLING_FIELDS = {
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "seed",
}


def _request(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("requestJson")
    if not isinstance(raw, str) or not raw:
        raise ValueError("requestJson is required")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("browser inference request must be an object")
    return request


def _messages(request: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = request.get("messages")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
        raise ValueError("messages must contain between 1 and 64 items")
    messages = []
    total = 0
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in _ALLOWED_ROLES or not isinstance(content, str):
            raise ValueError("each message requires a supported role and text content")
        total += len(content)
        messages.append({"role": role, "content": content})
    if total > 220_000:
        raise ValueError("browser inference context is too large")
    return messages


@execution_handler("browser-inference")
def browser_inference(payload: Dict[str, Any]) -> Dict[str, str]:
    request = _request(payload)
    messages = _messages(request)
    max_tokens = request.get("max_tokens", 1024)
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be between 1 and 8192")

    response_format = request.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")

    sampling = {
        key: request[key]
        for key in _SAMPLING_FIELDS
        if isinstance(request.get(key), (int, float))
    }
    template_kwargs = request.get("chat_template_kwargs")
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")

    llm = get_llm_service(**get_llm_params("browser-inference"))
    content = llm.chat(
        messages,
        max_tokens=max_tokens,
        response_format=response_format,
        allow_thinking=True,
        inference_name=str(payload.get("label") or "browser-inference"),
        sampling_overrides=sampling,
        chat_template_kwargs=template_kwargs,
    )
    return {"content": content}
