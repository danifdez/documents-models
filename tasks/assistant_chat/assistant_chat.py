import json
from typing import Any, Dict, List
from uuid import uuid4

from common.execution_registry import execution_handler
from lib.execution.outcome import InferenceOutcome
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from services.llm_service import get_llm_service

_SYSTEM_PROMPT = get_prompt("assistant-chat").strip()
_DOCUMENT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "documents.search",
        "description": "Search documents available in the current workspace.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
    },
}


def _conversation(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(payload)}
    ]
    for message in payload.get("conversation") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    for batch in payload.get("toolHistory") or []:
        if not isinstance(batch, dict):
            continue
        calls = batch.get("calls") or []
        results = batch.get("results") or []
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call["toolCallId"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call.get("arguments") or {}, ensure_ascii=False
                            ),
                        },
                    }
                    for call in calls
                    if isinstance(call, dict)
                    and isinstance(call.get("toolCallId"), str)
                    and isinstance(call.get("name"), str)
                ],
            }
        )
        for result in results:
            if not isinstance(result, dict):
                continue
            tool_call_id = result.get("toolCallId")
            if not isinstance(tool_call_id, str):
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result.get("content") or json.dumps(
                        result.get("structuredContent"), ensure_ascii=False
                    ),
                }
            )
    return messages


def _system_prompt(payload: Dict[str, Any]) -> str:
    prompt = _SYSTEM_PROMPT
    configured = payload.get("systemPrompt")
    if isinstance(configured, str) and configured.strip():
        prompt = f"{prompt}\n\nAgent instructions:\n{configured.strip()}"
    folder_scope = payload.get("folderScope")
    if folder_scope:
        prompt = f"{prompt}\n\nWorkspace folder scope: {folder_scope}"
    return prompt


def _arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Tool arguments must be a JSON object")


def _outcome(message: Dict[str, Any], max_tool_calls: int) -> Dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        calls = []
        for raw in tool_calls[:max_tool_calls]:
            function = raw.get("function") if isinstance(raw, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if name != "documents.search":
                raise ValueError(f"Unsupported tool requested: {name}")
            calls.append(
                {
                    "toolCallId": str(uuid4()),
                    "name": name,
                    "arguments": _arguments(function.get("arguments")),
                }
            )
        if calls:
            return {"kind": "tool_requests", "calls": calls}

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned neither text nor tool requests")
    return {"kind": "final_text", "text": content.strip()}


@execution_handler("assistant-chat")
@execution_handler("agent-chat")
def chat_inference(payload: Dict[str, Any]) -> InferenceOutcome:
    task_type = str(payload.get("_task_type") or "assistant-chat")
    config = get_task_config(task_type)
    llm = get_llm_service(**get_llm_params(task_type))
    message = llm.chat_with_tools(
        _conversation(payload),
        [_DOCUMENT_SEARCH_TOOL],
        max_tokens=int(config.get("max_tokens", 1200)),
        tool_choice="auto",
        inference_name=task_type,
    )
    return InferenceOutcome(
        _outcome(message, max(1, int(config.get("max_tool_calls", 4))))
    )
