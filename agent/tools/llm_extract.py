"""LLM extraction tool. Optionally switches model/LoRA per call."""

from typing import Any, Dict, Optional

from agent.llm import get_llm_for_spec
from agent.tools.base import tool
from agent.types import ModelSpec, ToolContext
from services.llm_json import chat_json


def _resolve_model(args: Dict[str, Any], ctx: ToolContext) -> ModelSpec:
    """Args may carry an explicit model spec; otherwise fall back to agent's main model."""
    if "model" in args and args["model"] is not None:
        return ModelSpec.from_any(args["model"]) or ctx.agent_def.model
    if "path" in args:
        return ModelSpec(
            path=args["path"],
            lora=args.get("lora"),
            lora_scale=float(args.get("lora_scale", 1.0)),
        )
    if ctx.agent_def.model is None:
        raise ValueError("No model configured for the agent and none provided in tool args")
    return ctx.agent_def.model


@tool(
    name="llm_extract",
    description=(
        "Run a focused LLM call with a prompt and an expected JSON schema. "
        "Optionally targets a specific model or LoRA adapter for this single call."
    ),
    args_schema={
        "prompt": "string instruction for the LLM (required)",
        "schema_hint": "string describing the expected JSON shape (required)",
        "system": "optional system message; defaults to a strict JSON-only instruction",
        "max_tokens": "optional int (default 600)",
        "model": "optional {path, lora?, lora_scale?} to override per-call model",
    },
)
def llm_extract(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    prompt = args.get("prompt")
    schema_hint = args.get("schema_hint")
    if not prompt or not schema_hint:
        return {"error": "Both 'prompt' and 'schema_hint' are required"}

    system = args.get("system") or (
        "You are a JSON-only extractor. Respond with ONE JSON value matching the "
        "requested schema. No prose, no markdown fences."
    )
    max_tokens = int(args.get("max_tokens", 600))

    spec = _resolve_model(args, ctx)
    llm = get_llm_for_spec(spec)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{prompt}\n\nExpected JSON schema: {schema_hint}"},
    ]
    parsed = chat_json(llm, messages, schema_hint=schema_hint, max_tokens=max_tokens)
    if parsed is None:
        return {"error": "LLM did not return valid JSON after retries"}

    return {"data": parsed}
