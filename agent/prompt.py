"""Renders the chat messages sent to the LLM at each agent step."""

import json
import os
from typing import Any, Dict, List

from agent.tools.base import TOOL_REGISTRY
from agent.types import AgentDefinition
from services.prompts import load_prompt


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_DECISION_INSTRUCTIONS = load_prompt(_PROMPTS_DIR, "decision_instructions.md").strip()


def _tool_catalog(agent_def: AgentDefinition) -> str:
    lines = []
    for tool_name in agent_def.tools:
        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            lines.append(f"- {tool_name}  (unknown tool, not registered)")
            continue
        args_desc = ", ".join(f"{k}: {v}" for k, v in spec.args_schema.items()) or "(none)"
        lines.append(f"- {spec.name} — {spec.description}  args: {args_desc}")
    return "\n".join(lines) if lines else "(no tools available)"


def _format_transcript(transcript: List[Dict[str, Any]], max_entries: int = 15) -> str:
    if not transcript:
        return "(no prior steps)"
    recent = transcript[-max_entries:]
    parts = []
    for entry in recent:
        step = entry.get("step")
        tool = entry.get("tool")
        args = entry.get("args")
        obs = entry.get("observation")
        thought = entry.get("thought")
        block = [f"Step {step}:"]
        if thought:
            block.append(f"  thought: {thought}")
        if tool:
            block.append(f"  tool: {tool}")
            block.append(f"  args: {json.dumps(args, ensure_ascii=False)[:400]}")
        if obs is not None:
            obs_text = json.dumps(obs, ensure_ascii=False)
            if len(obs_text) > 800:
                obs_text = obs_text[:800] + "...(truncated)"
            block.append(f"  observation: {obs_text}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def render_messages(agent_def: AgentDefinition, state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the chat messages for the next agent step."""
    payload = state.get("payload", {})
    transcript = state.get("transcript", [])

    system = (
        f"{agent_def.system_prompt.strip()}\n\n"
        f"Available tools:\n{_tool_catalog(agent_def)}\n\n"
        f"{_DECISION_INSTRUCTIONS}"
    )

    user_lines = [
        f"Input payload:\n{json.dumps(payload, ensure_ascii=False)[:4000]}",
        "",
        f"Transcript so far:\n{_format_transcript(transcript)}",
        "",
        "Decide the next single action. Respond with ONE JSON object only.",
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
