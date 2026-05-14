"""Driver of one agent step. One Job execution = one step."""

import json
import logging
from typing import Any, Dict, Optional

from agent.llm import get_llm_for_spec
from agent.parse import parse_decision
from agent.prompt import render_messages
from agent.tools.base import TOOL_REGISTRY
from agent.types import AgentDefinition, StepOutcome, ToolContext

logger = logging.getLogger(__name__)


def _truncate(value: Any, limit: int = 1000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...(truncated)"
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return value
    return {"_truncated": text[:limit] + "...(truncated)"}


def _init_state(agent_def: AgentDefinition, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "payload": payload or {},
        "transcript": [],
        "depth": 0,
    }


def _finalize(
    job: Dict[str, Any],
    state: Dict[str, Any],
    db,
    *,
    reason: str,
    result: Optional[Dict[str, Any]] = None,
) -> StepOutcome:
    final = dict(result or {})
    final["_agent"] = {
        "reason": reason,
        "iterations": job.get("agent_iteration", 0) + 1,
        "transcript_len": len(state.get("transcript", [])),
    }
    db.update_job_result(job["id"], final)
    db.update_job_status(job["id"], "processed")
    logger.info("Agent job %s finalized (reason=%s)", job["id"], reason)
    return StepOutcome.FINISHED


def run_one_step(job: Dict[str, Any], agent_def: AgentDefinition, db) -> StepOutcome:
    """Execute exactly one agent step against the given Job row."""
    state: Dict[str, Any] = job.get("agent_state") or _init_state(agent_def, job.get("payload"))
    iteration = int(job.get("agent_iteration") or 0)
    max_steps = int(job.get("agent_max_steps") or agent_def.max_steps)

    if state.get("waiting_for_child"):
        state.pop("waiting_for_child", None)

    if iteration >= max_steps:
        return _finalize(job, state, db, reason="max_steps", result={"partial": state.get("transcript", [])[-1] if state.get("transcript") else {}})

    if not agent_def.model:
        logger.error("Agent %s has no model configured", agent_def.name)
        db.update_job_status(job["id"], "failed")
        return StepOutcome.FINISHED

    messages = render_messages(agent_def, state)
    llm = get_llm_for_spec(agent_def.model)
    raw = llm.chat(messages, max_tokens=600)
    decision = parse_decision(raw)

    if decision is None:
        observation = {"error": "Could not parse a JSON decision from the model output", "raw_excerpt": (raw or "")[:300]}
        state.setdefault("transcript", []).append({
            "step": iteration,
            "tool": None,
            "args": None,
            "observation": observation,
        })
        db.update_agent_progress(job["id"], iteration + 1, state)
        db.update_job_status(job["id"], "pending")
        return StepOutcome.CONTINUE

    if "finish" in decision:
        return _finalize(job, state, db, reason="finish", result=decision["finish"] if isinstance(decision["finish"], dict) else {"value": decision["finish"]})

    tool_name = decision.get("tool")
    args = decision.get("args") or {}
    thought = decision.get("thought")
    spec = TOOL_REGISTRY.get(tool_name) if tool_name else None

    if spec is None:
        observation = {"error": f"Unknown or unspecified tool: {tool_name}"}
    elif spec.name not in agent_def.tools:
        observation = {"error": f"Tool '{spec.name}' is not available to agent '{agent_def.name}'"}
    else:
        tool_defaults = agent_def.tool_defaults.get(spec.name, {})
        merged_args = {**tool_defaults, **args}
        ctx = ToolContext(
            job_id=job["id"],
            job_type=job["type"],
            payload=job.get("payload") or {},
            agent_def=agent_def,
            state=state,
        )
        try:
            observation = spec.run(merged_args, ctx)
        except Exception as e:
            logger.exception("Tool %s raised", spec.name)
            observation = {"error": f"{type(e).__name__}: {e}"}

    state.setdefault("transcript", []).append({
        "step": iteration,
        "thought": thought,
        "tool": tool_name,
        "args": args,
        "observation": _truncate(observation, 1000),
    })

    if isinstance(observation, dict) and observation.get("_sub_agent_pending"):
        state["waiting_for_child"] = True
        db.update_agent_progress(job["id"], iteration + 1, state)
        db.update_job_status(job["id"], "waiting")
        return StepOutcome.WAITING

    db.update_agent_progress(job["id"], iteration + 1, state)
    db.update_job_status(job["id"], "pending")
    return StepOutcome.CONTINUE
