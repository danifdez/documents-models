"""Driver for one step of an agent execution."""

import json
import logging
from typing import Any, Dict, Optional

from agent.llm import get_llm_for_spec
from agent.parse import parse_decision
from agent.prompt import render_messages
from agent.tools.base import TOOL_REGISTRY
from agent.types import AgentDefinition, StepOutcome, ToolContext
from lib.framework.agent_protocol import (
    AgentRunResult,
    ModelOutcomeKind,
)
from lib.llm.grammars import AGENT_DECISION_GBNF, AGENT_FINISH_GBNF

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
    execution: Dict[str, Any],
    state: Dict[str, Any],
    db,
    *,
    reason: str,
    result: Optional[Dict[str, Any]] = None,
) -> StepOutcome:
    final = AgentRunResult.structured_result(
        dict(result or {})
    ).as_payload()
    final["_agent"] = {
        "reason": reason,
        "iterations": execution.get("step", 0) + 1,
        "transcript_len": len(state.get("transcript", [])),
    }
    attempt_id = execution.get("attempt_id")
    if not db.update_execution_result(execution["execution_id"], final, attempt_id=attempt_id):
        return StepOutcome.FINISHED
    db.update_execution_status(
        execution["execution_id"],
        "running",
        phase="backend_finalization",
        attempt_id=attempt_id,
    )
    logger.info("Agent execution %s finalized (reason=%s)", execution["execution_id"], reason)
    return StepOutcome.FINISHED


def _force_finish(
    execution: Dict[str, Any],
    state: Dict[str, Any],
    agent_def: AgentDefinition,
    db,
) -> StepOutcome:
    """Step budget exhausted: make one last grammar-constrained call asking
    the model to synthesize a final result from the transcript, instead of
    returning a raw transcript fragment to the caller."""
    transcript = state.get("transcript") or []
    fallback = {"partial": transcript[-1] if transcript else {}}
    if not agent_def.model:
        return _finalize(execution, state, db, reason="max_steps", result=fallback)
    try:
        messages = render_messages(agent_def, state)
        messages.append({
            "role": "user",
            "content": (
                "You have exhausted your step budget. Do not request any more "
                "tools. Emit the finish JSON now with the best result you can "
                "produce from the transcript so far."
            ),
        })
        llm = get_llm_for_spec(agent_def.model)
        raw = llm.chat(messages, max_tokens=600, grammar=AGENT_FINISH_GBNF, temperature=0.0)
        outcome = parse_decision(raw)
        if outcome.kind == ModelOutcomeKind.STRUCTURED_RESULT:
            return _finalize(
                execution,
                state,
                db,
                reason="max_steps_forced",
                result=outcome.value,
            )
    except Exception:
        logger.exception("Forced finish failed for agent execution %s", execution["execution_id"])
    return _finalize(execution, state, db, reason="max_steps", result=fallback)


def run_one_step(execution: Dict[str, Any], agent_def: AgentDefinition, db) -> StepOutcome:
    """Execute exactly one agent step against the given Execution row."""
    state: Dict[str, Any] = execution.get("checkpoint") or _init_state(agent_def, execution.get("payload"))
    iteration = int(execution.get("step") or 0)
    max_steps = int(execution.get("max_steps") or agent_def.max_steps)

    if state.get("waiting_for_child"):
        state.pop("waiting_for_child", None)

    if iteration >= max_steps:
        return _force_finish(execution, state, agent_def, db)

    if not agent_def.model:
        logger.error("Agent %s has no model configured", agent_def.name)
        db.update_execution_status(
            execution["execution_id"],
            "failed",
            attempt_id=execution.get("attempt_id"),
            failure_message=f"Agent {agent_def.name} has no model configured",
        )
        return StepOutcome.FINISHED

    messages = render_messages(agent_def, state)
    llm = get_llm_for_spec(agent_def.model)
    raw = llm.chat(messages, max_tokens=600, grammar=AGENT_DECISION_GBNF, temperature=0.0)
    outcome = parse_decision(raw)

    if outcome.kind == ModelOutcomeKind.INVALID:
        observation = {
            "error": outcome.reason or "invalid_model_decision",
            "raw_excerpt": (raw or "")[:300],
        }
        state.setdefault("transcript", []).append({
            "step": iteration,
            "tool": None,
            "args": None,
            "observation": observation,
        })
        attempt_id = execution.get("attempt_id")
        if db.update_agent_progress(execution["execution_id"], iteration + 1, state, attempt_id=attempt_id):
            db.update_execution_status(execution["execution_id"], "queued", attempt_id=attempt_id)
        return StepOutcome.CONTINUE

    if outcome.kind == ModelOutcomeKind.STRUCTURED_RESULT:
        return _finalize(
            execution,
            state,
            db,
            reason="finish",
            result=outcome.value,
        )

    request = outcome.tool_requests[0]
    tool_name = request.name
    args = request.arguments or {}
    thought = outcome.thought
    spec = TOOL_REGISTRY.get(tool_name) if tool_name else None

    transcript = state.get("transcript") or []
    last = transcript[-1] if transcript else None

    if last and request.same_operation(last.get("tool"), last.get("args")):
        # Don't burn a real tool execution on a stuck model; feed the loop
        # back as an observation so the next step changes course.
        observation = {
            "error": (
                "You repeated the exact same tool call as the previous step. "
                "Choose a different action, change the arguments, or finish."
            )
        }
    elif spec is None:
        observation = {"error": f"Unknown or unspecified tool: {tool_name}"}
    elif spec.name not in agent_def.tools:
        observation = {"error": f"Tool '{spec.name}' is not available to agent '{agent_def.name}'"}
    else:
        tool_defaults = agent_def.tool_defaults.get(spec.name, {})
        merged_args = {**tool_defaults, **args}
        ctx = ToolContext(
            execution_id=execution["execution_id"],
            task_type=execution["task_type"],
            payload=execution.get("payload") or {},
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
        attempt_id = execution.get("attempt_id")
        if db.update_agent_progress(execution["execution_id"], iteration + 1, state, attempt_id=attempt_id):
            db.update_execution_status(execution["execution_id"], "waiting", attempt_id=attempt_id)
        return StepOutcome.WAITING

    attempt_id = execution.get("attempt_id")
    if db.update_agent_progress(execution["execution_id"], iteration + 1, state, attempt_id=attempt_id):
        db.update_execution_status(execution["execution_id"], "queued", attempt_id=attempt_id)
    return StepOutcome.CONTINUE
