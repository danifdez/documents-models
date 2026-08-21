"""Pure abstractions for the agent and tool framework.

``agent.py`` defines ``AgentSpec`` and ``tool.py`` defines the tool contracts.
``agent_protocol.py`` owns the semantic boundary shared by the conversational
and durable runners. These modules depend only on the standard library, so
importing them does not pull runtime wiring or create repository cycles.
"""

from .agent_protocol import (
    AgentRunResult,
    LoopStepOutcome,
    ModelOutcome,
    ModelOutcomeKind,
    ToolRequest,
)

__all__ = [
    "AgentRunResult",
    "LoopStepOutcome",
    "ModelOutcome",
    "ModelOutcomeKind",
    "ToolRequest",
]
