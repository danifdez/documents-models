"""Abstracciones puras del framework de agentes y tools.

`agent.py` define `AgentSpec` (qué ES un agente) y `tool.py` define
`Tool`/`ToolContext`/`register`/`REGISTRY` (qué ES una tool). Ambos son módulos
puros (solo stdlib): ni `core/tools` ni `core/agents` arrastran wiring al
importarlos, y no hay ciclo entre ellos.
"""
