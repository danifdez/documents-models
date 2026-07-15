"""Librerías y abstracciones del subsistema LLM/agentes.

Todo lo que aquí vive es consumido exclusivamente por la parte de la aplicación
que hace inferencia y orquesta agentes. Está organizado jerárquicamente:

- `lib.llm`       — plumbing de inferencia: carga de config y prompts, gramáticas
                    GBNF, parseo de JSON del modelo y limpieza de <think>.
- `lib.framework` — las abstracciones puras del framework: `AgentSpec` (qué es un
                    agente) y `Tool`/`ToolContext`/`register`/`REGISTRY` (qué es
                    una tool). Sin imports fuera de la stdlib.
- `lib.backend`   — transporte HTTP al backend NestJS y resolvers de dominio
                    (calendario, tareas, carpeta) que las tools comparten.
"""
