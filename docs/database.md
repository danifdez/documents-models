# Database access

PostgreSQL is available to task implementations for application-domain data.
It is not the worker queue.

## Ownership

Backend exclusively owns these execution control-plane records:

- `executions`
- `execution_steps`
- `execution_step_dependencies`
- `execution_step_attempts`
- `execution_result_receipts`
- `execution_events`
- `execution_artifacts`
- worker credentials and registration state

Models accesses them only through the authenticated HTTP protocol. There is no
execution database adapter or SQL claim path in this repository.

## Task-domain modules

The `database/` package contains narrowly scoped data access used by task
logic:

- `connection.py` opens a short-lived PostgreSQL connection for relational
  reads used by dataset tasks.
- `dataset.py` reads dataset schemas and records.
- `memory.py` reads assistant memory rows.
- `rag.py` manages pgvector-backed task data.
- `graph_db.py` manages the Apache AGE entity graph.

Connections use the configured `POSTGRES_*` values and should be scoped with a
context manager or closed explicitly. Adding a domain query does not authorize
changing execution lifecycle state.

## Artifacts

Input file bodies are not read from an execution blob column. Backend returns
artifact references in the assignment, and Models downloads each body through
the attempt-scoped artifact endpoint. The executor exposes them to handlers as:

```python
payload["_input_artifacts"]["document"]
```

The role is part of the step contract. A handler must fail clearly when a
required role is absent.
