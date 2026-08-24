# Creating a Models task

A Models task is a self-contained executor for one Backend step. It must not
claim work, update execution state, or create child executions.

## 1. Add the handler

Use the conventional path `tasks/<module>/<module>.py`, where hyphens and dots
in the task type become underscores in the module name.

```python
from common.execution_registry import execution_handler


@execution_handler("example-task")
def example_task(payload):
    text = payload.get("text", "")
    return {"result": text.upper()}
```

The handler must return a dictionary. Keep its input and output shapes stable;
Backend stores the output in the step result and applies domain finalization.

## 2. Declare task configuration

Add an exact task entry to `config/tasks.json`. Pin model and runtime choices;
do not use floating dependency versions.

```json
{
  "example-task": {
    "enabled": true,
    "type": "utility",
    "capabilities": []
  }
}
```

## 3. Advertise the effective capability

Add the task type to the capabilities built by `executions.py` only when the
handler, its dependencies, its artifact roles, and its tests are ready. An
unadvertised task cannot be claimed.

## 4. Consume artifacts by role

Backend may attach attempt-scoped input artifacts:

```python
@execution_handler("document-extraction")
def extract(payload):
    document = (payload.get("_input_artifacts") or {}).get("document")
    if document is None:
        return {"error": "document artifact is required"}
    return {"content": document.decode("utf-8", errors="replace")}
```

Do not accept filesystem paths or PostgreSQL blob columns as an alternative
transport.

## 5. Keep coordination in Backend

If work needs fan-out, dependencies, retries, confirmation, or a child
execution with its own lifecycle, define successor steps in the Backend
coordinator. A Models handler may perform bounded in-process computation, but
it cannot enqueue or resume executions.

## 6. Test the observable contract

Add standard-library tests under `tests/` for:

- valid and invalid payloads;
- required artifact roles;
- deterministic output shape;
- failure conversion to a failed `StepResult`;
- result outbox retry and duplicate ACK behavior when protocol code changes.

Run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

For protocol changes, also run the Backend PostgreSQL E2E suite.
