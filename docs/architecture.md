# Models architecture

Models is a worker process that executes self-contained steps granted by the
Documents Backend. Backend is the sole authority for execution state, step
selection, retries, dependencies, leases, cancellation, and finalization.

## Runtime flow

1. `executions.py` registers a stable worker identity through Backend.
2. The worker heartbeats its effective capabilities and hardware metadata.
3. It claims a compatible ready step over HTTP.
4. Backend creates the `StepAttempt` and returns a fenced assignment.
5. Models starts the attempt, renews its lease, checks cancellation, and
   downloads only the artifacts referenced by that assignment.
6. `lib/execution/step_executor.py` loads the task handler and executes it.
7. The result is stored in a local outbox and retried until Backend returns a
   terminal ACK (`received`, `duplicate`, `stale_attempt`,
   `result_conflict`, or `rejected`).

Models never claims or updates execution rows in PostgreSQL. It does not create
child executions. Durable fan-out and successor steps are materialized by the
Backend coordinator.

## Main modules

```text
executions.py                         worker loop and durable result outbox
worker/identity.py                    stable local worker identity
lib/execution/protocol_client.py      authenticated Backend protocol
lib/execution/step_executor.py        assignment-to-handler adapter
utils/task_dispatch.py                conventional task module loading
common/execution_registry.py          @execution_handler registry
tasks/<task>/<task>.py                self-contained task handlers
config/tasks.json                     task configuration
database/                             task-domain data access only
```

## Task dispatch

A handler registers one task type:

```python
from common.execution_registry import execution_handler


@execution_handler("detect-language")
def detect_language(payload):
    return {"results": []}
```

`utils/task_dispatch.py` resolves conventional module paths and invokes the
handler. A handler receives explicit JSON payload data and downloaded artifacts
under `_input_artifacts`, keyed by assignment role. It returns a dictionary
that becomes the value of the `StepResult` output.

## Data access boundary

Task implementations may read or write their own domain stores when the step
contract authorizes that effect, such as pgvector data or the entity graph.
They must not write `executions`, `execution_steps`, attempts, receipts, or
events. Those tables are owned exclusively by Backend.

## Failure and recovery

- A result is persisted locally before delivery and removed only after ACK.
- A lost response is safe: an identical retry returns `duplicate`.
- An expired or superseded lease fences late results as `stale_attempt`.
- A worker credential can be rotated; a rejected credential triggers
  re-enrollment.
- A task is advertised only after it can run through this protocol without
  execution-table SQL access.
