# Models architecture

Models is a worker process that executes self-contained steps granted by the
Documents Backend. Backend is the sole authority for execution state, step
selection, retries, dependencies, leases, cancellation, and finalization.

## Runtime flow

1. `executions.py` registers a stable worker identity, protocol, supported step
   kinds and concurrency through Backend.
2. The worker heartbeats its effective capabilities and hardware metadata.
3. It claims a compatible ready step over HTTP.
4. Backend creates the `StepAttempt` and returns a fenced assignment.
5. Models starts the attempt, renews its lease, checks cancellation, and
   downloads only the artifacts referenced by that assignment.
6. `lib/execution/step_executor.py` loads the task handler and executes it.
7. Large handler outputs are encoded as immutable attempt-scoped artifacts.
8. Artifacts and the lightweight result are stored together in a local outbox;
   artifacts are uploaded first and the result is retried until Backend returns
   a terminal ACK (`received`, `duplicate`, `stale_attempt`,
   `result_conflict`, or `rejected`).

Models never claims or updates execution rows in PostgreSQL. It does not create
child executions. Durable fan-out and successor steps are materialized by the
Backend coordinator.

The current loop executes one assignment at a time and therefore declares
`maximumConcurrency: 1`. Backend derives active assignments from live leases
and refuses another claim until that slot is available.

## Main modules

```text
executions.py                         worker loop and durable result outbox
worker/identity.py                    stable local worker identity
lib/execution/protocol_client.py      authenticated Backend protocol
lib/execution/step_executor.py        assignment-to-handler adapter
lib/execution/output_artifact.py      deterministic output artifact encoding
utils/task_dispatch.py                conventional task module loading
common/execution_registry.py          @execution_handler registry
tasks/<task>/<task>.py                self-contained task handlers
config/tasks.json                     task configuration
common/vector_contract.py             frozen vector candidate validation/ranking
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
that becomes the value of the `StepResult` output, or `HandlerOutput` when a
large body must travel as one or more output artifacts.

## Data access boundary

Task implementations do not open application datastores. Backend supplies
scope-checked payloads and artifacts; Models returns calculated data and
Backend applies every relational, vector or graph effect during finalization.

## Failure and recovery

- Artifacts and their result are persisted locally before delivery and removed
  only after the result ACK.
- Artifact retries are idempotent and fenced to the attempt that produced them.
- A lost response is safe: an identical retry returns `duplicate`.
- An expired or superseded lease fences late results as `stale_attempt`.
- A worker credential can be rotated; a rejected credential triggers
  re-enrollment.
- A task is advertised only after it can run through this protocol without
  execution-table SQL access.
