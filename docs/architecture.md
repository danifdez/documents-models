# Models architecture

Models is a worker process that executes self-contained steps granted by the
Documents Backend. Backend is the sole authority for execution state, step
selection, retries, dependencies, leases, cancellation, and finalization.

## Runtime flow

1. `executions.py` registers a stable worker identity, protocol, supported step
   kinds and concurrency through Backend.
2. The worker heartbeats its effective capabilities and hardware metadata.
3. It long-polls Backend while its local pool has an available slot, bounding
   each wait so heartbeats and lease renewals are still sent on schedule.
4. Backend creates the `StepAttempt` and returns a fenced assignment.
5. Models starts the attempt in an independent pool slot; the control thread
   renews its lease and records cancellation while the handler runs.
6. The slot downloads only the referenced artifacts and
   `lib/execution/step_executor.py` executes the task handler.
7. Large handler outputs are encoded as immutable attempt-scoped artifacts.
8. Each result and its artifacts are stored atomically in a per-attempt local
   outbox entry;
   artifacts are uploaded first and the result is retried until Backend returns
   a terminal ACK (`received`, `duplicate`, `stale_attempt`,
   `result_conflict`, or `rejected`).

Models never claims or updates execution rows in PostgreSQL. It does not create
child executions. Durable fan-out and successor steps are materialized by the
Backend coordinator.

The worker instantiates exactly `worker.maximum_concurrency` execution slots
and advertises that same value. The default is two, matching the shared
`llama-server` default slot count. Backend derives active assignments from live
leases and refuses another claim when all declared slots are occupied. An empty
claim waits in Backend for at most ten seconds when the pool is idle and one
second while other slots are active; transport failures use a separate
one-second retry backoff. Embedded singleton runtimes for embeddings, Whisper
and translation serialize their own initialization and inference calls; this
prevents unsafe shared-model access without serializing unrelated assignments.

## Main modules

```text
executions.py                         worker control loop and claim scheduling
worker/identity.py                    stable local worker identity
lib/execution/protocol_client.py      authenticated Backend protocol
lib/execution/worker_runtime.py       concurrent slots and lease maintenance
lib/execution/assignment_runner.py    isolated assignment lifecycle
lib/execution/result_outbox.py        per-attempt durable result delivery
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
- Cancellation is checked before artifact loading and after handler execution;
  a running Python handler is not preempted, but its output is discarded when
  cancellation was observed.
- A worker credential can be rotated; a rejected credential triggers
  re-enrollment.
- A task is advertised only after it can run through this protocol without
  execution-table SQL access.
