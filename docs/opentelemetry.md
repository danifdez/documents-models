# Models OpenTelemetry

The Python dependency versions are pinned in `../requirements-telemetry.txt`.
The exporter is initialized independently inside each spawned handler process;
it is never inherited through a fork or used to determine result delivery.

Assignments restore validated W3C context for the handler and its actual inference
calls. The context is restored on both return and exception. The worker emits no
separate execution-lifecycle span: Backend's committed execution events are the
source for starts, state changes, attempts and terminal outcomes. This avoids a
second state machine or duplicate worker/action tracking.
