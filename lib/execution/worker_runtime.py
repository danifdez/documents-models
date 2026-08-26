import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event
from typing import Callable

from lib.execution.assignment_runner import run_assignment
from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
    WorkerAuthenticationError,
)
from lib.execution.result_outbox import ResultOutbox

logger = logging.getLogger(__name__)

AssignmentRunner = Callable[
    [ExecutionProtocolClient, ResultOutbox, dict, Event], None
]


@dataclass
class ActiveAssignment:
    future: Future[None]
    cancellation: Event
    next_lease_renewal: float


class WorkerRuntime:
    def __init__(
        self,
        client: ExecutionProtocolClient,
        outbox: ResultOutbox,
        maximum_concurrency: int,
        lease_duration_ms: int,
        runner: AssignmentRunner = run_assignment,
    ) -> None:
        self.client = client
        self.outbox = outbox
        self.maximum_concurrency = maximum_concurrency
        self.lease_duration_ms = lease_duration_ms
        self.lease_renewal_seconds = lease_duration_ms / 3000
        self.runner = runner
        self.executor = ThreadPoolExecutor(
            max_workers=maximum_concurrency,
            thread_name_prefix="models-step",
        )
        self.active: dict[str, ActiveAssignment] = {}

    @property
    def available_slots(self) -> int:
        return self.maximum_concurrency - len(self.active)

    @property
    def active_attempt_ids(self) -> set[str]:
        return set(self.active)

    def submit(self, assignment: dict) -> None:
        if self.available_slots <= 0:
            raise RuntimeError("Worker has no available assignment slot")
        attempt_id = assignment["attemptId"]
        if attempt_id in self.active:
            raise RuntimeError("Attempt is already active")

        self.client.start(attempt_id)
        cancellation = Event()
        future = self.executor.submit(
            self.runner,
            self.client,
            self.outbox,
            assignment,
            cancellation,
        )
        self.active[attempt_id] = ActiveAssignment(
            future=future,
            cancellation=cancellation,
            next_lease_renewal=time.monotonic()
            + self.lease_renewal_seconds,
        )

    def collect_completed(self) -> None:
        for attempt_id, active in list(self.active.items()):
            if not active.future.done():
                continue
            del self.active[attempt_id]
            active.future.result()

    def maintain_leases(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        for attempt_id, active in self.active.items():
            if current < active.next_lease_renewal:
                continue
            try:
                control = self.client.renew_lease(
                    attempt_id, self.lease_duration_ms
                )
            except WorkerAuthenticationError:
                raise
            except ProtocolTransportError as error:
                active.next_lease_renewal = current + 1
                logger.warning(
                    "Lease renewal failed for attempt %s: %s",
                    attempt_id,
                    error,
                )
                continue
            if control.get("cancelled"):
                active.cancellation.set()
            active.next_lease_renewal = (
                current + self.lease_renewal_seconds
            )

    def seconds_until_maintenance(self, now: float | None = None) -> float:
        if not self.active:
            return float("inf")
        current = time.monotonic() if now is None else now
        return max(
            0,
            min(
                active.next_lease_renewal for active in self.active.values()
            )
            - current,
        )

    def wait_for_completion(self, timeout: float) -> None:
        if not self.active:
            if timeout > 0:
                time.sleep(timeout)
            return
        wait(
            [active.future for active in self.active.values()],
            timeout=max(0, timeout),
            return_when=FIRST_COMPLETED,
        )

    def close(self) -> None:
        for active in self.active.values():
            active.cancellation.set()
        self.executor.shutdown(wait=True, cancel_futures=False)
