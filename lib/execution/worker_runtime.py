import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event
from typing import Callable

from lib.execution.assignment_runner import run_assignment
from lib.execution.handler_process_pool import HandlerProcessPool
from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolRejectionError,
    ProtocolTransportError,
    WorkerAuthenticationError,
)
from lib.execution.result_outbox import ResultOutbox

logger = logging.getLogger(__name__)

AssignmentRunner = Callable[
    [ExecutionProtocolClient, ResultOutbox, dict, Event], None
]

CONTROL_POLL_INTERVAL_SECONDS = 1.0
TERMINAL_ATTEMPT_REJECTIONS = {
    "attempt_not_current",
    "attempt_not_found",
    "lease_expired",
    "step_deadline_reached",
}


@dataclass
class ActiveAssignment:
    future: Future[None]
    cancellation: Event
    next_lease_renewal: float
    next_control_poll: float
    lease_expires_at: float


class WorkerRuntime:
    def __init__(
        self,
        client: ExecutionProtocolClient,
        outbox: ResultOutbox,
        maximum_concurrency: int,
        lease_duration_ms: int,
        runner: AssignmentRunner | None = None,
        handler_pool: HandlerProcessPool | None = None,
        control_poll_interval_seconds: float = CONTROL_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.client = client
        self.outbox = outbox
        self.maximum_concurrency = maximum_concurrency
        self.lease_duration_ms = lease_duration_ms
        self.lease_renewal_seconds = lease_duration_ms / 3000
        self.control_poll_interval_seconds = control_poll_interval_seconds
        self.handler_pool = None
        if runner is None:
            self.handler_pool = handler_pool or HandlerProcessPool(
                maximum_concurrency
            )

            def preemptible_runner(
                client: ExecutionProtocolClient,
                outbox: ResultOutbox,
                assignment: dict,
                cancellation: Event,
            ) -> None:
                run_assignment(
                    client,
                    outbox,
                    assignment,
                    cancellation,
                    handler_executor=self.handler_pool.execute,
                )

            self.runner = preemptible_runner
        else:
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
        submitted_at = time.monotonic()
        self.active[attempt_id] = ActiveAssignment(
            future=future,
            cancellation=cancellation,
            next_lease_renewal=submitted_at + self.lease_renewal_seconds,
            next_control_poll=(
                submitted_at + self.control_poll_interval_seconds
            ),
            lease_expires_at=(
                submitted_at + self.lease_duration_ms / 1000
            ),
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
            if current >= active.lease_expires_at:
                if not active.cancellation.is_set():
                    logger.warning(
                        "Preempting attempt %s after its local lease window",
                        attempt_id,
                    )
                active.cancellation.set()
                continue
            if current >= active.next_control_poll:
                if self._refresh_control(attempt_id, active, current):
                    continue
            if current < active.next_lease_renewal:
                continue
            try:
                control = self.client.renew_lease(
                    attempt_id, self.lease_duration_ms
                )
            except WorkerAuthenticationError:
                raise
            except ProtocolRejectionError as error:
                if _is_terminal_attempt_rejection(error):
                    logger.warning(
                        "Preempting attempt %s after lease rejection %s",
                        attempt_id,
                        error.error_code or error.status_code,
                    )
                    active.cancellation.set()
                    continue
                active.next_lease_renewal = current + 1
                logger.warning(
                    "Lease renewal rejected for attempt %s: %s",
                    attempt_id,
                    error,
                )
                continue
            except ProtocolTransportError as error:
                active.next_lease_renewal = current + 1
                logger.warning(
                    "Lease renewal failed for attempt %s: %s",
                    attempt_id,
                    error,
                )
                continue
            if control.get("cancelled") is True:
                active.cancellation.set()
                continue
            remaining_ms = control.get("leaseRemainingMs")
            if (
                isinstance(remaining_ms, (int, float))
                and not isinstance(remaining_ms, bool)
                and remaining_ms > 0
            ):
                active.lease_expires_at = current + remaining_ms / 1000
            else:
                active.lease_expires_at = (
                    current + self.lease_duration_ms / 1000
                )
            active.next_lease_renewal = (
                current + self.lease_renewal_seconds
            )

    def seconds_until_maintenance(self, now: float | None = None) -> float:
        if not self.active:
            return float("inf")
        current = time.monotonic() if now is None else now
        next_maintenance = min(
            min(
                active.next_lease_renewal
                for active in self.active.values()
            ),
            min(
                active.next_control_poll for active in self.active.values()
            ),
            min(active.lease_expires_at for active in self.active.values()),
        )
        return max(
            0,
            next_maintenance - current,
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
        if self.handler_pool is not None:
            self.handler_pool.close()

    def _refresh_control(
        self,
        attempt_id: str,
        active: ActiveAssignment,
        current: float,
    ) -> bool:
        try:
            control = self.client.read_control(attempt_id)
        except WorkerAuthenticationError:
            raise
        except ProtocolRejectionError as error:
            if _is_terminal_attempt_rejection(error):
                logger.warning(
                    "Preempting attempt %s after control rejection %s",
                    attempt_id,
                    error.error_code or error.status_code,
                )
                active.cancellation.set()
                return True
            logger.warning(
                "Control read rejected for attempt %s: %s",
                attempt_id,
                error,
            )
        except ProtocolTransportError as error:
            logger.warning(
                "Control read failed for attempt %s: %s",
                attempt_id,
                error,
            )
        else:
            if control.get("cancelled") is True:
                active.cancellation.set()
                return True
        finally:
            active.next_control_poll = (
                current + self.control_poll_interval_seconds
            )
        return False


def _is_terminal_attempt_rejection(error: ProtocolRejectionError) -> bool:
    return (
        error.status_code in {404, 409}
        or error.error_code in TERMINAL_ATTEMPT_REJECTIONS
    )
