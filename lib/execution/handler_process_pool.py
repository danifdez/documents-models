import multiprocessing
import queue
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Callable
from uuid import uuid4

from lib.execution.step_executor import execute_assignment

PROCESS_POLL_SECONDS = 0.05
PROCESS_STOP_GRACE_SECONDS = 1.0

HandlerExecutor = Callable[..., dict]


class HandlerPreempted(RuntimeError):
    pass


class HandlerProcessFailed(RuntimeError):
    pass


@dataclass
class _HandlerSlot:
    process: BaseProcess
    connection: Connection


class HandlerProcessPool:
    def __init__(
        self,
        size: int,
        executor: HandlerExecutor = execute_assignment,
    ) -> None:
        if size < 1:
            raise ValueError("Handler process pool size must be positive")
        self._context = multiprocessing.get_context("spawn")
        self._executor = executor
        self._available: queue.Queue[_HandlerSlot] = queue.Queue(size)
        self._slots: list[_HandlerSlot] = []
        self._lock = threading.Lock()
        self._closed = False
        try:
            for _ in range(size):
                slot = self._spawn_slot()
                self._slots.append(slot)
                self._available.put(slot)
        except BaseException:
            self.close()
            raise

    def execute(
        self,
        assignment: dict,
        artifacts: dict[str, bytes],
        cancellation: threading.Event,
    ) -> tuple[dict, list[dict]]:
        slot = self._acquire(cancellation)
        command_id = str(uuid4())
        released = False
        try:
            slot.connection.send(
                {
                    "kind": "execute",
                    "commandId": command_id,
                    "assignment": assignment,
                    "artifacts": artifacts,
                }
            )
            while True:
                if cancellation.is_set():
                    self._replace(slot)
                    released = True
                    raise HandlerPreempted(
                        f"Handler for attempt {assignment.get('attemptId')} "
                        "was preempted"
                    )
                if slot.connection.poll(PROCESS_POLL_SECONDS):
                    try:
                        response = slot.connection.recv()
                    except EOFError as error:
                        self._replace(slot)
                        released = True
                        raise HandlerProcessFailed(
                            "Handler process closed its result channel"
                        ) from error
                    if not isinstance(response, dict):
                        self._replace(slot)
                        released = True
                        raise HandlerProcessFailed(
                            "Handler process returned an invalid envelope"
                        )
                    if response.get("commandId") != command_id:
                        self._replace(slot)
                        released = True
                        raise HandlerProcessFailed(
                            "Handler process returned a mismatched command"
                        )
                    if response.get("kind") == "failed":
                        self._replace(slot)
                        released = True
                        raise HandlerProcessFailed(
                            f"{response.get('errorType')}: "
                            f"{response.get('message')}"
                        )
                    result = response.get("result")
                    output_artifacts = response.get("outputArtifacts")
                    if not isinstance(result, dict) or not isinstance(
                        output_artifacts, list
                    ):
                        self._replace(slot)
                        released = True
                        raise HandlerProcessFailed(
                            "Handler process returned an invalid result"
                        )
                    self._available.put(slot)
                    released = True
                    return result, output_artifacts
                if not slot.process.is_alive():
                    self._replace(slot)
                    released = True
                    raise HandlerProcessFailed(
                        "Handler process exited without a result"
                    )
        except (BrokenPipeError, EOFError, OSError) as error:
            if not released:
                self._replace(slot)
                released = True
            raise HandlerProcessFailed(
                "Handler process communication failed"
            ) from error
        finally:
            if not released:
                self._available.put(slot)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots)
            self._slots.clear()
        for slot in slots:
            try:
                if slot.process.is_alive():
                    slot.connection.send({"kind": "shutdown"})
                    slot.process.join(PROCESS_STOP_GRACE_SECONDS)
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                _stop_process(slot.process)
                slot.connection.close()

    def _acquire(self, cancellation: threading.Event) -> _HandlerSlot:
        while True:
            if cancellation.is_set():
                raise HandlerPreempted("Handler was cancelled before dispatch")
            with self._lock:
                if self._closed:
                    raise RuntimeError("Handler process pool is closed")
            try:
                return self._available.get(timeout=PROCESS_POLL_SECONDS)
            except queue.Empty:
                continue

    def _replace(self, slot: _HandlerSlot) -> None:
        _stop_process(slot.process)
        slot.connection.close()
        with self._lock:
            if slot in self._slots:
                self._slots.remove(slot)
            if self._closed:
                return
            replacement = self._spawn_slot()
            self._slots.append(replacement)
        self._available.put(replacement)

    def _spawn_slot(self) -> _HandlerSlot:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_handler_slot_main,
            args=(child, self._executor),
            name="models-handler",
        )
        process.start()
        child.close()
        return _HandlerSlot(process=process, connection=parent)


def _handler_slot_main(
    connection: Connection,
    executor: HandlerExecutor,
) -> None:
    try:
        while True:
            command = connection.recv()
            if command.get("kind") == "shutdown":
                return
            if command.get("kind") != "execute":
                raise RuntimeError("Unknown handler process command")
            command_id = command.get("commandId")
            output_artifacts: list[dict] = []
            try:
                result = executor(
                    command["assignment"],
                    command.get("artifacts") or {},
                    output_artifacts=output_artifacts,
                )
                response = {
                    "kind": "result",
                    "commandId": command_id,
                    "result": result,
                    "outputArtifacts": output_artifacts,
                }
            except BaseException as error:
                response = {
                    "kind": "failed",
                    "commandId": command_id,
                    "errorType": type(error).__name__,
                    "message": str(error),
                }
            connection.send(response)
    except (BrokenPipeError, EOFError, OSError):
        return
    finally:
        connection.close()


def _stop_process(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
        process.join(PROCESS_STOP_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(PROCESS_STOP_GRACE_SECONDS)
    else:
        process.join(timeout=0)
