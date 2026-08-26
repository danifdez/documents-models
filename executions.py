import json
import logging
import signal
import sys
import time
from pathlib import Path

from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
    WorkerAuthenticationError,
)
from lib.execution.step_executor import execute_assignment
from lib.execution.code_identity import code_fingerprint
from lib.execution.runtime_identity import runtime_fingerprint
from utils.device import (
    CPU_COUNT,
    GPU_NAME,
    HAS_CUDA,
    RAM_GB,
    VRAM_GB,
    log_hardware_summary,
)
from worker.identity import (
    HEARTBEAT_INTERVAL,
    WORKER_ID,
    WORKER_NAME,
    worker_data_dir,
)
from worker.capabilities import (
    detect_worker_capabilities,
    get_supported_task_types,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1
LEASE_DURATION_MS = 30_000
SUPPORTED_TASK_TYPES = (
    "assistant-chat",
    "agent-chat",
    "ask",
    "detect-language",
    "document-extraction",
    "embedding",
    "dataset.extract-row",
    "dataset.propose-columns",
    "entity-extraction-map",
    "entity-extraction-reduce",
    "date-extraction-map",
    "date-extraction-reduce",
    "keywords-map",
    "keywords-reduce",
    "key-point-map",
    "key-point-reduce",
    "memory-ingest",
    "memory-search",
    "distribution",
    "correlation",
    "correlation-matrix",
    "group-by",
    "time-series",
    "outliers",
    "pivot-table",
    "summary",
    "query",
    "chart",
    "transcribe",
    "translate",
    "summarize-map",
    "summarize-reduce",
    "search",
    "ingest-content",
    "indexed-file-extraction",
    "indexed-file-ingest",
    "indexed-file-search",
    "relationship-extraction-map",
    "relationship-extraction-reduce",
)
STEP_KINDS = ["service", "code", "inference"]
ACK_CODES = {
    "received",
    "duplicate",
    "stale_attempt",
    "result_conflict",
    "rejected",
}


def effective_task_capabilities() -> list[str]:
    supported = set(
        get_supported_task_types(detect_worker_capabilities())
    )
    return [
        task_type
        for task_type in SUPPORTED_TASK_TYPES
        if task_type in supported
    ]


CAPABILITIES = effective_task_capabilities()


def _metadata() -> dict:
    return {
        "cpuCount": CPU_COUNT,
        "ramGb": RAM_GB,
        "hasCuda": HAS_CUDA,
        "gpuName": GPU_NAME,
        "vramGb": VRAM_GB,
        "codeFingerprint": code_fingerprint(),
        "runtimeFingerprint": runtime_fingerprint(),
    }


def _pending_path() -> Path:
    return Path(worker_data_dir()) / ".pending_step_result.json"


def _store_pending(result: dict, artifacts: list[dict] | None = None) -> None:
    path = _pending_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"result": result, "artifacts": artifacts or []},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_pending() -> dict | None:
    try:
        value = json.loads(_pending_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("result"), dict)
        or not isinstance(value.get("artifacts"), list)
    ):
        raise RuntimeError("Pending step result is invalid")
    return value


def _deliver_pending(client: ExecutionProtocolClient) -> None:
    pending = _load_pending()
    if pending is None:
        return
    result = pending["result"]
    for artifact in pending["artifacts"]:
        ack = client.upload_artifact(result["attemptId"], artifact)
        if ack.get("code") not in {"received", "duplicate"}:
            if ack.get("code") == "stale_attempt":
                break
            raise ProtocolTransportError(
                f"Output artifact rejected: {ack.get('code')}"
            )
    ack = client.submit_result(result)
    code = ack.get("code")
    if code not in ACK_CODES:
        raise ProtocolTransportError(f"Unknown result ACK: {code}")
    _pending_path().unlink(missing_ok=True)
    logger.info("Result %s acknowledged as %s", result.get("attemptId"), code)


def main() -> None:
    log_hardware_summary()
    client = ExecutionProtocolClient()
    client.ensure_registered(CAPABILITIES, _metadata())
    logger.info(
        "Worker registered through Backend: %s (%s)", WORKER_NAME, WORKER_ID
    )

    stopping = False

    def shutdown(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    next_heartbeat = 0.0

    while not stopping:
        try:
            client.ensure_registered(CAPABILITIES, _metadata())
            _deliver_pending(client)
            now = time.monotonic()
            if now >= next_heartbeat:
                client.heartbeat(CAPABILITIES, _metadata())
                next_heartbeat = now + HEARTBEAT_INTERVAL
            assignment = client.claim(
                CAPABILITIES, STEP_KINDS, LEASE_DURATION_MS
            )
            if assignment:
                client.start(assignment["attemptId"])
                control = client.read_control(assignment["attemptId"])
                if control.get("cancelled"):
                    result = {
                        "schemaVersion": "step-result/1",
                        "executionId": assignment["executionId"],
                        "stepId": assignment["stepId"],
                        "operationId": assignment["operationId"],
                        "attemptId": assignment["attemptId"],
                        "stepKind": assignment["stepKind"],
                        "status": "cancelled",
                        "codeFingerprint": code_fingerprint(),
                        "runtimeFingerprint": runtime_fingerprint(),
                        "artifactRefs": [],
                        "error": None,
                    }
                    _store_pending(result)
                    _deliver_pending(client)
                    continue
                client.renew_lease(
                    assignment["attemptId"], LEASE_DURATION_MS
                )
                artifacts = {
                    ref["role"]: client.download_artifact(
                        assignment["attemptId"], ref["artifactId"]
                    )
                    for ref in assignment.get("inputArtifactRefs", [])
                }
                output_artifacts = []
                result = execute_assignment(
                    assignment,
                    artifacts,
                    output_artifacts=output_artifacts,
                )
                if client.read_control(assignment["attemptId"]).get(
                    "cancelled"
                ):
                    result["status"] = "cancelled"
                    result.pop("output", None)
                    result["artifactRefs"] = []
                    result["error"] = None
                    output_artifacts = []
                _store_pending(result, output_artifacts)
                _deliver_pending(client)
                continue
        except WorkerAuthenticationError as error:
            client.reset_credential()
            next_heartbeat = 0.0
            logger.warning("Worker credential rejected; re-enrolling: %s", error)
        except ProtocolTransportError as error:
            logger.warning("Protocol unavailable: %s", error)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    if "--setup" in sys.argv:
        from setup_models import setup

        setup()
    else:
        main()
