import logging
import multiprocessing
import json
import os
import signal
import subprocess
import sys
import time

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--self-check" in sys.argv:
        os.environ["MODELS_SELF_CHECK"] = "1"

from lib.execution.result_outbox import ResultOutbox
from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
    WorkerAuthenticationError,
)
from lib.execution.code_identity import code_fingerprint
from lib.execution.runtime_identity import runtime_fingerprint
from lib.execution.worker_runtime import WorkerRuntime
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
    MAXIMUM_CONCURRENCY,
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

PROTOCOL_RETRY_BACKOFF_SECONDS = 1
LEASE_DURATION_MS = 30_000
CLAIM_WAIT_TIMEOUT_MS = 10_000
ACTIVE_CLAIM_WAIT_TIMEOUT_MS = 1_000
SUPPORTED_TASK_TYPES = (
    "assistant-chat",
    "agent-chat",
    "context-input-map",
    "context-input-reduce",
    "ask",
    "browser-inference",
    "detect-language",
    "document-extraction",
    "data-source-sync",
    "embedding",
    "dataset.extract-row",
    "dataset.extract-row-map",
    "dataset.extract-row-reduce",
    "dataset.propose-columns",
    "entity-extraction-map",
    "entity-extraction-reduce",
    "date-extraction-map",
    "date-extraction-reduce",
    "keywords-map",
    "keywords-reduce",
    "key-point-map",
    "key-point-reduce",
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
    "translate-map",
    "translate-reduce",
    "summarize-map",
    "summarize-reduce",
    "summarize-compose",
    "summarize-finalize",
    "search",
    "ingest-content",
    "indexed-file-extraction",
    "indexed-file-ingest",
    "indexed-file-search",
    "relationship-extraction-map",
    "relationship-extraction-reduce",
)
STEP_KINDS = ["service", "code", "inference"]


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


def self_check() -> int:
    from common.execution_registry import TASK_HANDLERS
    from lib.llm.config import get_tasks
    from services.llama_server import server_binary
    from utils.runtime_variant import validate_llama_variant
    from utils.task_dispatch import ensure_task_handler

    failed_handlers = [
        task_type
        for task_type in CAPABILITIES
        if not ensure_task_handler(task_type)
    ]
    binary = server_binary()
    variant = os.environ.get("DOCUMENTS_MODELS_VARIANT", "unknown")
    llama_output = ""
    llama_error = ""
    if binary:
        try:
            completed = subprocess.run(
                [binary, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            llama_output = f"{completed.stdout}\n{completed.stderr}".strip()
            llama_compatible, llama_error = validate_llama_variant(
                variant,
                llama_output,
            )
        except (OSError, subprocess.SubprocessError) as error:
            llama_compatible = False
            llama_error = str(error)
    else:
        llama_compatible = False
        llama_error = "llama-server was not found"
    defaults_available = bool(get_tasks())
    prompts_available = os.path.isdir(
        os.path.join(
            getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
            "tasks",
        )
    )
    if prompts_available:
        prompts_root = os.path.join(
            getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
            "tasks",
        )
        prompts_available = any(
            name.endswith(".md")
            for _, _, names in os.walk(prompts_root)
            for name in names
        )
    result = {
        "ok": (
            not failed_handlers
            and defaults_available
            and prompts_available
            and llama_compatible
        ),
        "component": "models",
        "variant": variant,
        "handlers": len(TASK_HANDLERS),
        "failedHandlers": failed_handlers,
        "llamaServer": {
            "available": bool(binary),
            "path": binary,
            "compatible": llama_compatible,
            "build": llama_output,
            "error": llama_error,
        },
        "resources": {
            "defaults": defaults_available,
            "prompts": prompts_available,
        },
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


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


def main() -> None:
    log_hardware_summary()
    client = ExecutionProtocolClient()
    client.ensure_registered(
        CAPABILITIES, STEP_KINDS, MAXIMUM_CONCURRENCY, _metadata()
    )
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
    outbox = ResultOutbox(worker_data_dir())
    runtime = WorkerRuntime(
        client,
        outbox,
        MAXIMUM_CONCURRENCY,
        LEASE_DURATION_MS,
    )

    try:
        while not stopping:
            try:
                runtime.collect_completed()
                client.ensure_registered(
                    CAPABILITIES,
                    STEP_KINDS,
                    MAXIMUM_CONCURRENCY,
                    _metadata(),
                )
                outbox.deliver_all(client, runtime.active_attempt_ids)
                now = time.monotonic()
                if now >= next_heartbeat:
                    client.heartbeat(
                        CAPABILITIES,
                        STEP_KINDS,
                        MAXIMUM_CONCURRENCY,
                        _metadata(),
                    )
                    next_heartbeat = now + HEARTBEAT_INTERVAL
                runtime.maintain_leases(now)

                if runtime.available_slots > 0:
                    wait_seconds = min(
                        CLAIM_WAIT_TIMEOUT_MS / 1000,
                        max(0, next_heartbeat - time.monotonic()),
                        runtime.seconds_until_maintenance(),
                    )
                    if runtime.active_attempt_ids:
                        wait_seconds = min(
                            wait_seconds,
                            ACTIVE_CLAIM_WAIT_TIMEOUT_MS / 1000,
                        )
                    assignment = client.claim(
                        CAPABILITIES,
                        STEP_KINDS,
                        LEASE_DURATION_MS,
                        max(0, int(wait_seconds * 1000)),
                    )
                    if assignment:
                        runtime.submit(assignment)
                        continue
                else:
                    runtime.wait_for_completion(
                        min(
                            max(0, next_heartbeat - time.monotonic()),
                            runtime.seconds_until_maintenance(),
                        )
                    )
                    continue
            except WorkerAuthenticationError as error:
                client.reset_credential()
                next_heartbeat = 0.0
                logger.warning(
                    "Worker credential rejected; re-enrolling: %s", error
                )
            except ProtocolTransportError as error:
                logger.warning("Protocol unavailable: %s", error)
            time.sleep(PROTOCOL_RETRY_BACKOFF_SECONDS)
    finally:
        runtime.close()


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        raise SystemExit(self_check())
    if "--setup" in sys.argv:
        from setup_models import setup

        setup()
    else:
        main()
