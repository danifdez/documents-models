import logging
import time
import signal
import sys
from database.job import get_job_database
from utils.process_job import process_job
from utils.device import log_hardware_summary, HAS_CUDA, CPU_COUNT, RAM_GB, GPU_NAME, VRAM_GB
from worker.capabilities import detect_worker_capabilities
from worker.identity import (
    WORKER_ID,
    WORKER_NAME,
    register_worker,
    start_heartbeat_thread,
    deregister_worker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 1000  # milliseconds


def main():
    log_hardware_summary()

    # Detect and register capabilities
    capabilities = detect_worker_capabilities()
    metadata = {
        "cpu_count": CPU_COUNT,
        "ram_gb": RAM_GB,
        "has_cuda": HAS_CUDA,
        "gpu_name": GPU_NAME,
        "vram_gb": VRAM_GB,
    }

    register_worker(capabilities, metadata)
    logger.info("Worker registered: %s (%s)", WORKER_NAME, WORKER_ID)
    logger.info("Capabilities: %s", capabilities)

    # Start heartbeat thread
    start_heartbeat_thread()

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Worker %s shutting down...", WORKER_NAME)
        deregister_worker()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    db = get_job_database()
    logger.info("Job service started. Polling for pending jobs...")

    while True:
        # Requeue jobs from dead workers
        db.requeue_stale_jobs()

        # Claim and process a job
        job = db.claim_pending_job(WORKER_ID, capabilities)
        if job:
            process_job(job)

        time.sleep(POLL_INTERVAL_MS / 1000.0)


if __name__ == "__main__":
    if "--setup" in sys.argv:
        # Pre-download all ML models without starting the worker
        from setup_models import setup
        setup()
    else:
        main()
