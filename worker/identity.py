import os
import sys
import uuid

from lib.llm.config import get_worker_config

def _default_data_dir() -> str:
    # In a PyInstaller bundle the source tree lives inside the archive, so a path
    # relative to __file__ (…/worker/..) can't be resolved on disk. Persist next
    # to the executable, which is writable in standalone.
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# MODELS_DATA_DIR points at a writable location for the persisted worker id.
_DATA_DIR = os.environ.get('MODELS_DATA_DIR') or _default_data_dir()
WORKER_ID_FILE = os.path.join(_DATA_DIR, '.worker_id')


def _load_or_create_worker_id() -> str:
    worker_cfg = get_worker_config()
    cfg_id = worker_cfg.get("id", "")
    if cfg_id:
        return cfg_id
    try:
        with open(WORKER_ID_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        new_id = str(uuid.uuid4())
        os.makedirs(os.path.dirname(WORKER_ID_FILE) or '.', exist_ok=True)
        with open(WORKER_ID_FILE, 'w') as f:
            f.write(new_id)
        return new_id


WORKER_ID = _load_or_create_worker_id()
_worker_cfg = get_worker_config()
WORKER_NAME = _worker_cfg.get("name", "") or f"worker-{WORKER_ID[:8]}"
HEARTBEAT_INTERVAL = int(_worker_cfg.get("heartbeat_interval", 15))
MAXIMUM_CONCURRENCY = _worker_cfg.get("maximum_concurrency", 2)
if (
    isinstance(MAXIMUM_CONCURRENCY, bool)
    or not isinstance(MAXIMUM_CONCURRENCY, int)
    or not 1 <= MAXIMUM_CONCURRENCY <= 64
):
    raise ValueError("worker.maximum_concurrency must be between 1 and 64")


def worker_data_dir() -> str:
    return _DATA_DIR
