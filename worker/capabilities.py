from services.model_config import get_all_task_requirements, get_worker_config, get_task_config

GPU = "gpu"
LLM = "llm"
EMBEDDINGS = "embeddings"


def detect_worker_capabilities() -> list:
    """Detect capabilities based on hardware and configuration."""
    from utils.device import HAS_CUDA

    worker = get_worker_config()
    caps = []
    if HAS_CUDA:
        caps.append(GPU)
    if not worker.get("disable_llm", False):
        caps.append(LLM)
    if not worker.get("disable_embeddings", False):
        caps.append(EMBEDDINGS)
    return caps


def get_supported_task_types(capabilities: list) -> list:
    """Return task types this worker can handle based on capabilities and per-task enabled flag."""
    task_requirements = get_all_task_requirements()
    supported = []
    for task_type, required_caps in task_requirements.items():
        task_cfg = get_task_config(task_type)
        if not task_cfg.get("enabled", True):
            continue
        if all(cap in capabilities for cap in required_caps):
            supported.append(task_type)

    return supported
