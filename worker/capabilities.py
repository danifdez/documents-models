import os

# Capability tags
GPU = "gpu"
LLM = "llm"
EMBEDDINGS = "embeddings"

# What capabilities each task type requires (empty = any worker can handle it)
TASK_REQUIREMENTS = {
    "detect-language":     [],
    "document-extraction": [],
    "embedding":           [EMBEDDINGS],
    "keywords":            [LLM],
    "key-point":           [LLM],
    "summarize":           [],
    "translate":           [],
    "entity-extraction":   [],
    "search":              [EMBEDDINGS],
    "ask":                 [LLM, EMBEDDINGS],
    "ingest-content":      [EMBEDDINGS],
    "delete-vectors":      [],
    "dataset-stats":       [],
}


def detect_worker_capabilities() -> list:
    """Detect capabilities based on hardware and configuration."""
    from utils.device import HAS_CUDA

    caps = []
    if HAS_CUDA:
        caps.append(GPU)
    if os.getenv("WORKER_DISABLE_LLM", "false").lower() != "true":
        caps.append(LLM)
    if os.getenv("WORKER_DISABLE_EMBEDDINGS", "false").lower() != "true":
        caps.append(EMBEDDINGS)
    return caps


def get_supported_task_types(capabilities: list) -> list:
    """Return task types this worker can handle based on its capabilities and task filters."""
    supported = []
    for task_type, required_caps in TASK_REQUIREMENTS.items():
        if all(cap in capabilities for cap in required_caps):
            supported.append(task_type)

    enabled = os.getenv("WORKER_ENABLED_TASKS", "").strip()
    disabled = os.getenv("WORKER_DISABLED_TASKS", "").strip()

    if enabled:
        allowed = [t.strip() for t in enabled.split(",") if t.strip()]
        supported = [t for t in supported if t in allowed]
    if disabled:
        blocked = [t.strip() for t in disabled.split(",") if t.strip()]
        supported = [t for t in supported if t not in blocked]

    return supported
