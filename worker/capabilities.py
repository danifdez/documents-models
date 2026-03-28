from services.model_config import get_all_task_requirements, get_worker_config, get_task_config, get_config

GPU = "gpu"
LLM = "llm"
EMBEDDINGS = "embeddings"

# Map task types to feature flag keys in config.features
TASK_FEATURE_MAP = {
    "entity-extraction": "entities",
    "distribution": "datasets",
    "correlation": "datasets",
    "correlation-matrix": "datasets",
    "group-by": "datasets",
    "time-series": "datasets",
    "outliers": "datasets",
    "pivot-table": "datasets",
    "summary": "datasets",
    "query": "datasets",
    "chart": "datasets",
    "image-generate": "canvas",
    "image-edit": "canvas",
    "ingest-content": "rag",
    "search": "rag",
    "ask": "rag",
    "embedding": "rag",
    "relationship-extraction": "relationships",
    "relationship-query": "relationships",
    "relationship-modify": "relationships",
}


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
    """Return task types this worker can handle based on capabilities, per-task enabled flag, and feature flags."""
    task_requirements = get_all_task_requirements()
    features = get_config().get("features", {})
    supported = []
    for task_type, required_caps in task_requirements.items():
        task_cfg = get_task_config(task_type)
        if not task_cfg.get("enabled", True):
            continue
        # Check feature flag
        feature_key = TASK_FEATURE_MAP.get(task_type)
        if feature_key and not features.get(feature_key, True):
            continue
        if all(cap in capabilities for cap in required_caps):
            supported.append(task_type)

    return supported
