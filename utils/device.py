import os
import logging
import multiprocessing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware detection (runs once at import time)
# ---------------------------------------------------------------------------

CPU_COUNT = multiprocessing.cpu_count() or 4

# RAM detection via /proc/meminfo (Linux) — no extra dependency
RAM_GB = 0.0
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                RAM_GB = round(int(line.split()[1]) / (1024 ** 2), 1)
                break
except Exception:
    pass

# GPU / CUDA detection
HAS_CUDA = False
GPU_NAME = None
VRAM_GB = 0.0

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        GPU_NAME = torch.cuda.get_device_name(0)
        VRAM_GB = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
        )
except ImportError:
    pass
except Exception:
    pass

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if HAS_CUDA else "cpu"


def get_optimal_threads() -> int:
    """Return optimal thread count for CPU-bound work.

    Respects the ``LLM_N_THREADS`` env var when set.
    """
    env_val = os.getenv("LLM_N_THREADS")
    if env_val is not None:
        return int(env_val)
    return min(max(1, CPU_COUNT - 1), 8)


def get_gpu_layers() -> int:
    """Return GPU layer count for llama-cpp.

    Respects the ``LLM_N_GPU_LAYERS`` env var when set.
    Without GPU defaults to 0 (all on CPU); with GPU defaults to -1 (all on GPU).
    """
    env_val = os.getenv("LLM_N_GPU_LAYERS")
    if env_val is not None:
        return int(env_val)
    return -1 if HAS_CUDA else 0


def get_spacy_model() -> str:
    """Return the best available spaCy model name.

    Respects the ``SPACY_MODEL`` env var when set.
    With GPU uses the transformer model; on CPU tries lg then sm.
    """
    env_val = os.getenv("SPACY_MODEL")
    if env_val is not None:
        return env_val
    if HAS_CUDA:
        return "en_core_web_trf"
    try:
        import spacy
        spacy.load("en_core_web_lg")
        return "en_core_web_lg"
    except Exception:
        return "en_core_web_sm"


def log_hardware_summary() -> None:
    """Print a hardware summary block to stdout and the logger."""
    lines = [
        "=" * 50,
        "  HARDWARE CONFIGURATION",
        "=" * 50,
        f"  CPU cores:       {CPU_COUNT}",
        f"  RAM:             {RAM_GB} GB",
        f"  CUDA available:  {HAS_CUDA}",
    ]
    if HAS_CUDA:
        lines.append(f"  GPU:             {GPU_NAME}")
        lines.append(f"  VRAM:            {VRAM_GB} GB")
    lines += [
        "-" * 50,
        f"  Device:          {get_device()}",
        f"  LLM GPU layers:  {get_gpu_layers()}",
        f"  LLM threads:     {get_optimal_threads()}",
        f"  spaCy model:     {get_spacy_model()}",
        "=" * 50,
    ]
    summary = "\n".join(lines)
    logger.info(summary)
