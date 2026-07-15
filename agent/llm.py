"""Resolve a ModelSpec into a cached LLMService instance."""

import os
from typing import Optional

from agent.types import ModelSpec
from services.llm_service import get_llm_service
from lib.llm.config import _PROJECT_DIR, get_llm_defaults  # type: ignore


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isabs(name_or_path):
        return name_or_path
    defaults = get_llm_defaults()
    model_dir = defaults.get("model_dir", "models")
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(_PROJECT_DIR, model_dir)
    return os.path.join(model_dir, name_or_path)


def _resolve_lora_path(name_or_path: Optional[str]) -> Optional[str]:
    if not name_or_path:
        return None
    if os.path.isabs(name_or_path):
        return name_or_path
    defaults = get_llm_defaults()
    model_dir = defaults.get("model_dir", "models")
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(_PROJECT_DIR, model_dir)
    return os.path.join(model_dir, name_or_path)


def get_llm_for_spec(spec: ModelSpec):
    """Get a cached LLMService for the given ModelSpec, using shared LLM defaults."""
    defaults = get_llm_defaults()
    return get_llm_service(
        model_path=_resolve_model_path(spec.path),
        n_ctx=int(defaults.get("n_ctx", 32768)),
        n_threads=int(defaults.get("n_threads", 4)),
        n_batch=int(defaults.get("n_batch", 64)),
        n_gpu_layers=int(defaults.get("n_gpu_layers", 0)),
        lora_path=_resolve_lora_path(spec.lora),
        lora_scale=float(spec.lora_scale),
    )
