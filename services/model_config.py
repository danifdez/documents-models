"""
Model configuration loader.

Loads per-task model configuration from config/models.json.
If models.json does not exist, it is created from templates/models.default.json.
"""

import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..'))
_CONFIG_FILE = os.path.join(_BASE_DIR, '..', 'config', 'models.json')
_DEFAULT_FILE = os.path.join(
    _BASE_DIR, '..', 'templates', 'models.default.json')

_config = None


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config

    if not os.path.exists(_CONFIG_FILE):
        if os.path.exists(_DEFAULT_FILE):
            shutil.copy2(_DEFAULT_FILE, _CONFIG_FILE)
            logger.info("Created config/models.json from defaults")
        else:
            logger.warning(
                "No models.json or models.default.json found in config/")
            _config = {}
            return _config

    with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
        _config = json.load(f)

    logger.info("Loaded model config from %s", _CONFIG_FILE)
    return _config


def get_llm_defaults() -> dict:
    """Get shared LLM default parameters."""
    config = _load_config()
    return config.get('llm_defaults', {})


def get_task_config(task_name: str) -> dict:
    """Get model configuration for a specific task."""
    config = _load_config()
    return config.get('tasks', {}).get(task_name, {})


def get_llm_params(task_name: str) -> dict:
    """Get full LLM parameters for a task, merging task overrides with defaults."""
    defaults = get_llm_defaults()
    task = get_task_config(task_name)

    model_dir = task.get('model_dir', defaults.get('model_dir', 'models'))
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(_PROJECT_DIR, model_dir)
    model_name = task.get('model', defaults.get('model', ''))
    model_path = task.get('model_path', os.path.join(model_dir, model_name))

    return {
        'model_path': model_path,
        'n_ctx': task.get('n_ctx', defaults.get('n_ctx', 32768)),
        'n_threads': task.get('n_threads', defaults.get('n_threads', 4)),
        'n_batch': task.get('n_batch', defaults.get('n_batch', 64)),
        'n_gpu_layers': task.get('n_gpu_layers', defaults.get('n_gpu_layers', 0)),
    }


def reload_config():
    """Force reload config from disk."""
    global _config
    _config = None
    return _load_config()
