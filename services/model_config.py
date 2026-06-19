"""
Configuration loader.

Loads general configuration from config/config.json and task
configuration from config/tasks.json.
If either file does not exist, it is created from the corresponding
default in common/.
Per-task overrides can be placed in config/tasks/<task-name>/config.json.
"""

import json
import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

# When packaged with PyInstaller (standalone), bundled data files (common/,
# config/, tasks/*) live under sys._MEIPASS; in a dev checkout they sit next to
# this module's parent directory.
_FROZEN = getattr(sys, 'frozen', False)
if _FROZEN:
    _PROJECT_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..'))

_CONFIG_DIR = os.path.join(_PROJECT_DIR, 'config')
_CONFIG_DEFAULT = os.path.join(_PROJECT_DIR, 'common', 'config.default.json')
_TASKS_DEFAULT = os.path.join(_PROJECT_DIR, 'common', 'tasks.default.json')

# The active config/tasks can be redirected to writable files outside the bundle.
# Standalone writes config there with the embedded services' dynamic ports + the
# chosen profile's feature flags; both are deep-merged on top of the defaults.
_CONFIG_FILE = os.environ.get('MODELS_CONFIG_PATH') or os.path.join(_CONFIG_DIR, 'config.json')
_TASKS_FILE = os.environ.get('MODELS_TASKS_PATH') or os.path.join(_CONFIG_DIR, 'tasks.json')

_config = None
_tasks = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto a copy of base."""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_file(target: str, default: str, label: str) -> None:
    # In a frozen bundle the data dir is read-only; never try to write there.
    if _FROZEN:
        return
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(default):
            shutil.copy2(default, target)
            logger.info("Created %s from defaults", label)
        else:
            logger.warning("Default file not found: %s", default)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    _ensure_file(_CONFIG_FILE, _CONFIG_DEFAULT, 'config/config.json')
    _config = _deep_merge(_load_json(_CONFIG_DEFAULT), _load_json(_CONFIG_FILE))
    logger.info("Loaded config from %s (merged over defaults)", _CONFIG_FILE)
    return _config


def _load_tasks() -> dict:
    global _tasks
    if _tasks is not None:
        return _tasks
    _ensure_file(_TASKS_FILE, _TASKS_DEFAULT, 'config/tasks.json')
    _tasks = _deep_merge(_load_json(_TASKS_DEFAULT), _load_json(_TASKS_FILE))
    logger.info("Loaded tasks from %s (merged over defaults)", _TASKS_FILE)
    return _tasks


def get_config() -> dict:
    """Get the general configuration dict."""
    return _load_config()


def get_tasks() -> dict:
    """Get the full tasks configuration dict."""
    return _load_tasks()


def get_llm_defaults() -> dict:
    """Get shared LLM default parameters."""
    return _load_config().get('llm_defaults', {})


def get_rag_config() -> dict:
    """Get RAG configuration."""
    return _load_config().get('rag', {})


def get_worker_config() -> dict:
    """Get worker configuration."""
    return _load_config().get('worker', {})


def get_task_config(task_name: str) -> dict:
    """Get configuration for a specific task.

    Merges config/tasks/<task_name>/config.json on top if present.
    """
    tasks = _load_tasks()
    base = dict(tasks.get(task_name, {}))

    override_path = os.path.join(
        _CONFIG_DIR, 'tasks', task_name, 'config.json')
    if os.path.isfile(override_path):
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
        base.update(overrides)

    return base


def get_all_task_names() -> list:
    """Get all task names."""
    return list(_load_tasks().keys())


def get_all_task_requirements() -> dict:
    """Get capabilities dict for all tasks."""
    tasks = _load_tasks()
    return {name: t.get('capabilities', []) for name, t in tasks.items()}


def get_llm_params(task_name: str, model_name: str | None = None) -> dict:
    """Get full LLM parameters for a task, merging task overrides with defaults.

    If `model_name` is provided, it overrides the task's `model` field — useful when
    the task's primary model is not an LLM (e.g. spaCy) and an LLM fallback needs a
    different model from the same task config.
    """
    defaults = get_llm_defaults()
    task = get_task_config(task_name)

    model_dir = task.get('model_dir', defaults.get('model_dir', 'models'))
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(_PROJECT_DIR, model_dir)
    if model_name is None:
        model_name = task.get('model', defaults.get('model', ''))
    if 'model_path' in task:
        model_path = task['model_path']
    else:
        model_path = os.path.join(model_dir, model_name)

    lora_name = task.get('lora_model')
    lora_path = task.get('lora_path')
    if lora_path is None and lora_name:
        lora_path = lora_name if os.path.isabs(lora_name) else os.path.join(model_dir, lora_name)
    lora_scale = task.get('lora_scale', 1.0)

    return {
        'model_path': model_path,
        'n_ctx': task.get('n_ctx', defaults.get('n_ctx', 32768)),
        'n_threads': task.get('n_threads', defaults.get('n_threads', 4)),
        'n_batch': task.get('n_batch', defaults.get('n_batch', 64)),
        'n_gpu_layers': task.get('n_gpu_layers', defaults.get('n_gpu_layers', 0)),
        'lora_path': lora_path,
        'lora_scale': lora_scale,
    }


def reload_config():
    """Force reload all config from disk."""
    global _config, _tasks
    _config = None
    _tasks = None
    _load_config()
    _load_tasks()
