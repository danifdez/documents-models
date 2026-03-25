"""
Prompt loader service.

Loads prompts from individual .md files.
For each prompt, config/tasks/<name>/prompt.md takes priority.
If not present there, falls back to tasks/<task-dir>/prompt.md.
"""

import os
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..'))
_CONFIG_TASKS_DIR = os.path.join(_PROJECT_DIR, 'config', 'tasks')
_TASKS_DIR = os.path.join(_PROJECT_DIR, 'tasks')

_TASK_DIR_MAP = {
    "key-point": "key_points",
    "detect-language": "detect_language",
    "entity-extraction": "entities",
    "document-extraction": "extraction",
    "ingest-content": "ingest",
    "delete-vectors": "ingest",
    "correlation-matrix": "correlation_matrix",
    "group-by": "group_by",
    "time-series": "time_series",
    "pivot-table": "pivot_table",
}


def _read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def _task_dir_name(task_name: str) -> str:
    return _TASK_DIR_MAP.get(task_name, task_name)


def get_prompt(name: str) -> str:
    """Get a prompt by task name.

    Looks up config/tasks/<name>/prompt.md first (user override);
    falls back to tasks/<task-dir>/prompt.md (default).
    Returns empty string if not found in either location.
    """
    config_path = os.path.join(_CONFIG_TASKS_DIR, name, 'prompt.md')
    if os.path.isfile(config_path):
        logger.debug("Loading prompt '%s' from config/tasks/", name)
        return _read_file(config_path)

    task_dir = _task_dir_name(name)
    default_path = os.path.join(_TASKS_DIR, task_dir, 'prompt.md')
    if os.path.isfile(default_path):
        logger.debug("Loading prompt '%s' from tasks/%s/", name, task_dir)
        return _read_file(default_path)

    logger.warning(
        "Prompt '%s' not found in config/tasks/ or tasks/", name)
    return ''
