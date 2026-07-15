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
# This module lives at core/lib/llm/; the project root (holding config/, tasks/)
# is two levels up.
_PROJECT_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..', '..'))
_CONFIG_TASKS_DIR = os.path.join(_PROJECT_DIR, 'config', 'tasks')
_TASKS_DIR = os.path.join(_PROJECT_DIR, 'tasks')

_TASK_DIR_MAP = {
    "key-point": "key_points",
    "detect-language": "detect_language",
    "entity-extraction": "entities",
    "date-extraction": "dates",
    "document-extraction": "extraction",
    "ingest-content": "ingest",
    "delete-vectors": "ingest",
    "correlation-matrix": "correlation_matrix",
    "group-by": "group_by",
    "time-series": "time_series",
    "pivot-table": "pivot_table",
    "relationship-extraction": "relationship_extraction",
    "relationship-query": "relationship_query",
    "relationship-modify": "relationship_modify",
}


def _read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def _task_dir_name(task_name: str) -> str:
    return _TASK_DIR_MAP.get(task_name, task_name)


def get_prompt(name: str, filename: str = 'prompt.md') -> str:
    """Get a task prompt by task name.

    Looks up config/tasks/<name>/<filename> first (user override);
    falls back to tasks/<task-dir>/<filename> (default). The optional
    `filename` lets a task keep more than one prompt (e.g. a main
    prompt.md and a refine_prompt.md), each independently overridable.
    Returns empty string if not found in either location.
    """
    config_path = os.path.join(_CONFIG_TASKS_DIR, name, filename)
    if os.path.isfile(config_path):
        logger.debug("Loading prompt '%s' (%s) from config/tasks/", name, filename)
        return _read_file(config_path)

    task_dir = _task_dir_name(name)
    default_path = os.path.join(_TASKS_DIR, task_dir, filename)
    if os.path.isfile(default_path):
        logger.debug("Loading prompt '%s' (%s) from tasks/%s/", name, filename, task_dir)
        return _read_file(default_path)

    logger.warning(
        "Prompt '%s' (%s) not found in config/tasks/ or tasks/", name, filename)
    return ''


def load_prompt(base_dir: str, filename: str) -> str:
    """Load a prompt that is NOT a user-overridable task prompt: the system
    prompts of the agent framework, shared services and subagents. Reads
    <base_dir>/<filename> directly, with no config/tasks override layer —
    those internal behaviours aren't meant to be re-tuned per workspace the
    way content tasks are. Returns empty string if the file is missing.
    """
    path = os.path.join(base_dir, filename)
    try:
        return _read_file(path)
    except OSError:
        logger.warning("Prompt file not found: %s", path)
        return ''
