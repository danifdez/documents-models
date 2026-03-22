"""
Prompt loader service.

Loads prompts from individual .md files.
For each prompt, config/prompts/<name>.md takes priority.
If not present there, falls back to templates/prompts/<name>.md.
"""

import os
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.join(_BASE_DIR, '..', 'config', 'prompts')
_TEMPLATES_DIR = os.path.join(_BASE_DIR, '..', 'templates', 'prompts')


def _read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_prompt(name: str) -> str:
    """Get a prompt by task name.

    Looks up config/prompts/<name>.md first; falls back to templates/prompts/<name>.md.
    Returns empty string if not found in either location.
    """
    filename = f"{name}.md"

    config_path = os.path.join(_PROMPTS_DIR, filename)
    if os.path.isfile(config_path):
        logger.debug("Loading prompt '%s' from config/prompts/", name)
        return _read_file(config_path)

    template_path = os.path.join(_TEMPLATES_DIR, filename)
    if os.path.isfile(template_path):
        logger.debug("Loading prompt '%s' from templates/prompts/", name)
        return _read_file(template_path)

    logger.warning(
        "Prompt '%s' not found in config/prompts/ or templates/prompts/", name)
    return ''
