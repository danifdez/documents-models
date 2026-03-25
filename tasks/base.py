"""
Task interface definition for the models worker.

Every task must follow this structure:

1. Directory: tasks/<task_name>/
   - <task_name>.py  -- handler module
   - prompt.md       -- default prompt template (optional, only for LLM tasks)

2. Handler: decorated with @job_handler("<task-type>")
   - Receives a payload dict
   - Returns a result dict

3. Configuration: entry in common/tasks.default.json under "tasks.<task-type>"
   - enabled: bool        -- whether the task is active
   - type: str            -- model type (llm, sentence-transformer, seq2seq, etc.)
   - capabilities: list   -- required worker capabilities (e.g. ["llm", "embeddings"])
   - model: str           -- model name or path (optional)
   - ...additional task-specific parameters

4. Registration: import the module in utils/process_job.py

Users can override task behavior via:
- config/tasks/<task-type>/prompt.md   -- custom prompt
- config/tasks/<task-type>/config.json -- parameter overrides
- config/tasks.json tasks.<task-type> -- direct config edits
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TaskDefinition:
    """Documents the expected shape of a task entry in tasks.json."""
    name: str
    type: str
    enabled: bool = True
    capabilities: List[str] = field(default_factory=list)
    model: Optional[str] = None
    model_prefix: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
