# Single source of truth for the handler registry: it now lives in
# `common.job_registry` (core structure). Re-exported here so the worker
# (`utils.process_job`) and the pre-existing data tasks that import
# `utils.job_registry` share the exact same `TASK_HANDLERS` dict as the
# conversational handlers registered via `common.job_registry`.
from common.job_registry import TASK_HANDLERS, job_handler  # noqa: F401
