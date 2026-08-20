import inspect
import logging
import importlib
import importlib.util
import json
import pkgutil
import os
import sys
import types
from pathlib import Path
from typing import Callable, Dict, Any

from common.execution_registry import TASK_HANDLERS


class HandlerCtx:
    """Lightweight context passed to reentrant task handlers.

    Carries the database handle and parent execution id so a handler can enqueue
    child executions (fan-out) without importing `database.execution` directly.
    """

    __slots__ = ("db", "execution_id", "task_type")

    def __init__(self, db, execution_id, task_type):
        self.db = db
        self.execution_id = execution_id
        self.task_type = task_type


def _call_handler(handler, payload, *, state=None, ctx=None):
    """Invoke a handler, passing only the kwargs its signature accepts."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(payload)

    kwargs = {}
    params = sig.parameters
    if "state" in params:
        kwargs["state"] = state
    if "ctx" in params:
        kwargs["ctx"] = ctx
    return handler(payload, **kwargs)


logger = logging.getLogger(__name__)

# Ensure task modules under the `tasks` package are imported so their
# `@execution_handler` decorators can register handlers in `TASK_HANDLERS`.
_TASKS_LOADED = False


def _load_task_modules() -> None:
    global _TASKS_LOADED
    if _TASKS_LOADED:
        return
    models_root = Path(__file__).resolve().parents[1]

    # If there's a tasks config file, load only enabled tasks from it.
    tasks_cfg_path = models_root / "config" / "tasks.json"
    if tasks_cfg_path.exists():
        try:
            cfg = json.loads(tasks_cfg_path.read_text(encoding="utf-8"))
            for jt, jt_cfg in cfg.items():
                try:
                    if not jt_cfg.get("enabled", True):
                        continue
                    # Ensure module for this execution type is imported (prefers config/task)
                    _ensure_task_for_type(jt)
                except Exception:
                    logger.exception("Failed to ensure task for type %s", jt)
            _TASKS_LOADED = True
            return
        except Exception:
            logger.exception("Failed to read tasks config %s", tasks_cfg_path)

    def _import_from_package(pkg_name: str) -> bool:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            return False
        except Exception:
            logger.exception("Failed to import package %s", pkg_name)
            return False

        if not hasattr(pkg, "__path__"):
            return False

        found = False
        for pkg_path in pkg.__path__:
            for root, dirs, files in os.walk(pkg_path):
                # skip bytecode/cache dirs
                dirs[:] = [d for d in dirs if not d.startswith("__")]
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    if filename == "__init__.py":
                        continue
                    if filename.startswith("_"):
                        continue
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, pkg_path)
                    module_name = rel_path[:-3].replace(os.path.sep, ".")
                    full_mod_name = pkg.__name__ + "." + module_name
                    try:
                        importlib.import_module(full_mod_name)
                        found = True
                    except Exception:
                        logger.exception(
                            "Failed to import task module %s", full_mod_name)

        return found

    # Prefer tasks defined under `config.task` (if present), otherwise fall back to `tasks`.
    if _import_from_package("config.task"):
        _TASKS_LOADED = True
        return

    if _import_from_package("tasks"):
        _TASKS_LOADED = True
        return

    logger.debug("No task modules found in config.task or tasks packages")


def _ensure_task_for_type(task_type: str) -> bool:
    """Ensure the module that registers a handler for `task_type` is imported.

    Tries candidates under `config.task` first (including loading from
    filesystem if `config` is not a package), then `tasks`.
    Returns True if a handler for `task_type` is present after imports.
    """
    if not task_type:
        return False

    if task_type in TASK_HANDLERS:
        return True

    # candidate base names derived from execution type
    bases = [task_type, task_type.replace("-", "_")]
    if "-" in task_type:
        bases.append(task_type.replace("-", ""))
    # remove duplicates while preserving order
    seen = set()
    bases = [b for b in bases if not (b in seen or seen.add(b))]

    models_root = Path(__file__).resolve().parents[1]

    def try_import(module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except ModuleNotFoundError:
            return False
        except Exception:
            logger.exception("Error importing %s", module_name)
            return False

    for base in bases:
        # First, try to find a module file that contains the decorator for this task_type
        # Search in models/tasks
        tasks_dir = models_root / "tasks"
        decorator_patterns = [
            f"@execution_handler(\"{task_type}\")", f"@execution_handler('{task_type}')"]
        if tasks_dir.exists():
            for fp in tasks_dir.rglob("*.py"):
                if fp.name == "__init__.py" or fp.name.startswith("_"):
                    continue
                try:
                    txt = fp.read_text(encoding="utf-8")
                except Exception:
                    continue
                if any(p in txt for p in decorator_patterns):
                    # compute module name relative to models_root
                    mod_name = fp.relative_to(
                        models_root).as_posix().replace("/", ".")
                    if mod_name.endswith(".py"):
                        mod_name = mod_name[:-3]
                    try:
                        if try_import(mod_name) and task_type in TASK_HANDLERS:
                            return True
                    except Exception:
                        logger.exception(
                            "Failed to import module %s for execution type %s", mod_name, task_type)

        # Also search in config/task folder on disk (if present)
        config_task_dir = models_root / "config" / "task"
        if config_task_dir.exists():
            for fp in config_task_dir.rglob("*.py"):
                if fp.name == "__init__.py" or fp.name.startswith("_"):
                    continue
                try:
                    txt = fp.read_text(encoding="utf-8")
                except Exception:
                    continue
                if any(p in txt for p in decorator_patterns):
                    mod_name = "config.task." + \
                        fp.relative_to(
                            config_task_dir).as_posix().replace("/", ".")
                    if mod_name.endswith(".py"):
                        mod_name = mod_name[:-3]
                    try:
                        if try_import(mod_name) and task_type in TASK_HANDLERS:
                            return True
                    except Exception:
                        logger.exception(
                            "Failed to import config.task module %s for execution type %s", mod_name, task_type)

        # Try package-style imports if file-search didn't find a match
        candidates = [
            f"config.task.{base}",
            f"config.task.{base}.{base}",
            f"tasks.{base}.{base}",
            f"tasks.{base}",
        ]

        for mod_name in candidates:
            if try_import(mod_name) and task_type in TASK_HANDLERS:
                return True

        # If config.task import failed as a package, try loading from filesystem
        config_task_dir = models_root / "config" / "task"
        if config_task_dir.exists():
            # look for config/task/<base>.py or config/task/<base>/<base>.py
            file_candidates = [
                config_task_dir / f"{base}.py",
                config_task_dir / base / f"{base}.py",
            ]
            for fp in file_candidates:
                if not fp.exists():
                    continue

                # ensure parent packages exist in sys.modules so dotted imports work
                cfg_pkg = sys.modules.get("config")
                if cfg_pkg is None:
                    cfg_pkg = types.ModuleType("config")
                    cfg_pkg.__path__ = [str(models_root / "config")]
                    sys.modules["config"] = cfg_pkg

                cfg_task_pkg = sys.modules.get("config.task")
                if cfg_task_pkg is None:
                    cfg_task_pkg = types.ModuleType("config.task")
                    cfg_task_pkg.__path__ = [str(config_task_dir)]
                    sys.modules["config.task"] = cfg_task_pkg

                mod_name = f"config.task.{base}"
                try:
                    spec = importlib.util.spec_from_file_location(
                        mod_name, str(fp))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[mod_name] = module
                        spec.loader.exec_module(module)
                except Exception:
                    logger.exception(
                        "Failed to load config task module from %s", fp)

                if task_type in TASK_HANDLERS:
                    return True

    return False


def process_execution(execution: Dict[str, Any]) -> None:
    from database.execution import get_execution_database
    from agent.registry import get_agent
    # Importing the tools subpackage registers all built-in tools.
    import agent.tools  # noqa: F401

    task_type = execution.get("task_type")
    db = get_execution_database()
    agent_def = get_agent(task_type)
    agent_max_steps = int(execution.get("max_steps") or 0)
    attempt_id = execution.get("attempt_id")

    # Agent path: only when an agent is registered for this execution type AND
    # the row was enqueued with agent_max_steps > 1 (opt-in per execution).
    if agent_def is not None and agent_max_steps > 1:
        from agent.loop import run_one_step
        logger.info(
            "Processing agent execution: %s of type: %s (iteration %s/%s)",
            execution["execution_id"], task_type, execution.get("step", 0), agent_max_steps,
        )
        try:
            outcome = run_one_step(execution, agent_def, db)
            if outcome.value == "finished":
                current = db.get_execution(execution["execution_id"]) or {}
                if current.get("attempt_id") is None and (
                    current.get("phase") == "backend_finalization"
                    or current.get("status") == "failed"
                ):
                    _maybe_resume_parent(execution, db)
        except Exception as e:
            logger.exception("Agent execution %s failed: %s", execution["execution_id"], e)
            if db.update_execution_status(
                execution["execution_id"],
                "failed",
                attempt_id=attempt_id,
                failure_message=str(e),
            ):
                _maybe_resume_parent(execution, db, error=str(e))
        return

    # One-shot path (existing behaviour).
    handler = TASK_HANDLERS.get(task_type)
    if not handler:
        logger.warning("No handler for execution type: %s", task_type)
        if db.update_execution_status(
            execution["execution_id"],
            "failed",
            attempt_id=attempt_id,
            failure_message=f"No handler for {task_type}",
        ):
            _maybe_resume_parent(execution, db, error=f"no handler for {task_type}")
        return

    logger.info("Processing execution: %s of type: %s", execution["execution_id"], task_type)
    try:
        # Surface the optional input_blob (bytes from the executions.input_blob column)
        # to the handler under a private payload key so handler signatures stay
        # uniform. Handlers that don't need it ignore the key.
        payload = execution.get("payload") or {}
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["executionId"] = execution["execution_id"]
            payload["execution"] = {
                "rootExecutionId": str(execution["root_execution_id"]),
                "executionId": str(execution["execution_id"]),
                "turnId": str(execution["turn_id"]) if execution.get("turn_id") else None,
                "attemptId": str(execution["attempt_id"]) if execution.get("attempt_id") else None,
                "causedByEventId": str(execution["last_event_id"]) if execution.get("last_event_id") else None,
            }
            input_blob = execution.get("input_blob")
            if input_blob is not None:
                payload["_input_blob"] = bytes(input_blob)

        # Reentrant handlers receive persisted state and a context that can
        # enqueue child executions. One-shot handlers ignore both.
        state = execution.get("checkpoint")
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except Exception:
                state = None
        ctx = HandlerCtx(db, execution["execution_id"], task_type)

        result = _call_handler(handler, payload, state=state, ctx=ctx)

        # Reentrant handler signaled a fan-out: persist state, mark waiting,
        # and let _maybe_resume_parent handle each child completion later.
        if isinstance(result, dict) and result.get("_sub_agent_pending_many"):
            new_state = result.get("_state") or {}
            pending = result.get("pending_children") or {}
            new_state["waiting_for_children"] = {str(k): v for k, v in pending.items()}
            if db.update_agent_state(execution["execution_id"], new_state, attempt_id=attempt_id):
                db.update_execution_status(execution["execution_id"], "waiting", attempt_id=attempt_id)
            return

        # Handlers may attach a binary result via the `_result_blob` key; pull
        # it out of the JSON result and persist it into the result_blob column.
        result_blob = None
        if isinstance(result, dict):
            result_blob = result.pop("_result_blob", None)
        if db.update_execution_result(execution["execution_id"], result, result_blob=result_blob, attempt_id=attempt_id):
            finalized = db.update_execution_status(
                execution["execution_id"],
                "running",
                phase="backend_finalization",
                attempt_id=attempt_id,
            )
            if finalized:
                _maybe_resume_parent(execution, db)
    except Exception as e:
        logger.error("Execution %s failed: %s", execution["execution_id"], e)
        if db.update_execution_status(
            execution["execution_id"],
            "failed",
            attempt_id=attempt_id,
            failure_message=str(e),
        ):
            _maybe_resume_parent(execution, db, error=str(e))


def _maybe_resume_parent(execution: Dict[str, Any], db, error: str | None = None) -> None:
    """If this execution has a parent, write the result/error into the parent's
    state and wake the parent back to 'queued' when appropriate.

    Two parent shapes are supported:
      - LLM-driven agent (single child): parent checkpoint transcript[-1]
        has `pending_child == this execution id`.
      - Reentrant task fan-out (N children): parent checkpoint waiting_for_children
        contains this execution id and is integrated by db.resume_parent_with_child.
    """
    parent_id = execution.get("parent_execution_id")
    if not parent_id:
        return
    parent = db.get_execution(parent_id)
    if not parent:
        return
    state = parent.get("checkpoint") or {}
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}

    # Many-children fan-out uses the shared checkpoint integration helper.
    waiting = state.get("waiting_for_children") or {}
    if str(execution["execution_id"]) in waiting:
        from lib.llm.config import get_task_config
        cfg = get_task_config(parent.get("task_type") or "")
        max_retries = int(cfg.get("chunk_max_retries", 0))
        if error is not None:
            db.resume_parent_with_child(
                parent_id, execution["execution_id"], error=error, max_retries=max_retries,
            )
        else:
            fresh = db.get_execution(execution["execution_id"]) or {}
            result = fresh.get("result") or {}
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = {}
            db.resume_parent_with_child(
                parent_id, execution["execution_id"],
                success_result=result if isinstance(result, dict) else {},
                max_retries=max_retries,
            )
        return

    # Single-child LLM-agent path.
    transcript = state.get("transcript") or []
    if transcript:
        last = transcript[-1]
        if last.get("pending_child") == execution["execution_id"]:
            if error is not None:
                last["observation"] = {"error": error, "child_execution_id": execution["execution_id"]}
            else:
                fresh = db.get_execution(execution["execution_id"]) or {}
                last["observation"] = fresh.get("result") or {}
            last.pop("pending_child", None)
            state["transcript"] = transcript
    state.pop("waiting_for_child", None)
    db.update_agent_state(parent_id, state)
    db.wake_waiting_execution(parent_id)


# Load all enabled tasks at import time so handlers are registered immediately.
_load_task_modules()
