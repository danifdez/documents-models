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

from utils.job_registry import TASK_HANDLERS

logger = logging.getLogger(__name__)

# Ensure task modules under the `tasks` package are imported so their
# `@job_handler` decorators can register handlers in `TASK_HANDLERS`.
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
                    # Ensure module for this job type is imported (prefers config/task)
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


def _ensure_task_for_type(job_type: str) -> bool:
    """Ensure the module that registers a handler for `job_type` is imported.

    Tries candidates under `config.task` first (including loading from
    filesystem if `config` is not a package), then `tasks`.
    Returns True if a handler for `job_type` is present after imports.
    """
    if not job_type:
        return False

    if job_type in TASK_HANDLERS:
        return True

    # candidate base names derived from job type
    bases = [job_type, job_type.replace("-", "_")]
    if "-" in job_type:
        bases.append(job_type.replace("-", ""))
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
        # First, try to find a module file that contains the decorator for this job_type
        # Search in models/tasks
        tasks_dir = models_root / "tasks"
        decorator_patterns = [
            f"@job_handler(\"{job_type}\")", f"@job_handler('{job_type}')"]
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
                        if try_import(mod_name) and job_type in TASK_HANDLERS:
                            return True
                    except Exception:
                        logger.exception(
                            "Failed to import module %s for job type %s", mod_name, job_type)

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
                        if try_import(mod_name) and job_type in TASK_HANDLERS:
                            return True
                    except Exception:
                        logger.exception(
                            "Failed to import config.task module %s for job type %s", mod_name, job_type)

        # Try package-style imports if file-search didn't find a match
        candidates = [
            f"config.task.{base}",
            f"config.task.{base}.{base}",
            f"tasks.{base}.{base}",
            f"tasks.{base}",
        ]

        for mod_name in candidates:
            if try_import(mod_name) and job_type in TASK_HANDLERS:
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

                if job_type in TASK_HANDLERS:
                    return True

    return False


def process_job(job: Dict[str, Any]) -> None:
    from database.job import get_job_database

    job_type = job.get("type")
    handler = TASK_HANDLERS.get(job_type)
    db = get_job_database()

    if not handler:
        logger.warning("No handler for job type: %s", job_type)
        db.update_job_status(job["id"], "failed")
        return

    logger.info("Processing job: %s of type: %s", job["id"], job_type)
    try:
        result = handler(job["payload"])
        db.update_job_result(job["id"], result)
        db.update_job_status(job["id"], "processed")
    except Exception as e:
        logger.error("Job %s failed: %s", job["id"], e)
        db.update_job_status(job["id"], "failed")


# Load all enabled tasks at import time so handlers are registered immediately.
_load_task_modules()
