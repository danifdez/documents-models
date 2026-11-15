"""Lifecycle and configuration for the Models llama-server process.

Models executions generate over HTTP through this endpoint. `manage llama
start` normally owns the process; `ensure_server` can start the same configured
binary when an execution needs it and no server answers. An explicitly
configured local or remote endpoint is reused without starting another process.
`engine_defaults()` remains the source of truth for model, context and slots.
"""

import atexit
import hashlib
import hmac
import re
import stat
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from lib.llm.config import (
    active_deployments,
    get_llm_defaults,
    get_tasks,
    llm_params_for,
)

logger = logging.getLogger(__name__)

# How long to wait for a server we started to finish loading the model. A 8B
# quant off a cold page cache takes its time.
STARTUP_TIMEOUT_S = float(os.environ.get("LLAMA_SERVER_STARTUP_TIMEOUT", "300"))

# The worker needs a deterministic default when no endpoint is configured.
DEFAULT_URL = "http://127.0.0.1:18080"

_lock = threading.Lock()
_child: Optional[subprocess.Popen] = None
_ready_url: Optional[str] = None


def server_url() -> str:
    """Resolve the Models inference endpoint. Env wins over config and default."""
    url = (
        os.environ.get("LLAMA_SERVER_URL")
        or get_llm_defaults().get("server_url")
        or DEFAULT_URL
    )
    return url.strip().rstrip("/")


def _bundled_binaries() -> List[str]:
    """Where documents-dev keeps the engine, most specific first.

    `manage llama install` drops it in `bin/llama/` at the repo root, next to the
    submodules that use it. In a PyInstaller bundle it ships inside the app
    directory instead, so both the frozen and the checkout layout are tried.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(here, ".."))
    roots = [models_dir, os.path.abspath(os.path.join(models_dir, ".."))]
    if getattr(sys, "frozen", False):
        roots = [
            os.path.dirname(os.path.abspath(sys.executable)),
            getattr(sys, "_MEIPASS", models_dir),
        ] + roots
    names = [os.path.join("bin", "llama", "llama-server"),
             os.path.join("bin", "llama-server"),
             os.path.join("llama", "llama-server"),
             "llama-server"]
    return [os.path.join(root, name) for root in roots for name in names]


def server_binary() -> str:
    """The llama-server to run, or '' when there is none to be found.

    An explicit `LLAMA_SERVER_BIN` (or `llm_defaults.server_bin`) is taken as
    given — that is how you point the engine at a CUDA build or remote runtime.
    Otherwise the Models copy is used, and only then `llama-server` on PATH.
    """
    explicit = (
        os.environ.get("LLAMA_SERVER_BIN")
        or get_llm_defaults().get("server_bin")
        or ""
    ).strip()
    if explicit:
        return os.path.expanduser(explicit)
    for candidate in _bundled_binaries():
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("llama-server") or ""


def is_alive(url: str, timeout: float = 2.0) -> bool:
    """True when the server answers /health and has the model loaded.

    llama-server replies 503 while it is still loading, which urllib raises as
    an HTTPError — that is a live server, but not a usable one yet.
    """
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            return resp.getcode() == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def props(
    url: str, timeout: float = 5.0, model_id: str | None = None,
) -> Dict[str, Any]:
    """What the server is serving: model path, context size, sampling defaults.

    Empty dict when it can't be asked — callers use this to warn, never to
    decide whether to run.
    """
    try:
        suffix = ""
        if model_id:
            suffix = "?" + urllib.parse.urlencode({"model": model_id})
        with urllib.request.urlopen(f"{url}/props{suffix}", timeout=timeout) as resp:
            raw = resp.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                return {}
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError, UnicodeDecodeError):
        return {}


def lora_adapters(
    url: str, timeout: float = 5.0, model_id: str | None = None,
) -> List[Dict[str, Any]]:
    """The adapters the server has loaded: `[{"id", "path", "scale"}, …]`.

    Adapters can only be attached when the server starts (`--lora`), so this is
    a read of what is already there. Empty list when it can't be asked, like
    `props`: callers use it to resolve an id or to inform, never to decide
    whether to run.
    """
    try:
        suffix = ""
        if model_id:
            suffix = "?" + urllib.parse.urlencode({"model": model_id})
        with urllib.request.urlopen(f"{url.rstrip('/')}/lora-adapters{suffix}",
                                    timeout=timeout) as resp:
            raw = resp.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                return []
            loaded = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError, UnicodeDecodeError):
        return []
    return loaded if isinstance(loaded, list) else []


def verified_lora_adapter(
    url: str, path: str, model_id: str | None = None,
    expected_sha256: str | None = None,
) -> Dict[str, Any]:
    """Verify local bytes and select the server-reported exact path, never a basename.

    The server reports a path and ID, not a digest of its in-memory weights.
    Only the configured local path is opened; remote paths are never read.
    """
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Invalid LoRA checksum")
        expected_sha256 = expected_sha256.lower()
    try:
        initial = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("LoRA must be a regular file")
        flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        with os.fdopen(os.open(path, flags), "rb") as file:
            before = os.fstat(file.fileno())
            if not stat.S_ISREG(before.st_mode) or (initial.st_dev, initial.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("LoRA must be a regular file")
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("LoRA changed during verification")
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(file.fileno())
            current = os.stat(path, follow_symlinks=False)
            signatures = {(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
                          for value in (before, after, current)}
            if file.read(1) or len(signatures) != 1:
                raise ValueError("LoRA changed during verification")
    except OSError:
        raise ValueError("LoRA file verification failed") from None
    checksum = digest.hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(checksum, expected_sha256):
        raise ValueError("LoRA checksum mismatch")
    wanted = os.path.abspath(path)
    catalog = lora_adapters(url, model_id=model_id)
    loaded = [item for item in catalog
              if isinstance(item, dict) and isinstance(item.get('path'), str)
              and os.path.isabs(item['path']) and os.path.normpath(item['path']) == wanted]
    if len(loaded) != 1 or type(loaded[0].get('id')) is not int or not 0 <= loaded[0]['id'] <= 2**31 - 1:
        raise ValueError("LoRA loaded identity is unavailable or ambiguous")
    if sum(isinstance(item, dict) and item.get('id') == loaded[0]['id'] for item in catalog) != 1:
        raise ValueError("LoRA loaded identity is unavailable or ambiguous")
    return {'id': loaded[0]['id'], 'sha256': checksum,
            'verification': 'local_file_sha256', 'binding': 'server_reported_exact_path',
            'expectedChecksumMatched': expected_sha256 is not None}


def loaded_model(url: str) -> str:
    """Basename of the .gguf the server has loaded, or '' if unknown."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=5) as resp:
            models = json.loads(resp.read().decode("utf-8")).get("data") or []
        loaded = [
            model.get("id") for model in models
            if isinstance(model, dict)
            and isinstance(model.get("status"), dict)
            and model["status"].get("value") == "loaded"
        ]
        if loaded:
            return str(loaded[0])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError, AttributeError):
        pass
    data = props(url)
    path = data.get("model_path") or data.get("model") or ""
    if not path and isinstance(data.get("default_generation_settings"), dict):
        path = data["default_generation_settings"].get("model") or ""
    return os.path.basename(path) if path else ""


def _most_used_model() -> str:
    """The .gguf most task configs name, which is the model to serve.

    Nothing declares "the" model of the installation: each task names the one it
    wants, and in practice they nearly all name the same. Serving the most
    popular one means the fewest tasks get a model other than the one they asked
    for (they still get an answer — `LLMService` warns and uses what is loaded).
    """
    counts: Dict[str, int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("model", "llm_model") and isinstance(value, str) \
                        and value.endswith(".gguf"):
                    counts[value] = counts.get(value, 0) + 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(get_tasks())
    if not counts:
        return ""
    # Alphabetical tie-break, so the same config always picks the same model.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def engine_defaults() -> Dict[str, Any]:
    """The one definition of documents-dev's engine: what to load and how.

    Used both by `manage llama start` and by the last-resort spawn from a execution, so
    the engine is the same server no matter who happened to start it. Every field
    can be overridden from the environment, which is how the service manager (or
    a one-off run) changes the engine without touching the config.
    """
    model = os.environ.get("LLAMA_SERVER_MODEL", "").strip() \
        or get_llm_defaults().get("model") \
        or _most_used_model()
    if not model:
        raise RuntimeError(
            "No model to serve: no LLAMA_SERVER_MODEL, no llm_defaults.model, and "
            "no task naming a .gguf in config/tasks.json."
        )
    task = {"model_path": model} if os.path.isabs(model) else {"model": model}
    params = llm_params_for(task)

    def _int(name: str, fallback: Any) -> int:
        raw = os.environ.get(name, "").strip()
        try:
            return int(raw) if raw else int(fallback)
        except (TypeError, ValueError):
            return int(fallback)

    models_dir = os.environ.get("LLAMA_SERVER_MODELS_DIR", "").strip()

    def _gpu_layers(name: str, fallback: Any) -> int | str:
        raw = os.environ.get(name, "").strip()
        if raw in {"auto", "all"}:
            return raw
        return _int(name, fallback)

    return {
        "model_path": params["model_path"],
        "models_dir": models_dir or None,
        "models_max": max(1, _int("LLAMA_SERVER_MODELS_MAX", 1)),
        "n_ctx": _int("LLAMA_SERVER_CTX", params["n_ctx"]),
        "n_threads": _int("LLAMA_SERVER_THREADS", params["n_threads"]),
        "n_gpu_layers": _gpu_layers(
            "LLAMA_SERVER_GPU_LAYERS", params["n_gpu_layers"],
        ),
        # Every adapter deployed on a task, loaded once even if several tasks
        # share it. They have to be here because llama-server only attaches
        # adapters at startup: a task whose LoRA didn't make it into the command
        # line would silently answer as the base model.
        "lora_paths": _deployed_adapters(),
    }


def _deployed_adapters() -> List[str]:
    """The distinct adapter paths deployed on tasks, in a stable order.

    Missing files are dropped rather than passed on: llama-server refuses to
    start when `--lora` names a file that isn't there, and one stale deployment
    must not leave the whole installation without an engine.
    """
    paths = []
    for deployment in active_deployments().values():
        path = os.path.abspath(deployment["path"])
        if path not in paths and os.path.isfile(path):
            paths.append(path)
    return paths


def engine_cmd(binary: str, url: str, engine: Dict[str, Any]) -> List[str]:
    """The command line that starts the engine.

    `engine` is what `engine_defaults()` returns, passed whole so that a new
    field there reaches the command line without a signature to widen.

    Its `n_ctx` is the context one caller gets, which is what the task configs
    mean by it. llama-server's --ctx-size is the total shared by all slots, so it
    is multiplied here — otherwise adding a slot would quietly halve everybody's
    context and long prompts would start getting truncated.
    """
    host, port = _split_host_port(url)
    # One slot per concurrent assignment: independent worker executions would
    # otherwise queue behind each other. Each
    # slot keeps its own KV cache, which is what makes `cache_prompt` worth
    # anything.
    slots = max(1, int(os.environ.get("LLAMA_SERVER_SLOTS", "2")))
    cmd = [
        binary,
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(engine["n_ctx"] * slots),
        "--n-gpu-layers", str(engine["n_gpu_layers"]),
        "--parallel", str(slots),
        # 8-bit KV cache: what pays for a full context per slot. At f16 the cache
        # of two 8k slots is over a gigabyte of VRAM on an 8B model, which on a
        # laptop GPU is the difference between the weights fitting and not.
        "--cache-type-k", os.environ.get("LLAMA_SERVER_CACHE_TYPE", "q8_0"),
        "--cache-type-v", os.environ.get("LLAMA_SERVER_CACHE_TYPE", "q8_0"),
        # Renders the model's own chat template, which is what tool calling
        # (`chat_with_tools`) needs to produce tool_calls instead of prose.
        "--jinja",
        # /metrics: with everything generating through one server, its counters are
        # the only place where "what is the machine actually spending tokens on"
        # can be seen at all.
        "--metrics",
    ]
    if engine.get("models_dir"):
        cmd += [
            "--models-dir", str(engine["models_dir"]),
            "--models-max", str(engine["models_max"]),
        ]
    else:
        cmd += ["--model", engine["model_path"]]
    if engine["n_threads"]:
        cmd += ["--threads", str(engine["n_threads"])]
    for adapter in engine.get("lora_paths") or []:
        cmd += ["--lora", str(adapter)]
    if engine.get("lora_paths"):
        # Loaded but not applied: with several adapters on one server, applying
        # them all at once would mix them into every answer. Each request cites
        # the id it wants (`lora: [{"id", "scale"}]`), so a task without an
        # adapter keeps talking to the plain base model.
        cmd += ["--lora-init-without-apply"]
    extra = os.environ.get("LLAMA_SERVER_ARGS", "").strip()
    if extra:
        cmd += shlex.split(extra)
    return cmd


def ensure_server(model_path: str = "") -> str:
    """Return the URL of a live Models engine, starting one if needed.

    `model_path` is what the calling task wants. A router selects it per request;
    on a legacy single-model engine it remains only a startup fallback.

    Raises RuntimeError when there is no engine and no way to start one, which is
    a configuration problem: start it with `manage llama start`, or point
    LLAMA_SERVER_BIN at a binary.
    """
    url = server_url()

    global _ready_url
    if _ready_url == url and is_alive(url):
        return url

    with _lock:
        # Somebody may have started it while we waited for the lock.
        if is_alive(url):
            _ready_url = url
            return url
        _spawn(url, model_path)
        _ready_url = url
        return url


def _engine_environment() -> dict[str, str]:
    # Exporter endpoints, auth headers and TLS keys belong to the application,
    # not the inference engine. Preserve the parent's copy for worker restarts.
    return {
        key: value for key, value in os.environ.items()
        if not key.startswith("OTEL_EXPORTER_OTLP")
    }


def _spawn(url: str, task_model_path: str) -> None:
    binary = server_binary()
    if not binary:
        raise RuntimeError(
            f"Nothing is listening on {url} and no llama-server was found, so there "
            "is no engine to talk to. Run `manage llama install` to put one in "
            "documents-dev, start it with `manage llama start`, or point "
            "LLAMA_SERVER_BIN at a binary."
        )

    engine = engine_defaults()
    if not engine.get("models_dir") and not os.path.isfile(engine["model_path"]):
        # The installation's model is missing but the execution's is there: serve that
        # rather than refuse to run.
        if task_model_path and os.path.isfile(task_model_path):
            logger.warning(
                "Engine model %s not found; serving %s instead.",
                engine["model_path"], task_model_path,
            )
            engine["model_path"] = task_model_path
        else:
            raise RuntimeError(f"Model file not found: {engine['model_path']}")

    cmd = engine_cmd(binary, url, engine)
    log_path = os.environ.get("LLAMA_SERVER_LOG", "/tmp/llama-server.log")
    logger.warning(
        "No engine answering at %s — starting one for this worker. It should be a "
        "documents-dev service (`manage llama start`).", url,
    )
    logger.info("Starting Documents llama-server: %s (log: %s)", " ".join(cmd), log_path)
    log = open(log_path, "ab")
    global _child
    # Its own process group, so a Ctrl-C in the worker's terminal doesn't reach
    # the server before we get to shut it down in order.
    _child = subprocess.Popen(
        cmd, stdout=log, stderr=log, start_new_session=True,
        env=_engine_environment(),
    )
    atexit.register(stop)

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if _child.poll() is not None:
            raise RuntimeError(
                f"llama-server died while loading (exit {_child.returncode}). See {log_path}."
            )
        if is_alive(url):
            logger.info("Shared llama-server ready at %s", url)
            return
        time.sleep(1.0)
    stop()
    raise RuntimeError(
        f"llama-server did not become ready within {STARTUP_TIMEOUT_S:.0f}s. See {log_path}."
    )


def stop() -> None:
    """Kill the server we started. Does nothing when the engine is external."""
    global _child
    if _child is None:
        return
    child, _child = _child, None
    if child.poll() is not None:
        return
    logger.info("Stopping Documents llama-server (pid %s)", child.pid)
    try:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
    except OSError:
        pass


def _split_host_port(url: str) -> tuple:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host or "127.0.0.1", int(port or 8080)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the engine in the foreground, for a service manager to supervise.

    `python -m services.llama_server` replaces this process with llama-server, so
    whoever started it (`manage llama start`) keeps one PID to watch and to stop,
    and the engine's own log is this process's stdout.

    Flags, for the service manager's benefit rather than a human's:
      --url        print where the engine is expected to answer
      --print-cmd  print the command that would be run, and exit
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    url = server_url()
    if "--url" in argv:
        print(url)
        return 0

    binary = server_binary()
    if not binary:
        logger.error(
            "No llama-server found. Run `manage llama install` to put one in "
            "documents-dev, or set LLAMA_SERVER_BIN."
        )
        return 2
    try:
        engine = engine_defaults()
    except RuntimeError as e:
        logger.error("%s", e)
        return 2
    cmd = engine_cmd(binary, url, engine)
    if "--print-cmd" in argv:
        print(" ".join(shlex.quote(part) for part in cmd))
        return 0
    if engine.get("models_dir"):
        if not os.path.isdir(engine["models_dir"]):
            logger.error("Model directory not found: %s", engine["models_dir"])
            return 2
    elif not os.path.isfile(engine["model_path"]):
        logger.error(
            "Model file not found: %s. Download it with `python setup_models.py`.",
            engine["model_path"],
        )
        return 2
    if is_alive(url):
        logger.info("An engine already answers at %s — nothing to start.", url)
        return 0

    logger.info("Starting engine: %s", " ".join(cmd))
    # exec, not spawn: the service manager's PID must be the server's, or
    # stopping the service would leave the engine running.
    os.execve(binary, cmd, _engine_environment())


if __name__ == "__main__":
    raise SystemExit(main())
