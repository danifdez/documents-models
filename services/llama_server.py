"""The inference engine, which no longer lives inside this process.

Every worker used to load its own .gguf through llama-cpp-python, and the
embedded browser started a llama-server of its own: two copies of the same model
fighting over the RAM and VRAM of one machine. Now there is a single
llama-server, it belongs to documents-dev, and everything on the machine — every
job in this worker plus the embedded browser — generates through it over HTTP.

Who starts it:

  - `manage llama start` (the normal way): documents-dev runs the engine as one
    more of its services, from `python -m services.llama_server`, which execs the
    binary in the foreground so the service manager owns the process.
  - a job, as a last resort: if nothing answers when the first job needs the LLM,
    it starts the same engine with the same settings rather than failing. See
    `ensure_server`.

Whoever gets there first wins: if the server already answers /health we use it as
it is, whatever model it has loaded. That is the point of sharing one — one copy
of the weights, one queue, and the chat no longer competes with the indexer.

The engine is defined in exactly one place, `engine_defaults()`, so however it
came up it is the same server: same model, same context, same slots.
"""

import atexit
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

# Where the engine answers when nobody said otherwise. Unlike the backend — which
# refuses to guess, because announcing a URL makes clients kill their own engine —
# the worker needs a default: a job that can't find the engine can't run at all.
DEFAULT_URL = "http://127.0.0.1:18080"

_lock = threading.Lock()
_child: Optional[subprocess.Popen] = None
_ready_url: Optional[str] = None


def server_url() -> str:
    """Where the shared engine answers. Env wins over config, config over default."""
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
    given — that is how you point the engine at a build with CUDA, or at one
    shared with another app. Otherwise documents-dev's own copy is used, and only
    then whatever `llama-server` is on PATH.
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


def props(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """What the server is serving: model path, context size, sampling defaults.

    Empty dict when it can't be asked — callers use this to warn, never to
    decide whether to run.
    """
    try:
        with urllib.request.urlopen(f"{url}/props", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")) or {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError):
        return {}


def lora_adapters(url: str, timeout: float = 5.0) -> List[Dict[str, Any]]:
    """The adapters the server has loaded: `[{"id", "path", "scale"}, …]`.

    Adapters can only be attached when the server starts (`--lora`), so this is
    a read of what is already there. Empty list when it can't be asked, like
    `props`: callers use it to resolve an id or to inform, never to decide
    whether to run.
    """
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/lora-adapters",
                                    timeout=timeout) as resp:
            loaded = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
            json.JSONDecodeError):
        return []
    return loaded if isinstance(loaded, list) else []


def lora_adapter_id(url: str, path: str) -> Optional[int]:
    """The id the server gave an adapter, or None if it hasn't got it loaded.

    Matches on the absolute path and, failing that, on the filename: the trainer
    and the server may name the same adapter through different mounts, and the
    id is what a request has to cite.
    """
    if not path:
        return None
    wanted = os.path.basename(path)
    for adapter in lora_adapters(url):
        if not isinstance(adapter, dict):
            continue
        loaded = str(adapter.get("path") or "")
        if loaded == path or (wanted and os.path.basename(loaded) == wanted):
            return adapter.get("id")
    return None


def loaded_model(url: str) -> str:
    """Basename of the .gguf the server has loaded, or '' if unknown."""
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

    Used both by `manage llama start` and by the last-resort spawn from a job, so
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

    return {
        "model_path": params["model_path"],
        "n_ctx": _int("LLAMA_SERVER_CTX", params["n_ctx"]),
        "n_threads": _int("LLAMA_SERVER_THREADS", params["n_threads"]),
        "n_gpu_layers": _int("LLAMA_SERVER_GPU_LAYERS", params["n_gpu_layers"]),
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
        path = deployment["path"]
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
    # One slot per concurrent caller: the browser's chat and this worker's jobs
    # land on the same server and would otherwise queue behind each other. Each
    # slot keeps its own KV cache, which is what makes `cache_prompt` worth
    # anything.
    slots = max(1, int(os.environ.get("LLAMA_SERVER_SLOTS", "2")))
    cmd = [
        binary,
        "--host", host,
        "--port", str(port),
        "--model", engine["model_path"],
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
    """Return the URL of a live shared engine, starting one if we have to.

    `model_path` is what the calling task wanted, and it is only a fallback: the
    engine is documents-dev's, and `engine_defaults()` decides what it serves. A
    task that wanted another model still gets an answer — `LLMService` says so in
    the log.

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
    if not os.path.isfile(engine["model_path"]):
        # The installation's model is missing but the job's is there: serve that
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
    logger.info("Starting shared llama-server: %s (log: %s)", " ".join(cmd), log_path)
    log = open(log_path, "ab")
    global _child
    # Its own process group, so a Ctrl-C in the worker's terminal doesn't reach
    # the server before we get to shut it down in order.
    _child = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
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
    logger.info("Stopping shared llama-server (pid %s)", child.pid)
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
    if not os.path.isfile(engine["model_path"]):
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
    os.execv(binary, cmd)


if __name__ == "__main__":
    raise SystemExit(main())
