"""
Pre-download all ML models required by the Documents models service.

Usage:
    python setup_models.py

Called by the Electron app after installing the models service bundle.
Outputs progress lines in the format: PROGRESS:<component>:<percent>
"""

import json
import os
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative time/size weight per model, so the single "ai-models" bar climbs in
# proportion to how long each download really takes. The GGUF LLM dwarfs the rest.
STEP_WEIGHTS = {
    "embeddings": 0.15,
    "spacy": 0.05,
    "summarization": 1.0,
    "whisper": 0.5,
    "llm": 5.7,
}


def _model_dir():
    """Writable dir for downloaded GGUF models. In standalone the bundle dir is
    read-only, so MODELS_MODEL_DIR points at a writable location the worker also
    reads (via llm_defaults.model_dir)."""
    return os.environ.get("MODELS_MODEL_DIR") or os.path.join(SCRIPT_DIR, "models")


def load_tasks():
    tasks_path = os.path.join(SCRIPT_DIR, "config", "tasks.json")
    if not os.path.exists(tasks_path):
        # Fall back to default
        tasks_path = os.path.join(SCRIPT_DIR, "common", "tasks.default.json")
    with open(tasks_path) as f:
        return json.load(f)


def progress(component, percent):
    print(f"PROGRESS:{component}:{percent}", flush=True)


def setup():
    tasks = load_tasks()

    steps = []

    # Embedding model (always needed for RAG/search)
    embedding_task = tasks.get("embedding", {})
    if embedding_task.get("enabled", False):
        model = embedding_task.get("model", "BAAI/bge-small-en-v1.5")
        steps.append(("embeddings", model, download_embedding))

    # spaCy NER model
    entity_task = tasks.get("entity-extraction", {})
    if entity_task.get("enabled", False):
        model = entity_task.get("model", "en_core_web_sm")
        steps.append(("spacy", model, download_spacy))

    # Summarization model
    summarize_task = tasks.get("summarize", {})
    if summarize_task.get("enabled", False):
        model = summarize_task.get("model", "facebook/mbart-large-50-one-to-many-mmt")
        steps.append(("summarization", model, download_seq2seq))

    # Whisper (transcription)
    transcribe_task = tasks.get("transcribe", {})
    if transcribe_task.get("enabled", False):
        model = transcribe_task.get("model", "base")
        steps.append(("whisper", model, download_whisper))

    # LLM (GGUF) — download from HuggingFace
    llm_tasks = ["keywords", "key-point", "ask"]
    for task_name in llm_tasks:
        task = tasks.get(task_name, {})
        if task.get("enabled", False) and task.get("type") == "llm":
            model = task.get("model", "")
            if model and model.endswith(".gguf"):
                steps.append(("llm", model, download_gguf))
                break  # Only need to download once

    # LoRA adapters are placed manually; warn if configured but missing
    check_lora_files(tasks)

    # Map each step onto a slice of the global 0-100 bar, sized by its weight, so
    # the bar advances smoothly (and crawls through the long GGUF download)
    # instead of jumping in equal chunks.
    total_w = sum(STEP_WEIGHTS.get(name, 0.3) for name, _, _ in steps) or 1
    acc = 0.0
    for name, model, fn in steps:
        w = STEP_WEIGHTS.get(name, 0.3)
        start, span = acc / total_w * 100, w / total_w * 100

        def report(local, _s=start, _sp=span):
            progress("ai-models", int(_s + _sp * max(0, min(100, local)) / 100))

        report(0)
        print(f"Downloading {name}: {model}", flush=True)
        try:
            fn(model, report)
            print(f"OK {name}: {model}", flush=True)
        except Exception as e:
            print(f"WARNING: Failed to download {name} ({model}): {e}", file=sys.stderr, flush=True)
        acc += w

    progress("ai-models", 100)
    print("All models downloaded.", flush=True)


def download_embedding(model_name, report=None):
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(model_name)


def download_spacy(model_name, report=None):
    import spacy
    try:
        spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        spacy.load(model_name)


def download_seq2seq(model_name, report=None):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)


def download_whisper(model_size, report=None):
    from faster_whisper import WhisperModel
    WhisperModel(model_size, device="cpu", compute_type="int8")


def check_lora_files(tasks):
    """Warn if any LLM task references a lora_model/lora_path that is not present on disk."""
    model_dir = _model_dir()
    for task_name, task in tasks.items():
        if task.get("type") != "llm" or not task.get("enabled", False):
            continue
        lora_path = task.get("lora_path")
        lora_name = task.get("lora_model")
        if not lora_path and lora_name:
            lora_path = lora_name if os.path.isabs(lora_name) else os.path.join(model_dir, lora_name)
        if lora_path and not os.path.exists(lora_path):
            print(
                f"WARNING: LoRA for task '{task_name}' not found at {lora_path}. "
                f"Place the adapter .gguf manually before running this task.",
                file=sys.stderr,
                flush=True,
            )


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def download_gguf(model_filename, report=None):
    """Download GGUF model from HuggingFace Hub.

    This is by far the longest step (~5.7 GB), so a background thread watches the
    download dir growing against the known file size and reports sub-progress —
    otherwise the bar would sit frozen here for minutes.
    """
    repo_id = "Qwen/Qwen3-8B-GGUF"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface_hub not available, skipping GGUF download", file=sys.stderr, flush=True)
        return

    model_dir = _model_dir()
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, model_filename)
    if os.path.exists(dest):
        print(f"  LLM already exists: {dest}", flush=True)
        if report:
            report(100)
        return

    total_size = None
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        total_size = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=model_filename)).size
    except Exception:
        pass  # progress will stay coarse, but the download still runs

    base = _dir_size(model_dir)
    stop = threading.Event()

    def _monitor():
        while not stop.wait(1.0):
            if report and total_size:
                downloaded = max(0, _dir_size(model_dir) - base)
                report(min(99, int(downloaded / total_size * 100)))

    monitor = threading.Thread(target=_monitor, daemon=True)
    if report and total_size:
        monitor.start()
    try:
        hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=model_dir)
    finally:
        stop.set()
        monitor.join(timeout=2)
    if report:
        report(100)


if __name__ == "__main__":
    setup()
