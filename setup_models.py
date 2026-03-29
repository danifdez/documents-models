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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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

    total = len(steps)
    for i, (name, model, fn) in enumerate(steps):
        pct = int((i / total) * 100)
        progress(name, pct)
        print(f"Downloading {name}: {model}", flush=True)
        try:
            fn(model)
            print(f"OK {name}: {model}", flush=True)
        except Exception as e:
            print(f"WARNING: Failed to download {name} ({model}): {e}", file=sys.stderr, flush=True)

    progress("done", 100)
    print("All models downloaded.", flush=True)


def download_embedding(model_name):
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(model_name)


def download_spacy(model_name):
    import spacy
    try:
        spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        spacy.load(model_name)


def download_seq2seq(model_name):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)


def download_whisper(model_size):
    from faster_whisper import WhisperModel
    WhisperModel(model_size, device="cpu", compute_type="int8")


def download_gguf(model_filename):
    """Download GGUF model from HuggingFace Hub."""
    # Default repo for Phi-4-mini-instruct GGUF
    repo_id = "microsoft/Phi-4-mini-instruct-gguf"

    try:
        from huggingface_hub import hf_hub_download
        model_dir = os.path.join(SCRIPT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        dest = os.path.join(model_dir, model_filename)
        if os.path.exists(dest):
            print(f"  LLM already exists: {dest}", flush=True)
            return
        hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            local_dir=model_dir,
        )
    except ImportError:
        print("  huggingface_hub not available, skipping GGUF download", file=sys.stderr, flush=True)


if __name__ == "__main__":
    setup()
