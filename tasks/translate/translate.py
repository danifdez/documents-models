from typing import List, Dict, Any, Optional, Tuple
from threading import Lock
from transformers import pipeline as hf_pipeline
from common.execution_registry import execution_handler
from utils.device import get_device
from services.text import chunk_units

_translation_pipelines: Dict[str, Any] = {}
_translation_pipeline_lock = Lock()
_translation_inference_lock = Lock()


def _get_translation_pipeline(source: str, target: str):
    key = f"{source}-{target}"
    if key not in _translation_pipelines:
        with _translation_pipeline_lock:
            if key not in _translation_pipelines:
                from lib.llm.config import get_task_config

                prefix = get_task_config("translate").get(
                    "model_prefix", "Helsinki-NLP/opus-mt"
                )
                model_name = f"{prefix}-{source}-{target}"
                device = get_device()
                _translation_pipelines[key] = hf_pipeline(
                    "translation", model=model_name, device=device
                )
    return _translation_pipelines[key]


def _normalize_text_items(texts: List[Any]) -> List[str]:
    """Return canonical text strings from string or {text, path?} items."""
    normalized = []
    for item in texts:
        if isinstance(item, dict):
            if not isinstance(item.get("text"), str):
                raise ValueError(
                    "Translation text items must contain a string text field"
                )
            normalized.append(item["text"])
        elif isinstance(item, str):
            normalized.append(item)
        else:
            raise ValueError("Translation text items must be strings or objects")
    return normalized


def _split_long_item(item: str, max_words: int) -> List[str]:
    """Split a single item into translatable sub-pieces if it exceeds max_words.
    Uses chunk_units (which respects paragraph/sentence boundaries via _recursive_split)."""
    if not item:
        return [item]
    if len(item.split()) <= max_words:
        return [item]
    return chunk_units([item], max_size=max_words) or [item]


@execution_handler("translate")
def translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate texts from source language to target language(s).

    Expected payload:
    - sourceLanguage: 'en' (optional, defaults to 'en')
    - targetLanguage: 'es' (single target) OR
    - targetLanguages: ['es', ...] (list of targets)
    - texts: list of strings or list of {text, path?} objects

    Returns: { response: [ { translation_text, original_text, path? }, ... ] }
    Invalid assignments raise so the protocol records a failed step.
    """
    if not isinstance(payload, dict):
        raise ValueError("Translation payload must be an object")

    source = payload.get('sourceLanguage') or 'en'

    target = None
    if 'targetLanguage' in payload and payload.get('targetLanguage'):
        target = payload.get('targetLanguage')
    elif 'targetLanguages' in payload and isinstance(payload.get('targetLanguages'), list) and payload.get('targetLanguages'):
        target = payload.get('targetLanguages')[0]
    else:
        target = 'es'

    texts = payload.get('texts') or []
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("Translation texts must be a non-empty list")

    translation = _get_translation_pipeline(source, target)

    from lib.llm.config import get_task_config
    task_config = get_task_config("translate")
    batch_size = task_config.get("chunk_size", 32)
    max_words_per_piece = task_config.get("max_words_per_item", 400)

    normalized_texts = _normalize_text_items(texts)

    # Flatten items into translation pieces, tracking origin item index.
    flat_pieces: List[Tuple[int, str]] = []
    for idx, item in enumerate(normalized_texts):
        for piece in _split_long_item(item, max_words_per_piece):
            flat_pieces.append((idx, piece))

    # Batch-translate the flat list, separating empty pieces (the pipeline can mishandle empty input).
    piece_translations: List[str] = [""] * len(flat_pieces)
    non_empty = [(i, p) for i, (_, p) in enumerate(flat_pieces) if p]

    for start in range(0, len(non_empty), batch_size):
        batch = non_empty[start:start + batch_size]
        batch_texts = [p for _, p in batch]
        with _translation_inference_lock:
            output = translation(batch_texts)
        for j, item in enumerate(output):
            tx = item.get('translation_text') if isinstance(item, dict) else str(item)
            piece_translations[batch[j][0]] = tx or ""

    # Reassemble per original item.
    per_item: Dict[int, List[str]] = {}
    for (item_idx, _), tx in zip(flat_pieces, piece_translations):
        per_item.setdefault(item_idx, []).append(tx)

    translated_texts: List[Dict[str, Optional[str]]] = []
    for idx in range(len(normalized_texts)):
        joined = " ".join(p for p in per_item.get(idx, []) if p)
        raw_item = texts[idx]
        original_text = raw_item["text"] if isinstance(raw_item, dict) else raw_item
        path = raw_item.get('path') if isinstance(raw_item, dict) else None
        translated_texts.append({
            "translation_text": joined,
            "original_text": original_text,
            "path": path,
        })

    return {"response": translated_texts}
