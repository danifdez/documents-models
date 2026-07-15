from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline as hf_pipeline
from utils.job_registry import job_handler
from utils.device import get_device
from services.text import chunk_units

_translation_pipelines: Dict[str, Any] = {}


def _get_translation_pipeline(source: str, target: str):
    key = f"{source}-{target}"
    if key not in _translation_pipelines:
        from lib.llm.config import get_task_config
        prefix = get_task_config("translate").get("model_prefix", "Helsinki-NLP/opus-mt")
        model_name = f"{prefix}-{source}-{target}"
        device = get_device()
        _translation_pipelines[key] = hf_pipeline("translation", model=model_name, device=device)
    return _translation_pipelines[key]


def _normalize_text_items(texts: List[Any]) -> List[str]:
    """Return a list of strings extracted from texts, which may be str or dict with 'text' key."""
    normalized = []
    for item in texts:
        if isinstance(item, dict):
            if 'text' in item and isinstance(item['text'], str):
                normalized.append(item['text'])
            else:
                normalized.append(str(item))
        else:
            normalized.append(str(item))
    return normalized


def _split_long_item(item: str, max_words: int) -> List[str]:
    """Split a single item into translatable sub-pieces if it exceeds max_words.
    Uses chunk_units (which respects paragraph/sentence boundaries via _recursive_split)."""
    if not item:
        return [item]
    if len(item.split()) <= max_words:
        return [item]
    return chunk_units([item], max_size=max_words) or [item]


@job_handler("translate")
def translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate texts from source language to target language(s).

    Expected payload shapes (backwards-compatible):
    - sourceLanguage: 'en' (optional, defaults to 'en')
    - targetLanguage: 'es' (single target) OR
    - targetLanguages: ['es', ...] (list of targets)
    - texts: list of strings or list of {text, path?} objects

    Returns: { response: [ { translation_text, original_text, path? }, ... ] }
    On error returns { error: "..." }
    """
    if not isinstance(payload, dict):
        return {"error": "Invalid payload: expected a dict."}

    source = payload.get('sourceLanguage') or payload.get('source') or 'en'

    target = None
    if 'targetLanguage' in payload and payload.get('targetLanguage'):
        target = payload.get('targetLanguage')
    elif 'targetLanguages' in payload and isinstance(payload.get('targetLanguages'), list) and payload.get('targetLanguages'):
        target = payload.get('targetLanguages')[0]
    elif 'target' in payload and payload.get('target'):
        target = payload.get('target')
    else:
        target = 'es'

    texts = payload.get('texts') or payload.get('textsToTranslate') or []
    if not isinstance(texts, list) or len(texts) == 0:
        return {"error": "No texts provided to translate (expected non-empty list in payload['texts'])."}

    try:
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
            if isinstance(raw_item, dict):
                original_text = raw_item.get('text') or raw_item.get('original_text') or normalized_texts[idx]
                path = raw_item.get('path')
            else:
                original_text = str(raw_item)
                path = None
            translated_texts.append({
                "translation_text": joined,
                "original_text": original_text,
                "path": path,
            })

    except Exception as e:
        return {"error": f"Error during translation: {type(e).__name__}: {e}"}

    return {"response": translated_texts}
