from typing import List, Dict, Any, Optional
from transformers import pipeline
from utils.job_registry import job_handler


def _normalize_text_items(texts: List[Any]) -> List[str]:
    """Return a list of strings extracted from texts, which may be str or dict with 'text' key."""
    normalized = []
    for item in texts:
        if isinstance(item, dict):
            if 'text' in item and isinstance(item['text'], str):
                normalized.append(item['text'])
            else:
                # fallback: stringify the whole dict
                normalized.append(str(item))
        else:
            normalized.append(str(item))
    return normalized


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
    # Defensive defaults / validation
    if not isinstance(payload, dict):
        return {"error": "Invalid payload: expected a dict."}

    source = payload.get('sourceLanguage') or payload.get('source') or 'en'

    # Support both single 'targetLanguage' and 'targetLanguages' (list). We'll translate only to the first in the list
    target = None
    if 'targetLanguage' in payload and payload.get('targetLanguage'):
        target = payload.get('targetLanguage')
    elif 'targetLanguages' in payload and isinstance(payload.get('targetLanguages'), list) and payload.get('targetLanguages'):
        target = payload.get('targetLanguages')[0]
    elif 'target' in payload and payload.get('target'):
        target = payload.get('target')
    else:
        target = 'es'

    # Validate texts
    texts = payload.get('texts') or payload.get('textsToTranslate') or []
    if not isinstance(texts, list) or len(texts) == 0:
        return {"error": "No texts provided to translate (expected non-empty list in payload['texts'])."}

    translated_texts: List[Dict[str, Optional[str]]] = []

    try:
        model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
        translation = pipeline("translation", model=model_name)

        chunk_size = 32
        normalized_texts = _normalize_text_items(texts)

        for i in range(0, len(normalized_texts), chunk_size):
            chunk = normalized_texts[i:i+chunk_size]
            output = translation(chunk)
            for idx, item in enumerate(output):
                orig_index = i + idx
                original_text = None
                path = None
                # try to extract original and path from original payload item when possible
                try:
                    raw_item = texts[orig_index]
                    if isinstance(raw_item, dict):
                        original_text = raw_item.get('text') or raw_item.get(
                            'original_text') or normalized_texts[orig_index]
                        path = raw_item.get('path')
                    else:
                        original_text = str(raw_item)
                except Exception:
                    original_text = chunk[idx] if idx < len(chunk) else None

                translated_texts.append({
                    "translation_text": item.get('translation_text') if isinstance(item, dict) else str(item),
                    "original_text": original_text,
                    "path": path,
                })

    except Exception as e:
        # Return structured error to make backend logging clearer
        return {"error": f"Error during translation: {type(e).__name__}: {e}"}

    return {"response": translated_texts}
