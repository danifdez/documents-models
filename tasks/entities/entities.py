import logging
from typing import Any, Dict, List

import spacy

from services.text import strip_dense_blobs
from utils.device import HAS_CUDA, get_spacy_model
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model to avoid crash at import time if not installed
nlp = None


def _get_nlp():
    global nlp
    if nlp is None:
        if HAS_CUDA:
            spacy.prefer_gpu()
        from lib.llm.config import get_task_config
        task_config = get_task_config("entity-extraction")
        model_name = task_config.get("model") or get_spacy_model()
        nlp = spacy.load(model_name)
        logger.info("spaCy loaded model: %s", model_name)
    return nlp


@job_handler("entity-extraction")
def entities(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract named entities from text using spaCy.

    Each text is sanitized with `strip_dense_blobs` (data URIs and >=2k-char
    unbroken tokens are replaced with placeholders) so an inline base64 image
    can't blow spaCy's memory budget. Returns `{entities: [...]}` on success
    or `{error: ...}` on failure (the backend skips persisting in the latter
    case).

    Parameters:
        payload['texts']: list of strings or {text: string} dicts.

    Returns:
        {"entities": [{"word": str, "entity": str}, ...]} or {"error": str}.
    """
    try:
        texts = payload.get("texts") or []
        if not texts:
            return {"entities": []}

        text_strings: List[str] = []
        for item in texts:
            if isinstance(item, dict) and "text" in item:
                text_strings.append(strip_dense_blobs(str(item["text"])))
            else:
                text_strings.append(strip_dense_blobs(str(item)))

        from lib.llm.config import get_task_config
        task_config = get_task_config("entity-extraction")

        ignored_entity_types = set(task_config.get("ignored_entity_types", [
            'CARDINAL', 'DATE', 'MONEY', 'ORDINAL', 'PERCENT', 'QUANTITY', 'TIME'
        ]))

        batch_size = int(task_config.get("batch_size", 32))

        nlp = _get_nlp()
        # Defensive: bump max_length if any single text approaches the cap.
        longest = max((len(t) for t in text_strings), default=0)
        needed = longest + 100_000
        if getattr(nlp, "max_length", 0) < needed:
            nlp.max_length = needed

        docs = nlp.pipe(text_strings, batch_size=batch_size)

        parse_result: List[Dict[str, str]] = []
        for doc in docs:
            for ent in doc.ents:
                if (len(ent.text.strip()) > 1
                        and ent.label_ not in ignored_entity_types):
                    parse_result.append({
                        "word": ent.text.strip(),
                        "entity": ent.label_,
                    })

        # Remove duplicates while preserving order.
        unique_result: List[Dict[str, str]] = []
        seen = set()
        for ent in parse_result:
            key = (ent["word"], ent["entity"])
            if key in seen:
                continue
            seen.add(key)
            unique_result.append(ent)

        return {"entities": unique_result}

    except Exception as e:
        logger.exception("entity-extraction failed")
        return {"error": f"entity-extraction failed: {e}"}
