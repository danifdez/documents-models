import json
import logging
import re
from typing import Any, Dict, List, Optional

from lib.llm.config import get_llm_params, get_task_config
from lib.llm.prompts import get_prompt
from lib.llm.text import strip_dense_blobs, truncate_for_llm
from services.llm_service import get_llm_service
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)

_PROMPT = get_prompt("entity-extraction")
_ALLOWED_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "FAC", "PRODUCT",
    "WORK_OF_ART", "LANGUAGE", "LAW",
}


def _extract_entities(text: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    safe_text = truncate_for_llm(strip_dense_blobs(text), config)
    if not safe_text.strip() or not _PROMPT:
        return []
    try:
        response = get_llm_service(**get_llm_params("entity-extraction")).chat(
            [{"role": "user", "content": _PROMPT.format(text=safe_text)}],
            max_tokens=int(config.get("max_tokens", 2000)), temperature=0.0,
        )
        parsed = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip()))
    except Exception:
        logger.exception("entity-extraction chat failed")
        return []

    ignored = set(config.get("ignored_entity_types", []))
    result, seen = [], set()
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word") or "").strip()
        entity = str(item.get("entity") or "").strip().upper()
        key = (word, entity)
        if len(word) > 1 and entity in _ALLOWED_LABELS and entity not in ignored and key not in seen:
            seen.add(key)
            result.append({"word": word, "entity": entity})
    return result


@job_handler("entity-extraction")
def entities(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """
    Extract named entities from text using the local LLM.

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
        texts = payload.get("texts") or ([payload["text"]] if payload.get("text") else [])
        if not texts:
            return {"entities": []}

        text_strings: List[str] = []
        for item in texts:
            if isinstance(item, dict) and "text" in item:
                text_strings.append(strip_dense_blobs(str(item["text"])))
            else:
                text_strings.append(strip_dense_blobs(str(item)))

        return {"entities": _extract_entities("\n\n".join(text_strings), get_task_config("entity-extraction"))}

    except Exception as e:
        logger.exception("entity-extraction failed")
        return {"error": f"entity-extraction failed: {e}"}
