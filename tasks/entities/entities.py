"""Self-contained steps for the durable entity-extraction workflow."""

from threading import Lock
from typing import Any, Dict, List

from common.execution_registry import execution_handler
from lib.llm.config import get_task_config
from lib.llm.text import strip_dense_blobs

_ALLOWED_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "EVENT",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "LANGUAGE",
    "LAW",
}
_MODEL_LABELS = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
}
_entity_pipelines: Dict[tuple, Any] = {}
_entity_pipeline_lock = Lock()
_entity_inference_lock = Lock()


def _get_entity_pipeline(config: Dict[str, Any]):
    model_name = str(
        config.get("model") or "Davlan/xlm-roberta-base-ner-hrl"
    )
    revision = config.get("model_revision")
    device = str(config.get("device") or "cpu")
    key = (model_name, revision, device)
    if key not in _entity_pipelines:
        with _entity_pipeline_lock:
            if key not in _entity_pipelines:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline as hf_pipeline,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    fix_mistral_regex=True,
                )
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    revision=revision,
                )
                _entity_pipelines[key] = hf_pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    aggregation_strategy="simple",
                )
    return _entity_pipelines[key]


def _text_windows(content: str, tokenizer: Any, stride: int) -> List[tuple]:
    encoded = tokenizer(
        content,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, list) or not offsets:
        return [(0, content)]

    max_length = int(getattr(tokenizer, "model_max_length", 512))
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    window_tokens = max_length - special_tokens
    if window_tokens < 1 or stride < 0 or stride >= window_tokens:
        raise ValueError("entity-extraction-map stride is invalid")
    if len(offsets) <= window_tokens:
        return [(0, content)]

    windows: List[tuple] = []
    step = window_tokens - stride
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + window_tokens, len(offsets))
        char_start = int(offsets[token_start][0])
        char_end = int(offsets[token_end - 1][1])
        windows.append((char_start, content[char_start:char_end]))
        if token_end == len(offsets):
            break
    return windows


def _extract_entities(
    content: str,
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    safe_content = strip_dense_blobs(content).strip()
    if not safe_content:
        raise ValueError("entity-extraction-map requires non-empty content")
    max_input_words = int(config.get("max_input_words", 1500))
    if len(safe_content.split()) > max_input_words:
        raise ValueError("entity-extraction-map content exceeds its word budget")

    classifier = _get_entity_pipeline(config)
    parsed: List[Dict[str, Any]] = []
    with _entity_inference_lock:
        for window_start, window in _text_windows(
            safe_content,
            classifier.tokenizer,
            int(config.get("stride", 64)),
        ):
            window_result = classifier(window)
            if not isinstance(window_result, list):
                raise ValueError(
                    "entity-extraction-map classifier returned an invalid result"
                )
            for item in window_result:
                if not isinstance(item, dict):
                    continue
                adjusted = dict(item)
                if isinstance(adjusted.get("start"), int):
                    adjusted["start"] += window_start
                if isinstance(adjusted.get("end"), int):
                    adjusted["end"] += window_start
                parsed.append(adjusted)

    ignored = {
        str(entity_type).strip().upper()
        for entity_type in config.get("ignored_entity_types", [])
    }
    max_entities = int(config.get("max_entities", 200))
    result: List[Dict[str, str]] = []
    seen_spans = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        model_label = str(
            item.get("entity_group") or item.get("entity") or ""
        ).strip().upper()
        entity = _MODEL_LABELS.get(model_label)
        if not entity or entity in ignored:
            continue

        start = item.get("start")
        end = item.get("end")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(safe_content)
        ):
            word = safe_content[start:end].strip()
        else:
            word = str(item.get("word") or "").replace("▁", " ").strip()
        if not word:
            continue
        span_key = (start, end, entity)
        if isinstance(start, int) and isinstance(end, int):
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
        result.append({"word": word, "entity": entity})
        if len(result) >= max_entities:
            break
    return result


def _dedupe(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result: List[Dict[str, str]] = []
    for entry in entities:
        key = entry["word"].casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


@execution_handler("entity-extraction-map")
def entity_extraction_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("entity-extraction-map content must be a string")
    return {
        "entities": _extract_entities(
            content,
            get_task_config("entity-extraction-map"),
        )
    }


@execution_handler("entity-extraction-reduce")
def entity_extraction_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list) or not partials:
        raise ValueError("entity-extraction-reduce requires partials")

    entities: List[Dict[str, str]] = []
    for partial in partials:
        if not isinstance(partial, list):
            raise ValueError("entity-extraction-reduce partials must be arrays")
        for entry in partial:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("word"), str)
                or not entry["word"].strip()
                or entry.get("entity") not in _ALLOWED_LABELS
            ):
                raise ValueError("entity-extraction-reduce received an invalid entity")
            entities.append(
                {"word": entry["word"].strip(), "entity": entry["entity"]}
            )
    return {"entities": _dedupe(entities)}
