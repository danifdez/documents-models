from threading import Lock
from typing import Any, Dict, List

from transformers import pipeline as hf_pipeline

from common.execution_registry import execution_handler
from utils.device import get_device


_translation_pipelines: Dict[str, Any] = {}
_translation_pipeline_lock = Lock()
_translation_inference_lock = Lock()


def _get_translation_pipeline(source: str, target: str):
    key = f"{source}-{target}"
    if key not in _translation_pipelines:
        with _translation_pipeline_lock:
            if key not in _translation_pipelines:
                from lib.llm.config import get_task_config

                prefix = get_task_config("translate-map").get(
                    "model_prefix", "Helsinki-NLP/opus-mt"
                )
                model_name = f"{prefix}-{source}-{target}"
                _translation_pipelines[key] = hf_pipeline(
                    "translation", model=model_name, device=get_device()
                )
    return _translation_pipelines[key]


@execution_handler("translate-map")
def translate_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Translation map payload must be an object")
    source = payload.get("sourceLanguage") or "en"
    target = payload.get("targetLanguage")
    units = payload.get("units")
    if not isinstance(target, str) or not target:
        raise ValueError("Translation map targetLanguage is required")
    if not isinstance(source, str) or not source:
        raise ValueError("Translation map sourceLanguage is invalid")
    if not isinstance(units, list) or not units or len(units) > 32:
        raise ValueError("Translation map units must contain between 1 and 32 items")

    normalized = [_validate_unit(unit) for unit in units]
    non_empty = [
        (index, unit["text"])
        for index, unit in enumerate(normalized)
        if unit["text"]
    ]
    translated = [""] * len(normalized)
    if non_empty:
        translation = _get_translation_pipeline(source, target)
        with _translation_inference_lock:
            output = translation([text for _, text in non_empty])
        if not isinstance(output, list) or len(output) != len(non_empty):
            raise ValueError("Translation pipeline returned an invalid batch")
        for output_index, item in enumerate(output):
            value = (
                item.get("translation_text")
                if isinstance(item, dict)
                else str(item)
            )
            translated[non_empty[output_index][0]] = value or ""

    return {
        "translations": [
            {
                "targetLanguage": target,
                "itemIndex": unit["itemIndex"],
                "pieceIndex": unit["pieceIndex"],
                "translationText": translated[index],
                "originalText": unit["originalText"],
                **({"path": unit["path"]} if "path" in unit else {}),
            }
            for index, unit in enumerate(normalized)
        ]
    }


@execution_handler("translate-reduce")
def translate_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Translation reduce payload must be an object")
    partials = payload.get("partials")
    if not isinstance(partials, list):
        raise ValueError("Translation reduce partials must be an array")

    translations: List[Dict[str, Any]] = []
    for partial in partials:
        if not isinstance(partial, list):
            raise ValueError("Translation reduce partial must be an array")
        translations.extend(_validate_translation(item) for item in partial)

    if payload.get("final") is not True:
        return {"translations": translations}

    response_mode = payload.get("responseMode")
    item_count = payload.get("itemCount")
    target_languages = payload.get("targetLanguages")
    if response_mode not in ("items", "targets"):
        raise ValueError("Translation reduce responseMode is invalid")
    if not isinstance(item_count, int) or item_count < 1:
        raise ValueError("Translation reduce itemCount is invalid")
    if not isinstance(target_languages, list) or not target_languages:
        raise ValueError("Translation reduce targetLanguages is invalid")
    if any(
        not isinstance(language, str) or not language
        for language in target_languages
    ):
        raise ValueError("Translation reduce target language is invalid")
    if len(set(target_languages)) != len(target_languages):
        raise ValueError("Translation reduce target languages must be unique")

    if response_mode == "items":
        if len(target_languages) != 1:
            raise ValueError("Item translation requires one target language")
        response = [
            _assemble_translation(translations, target_languages[0], item_index)
            for item_index in range(item_count)
        ]
    else:
        if item_count != 1:
            raise ValueError("Target translation requires one text item")
        response = [
            _assemble_translation(translations, target_language, 0)
            for target_language in target_languages
        ]

    return {"translations": translations, "response": response}


def _validate_unit(unit: Any) -> Dict[str, Any]:
    if not isinstance(unit, dict):
        raise ValueError("Translation unit must be an object")
    if not isinstance(unit.get("text"), str):
        raise ValueError("Translation unit text must be a string")
    if len(unit["text"].split()) > 400:
        raise ValueError("Translation unit exceeds the word limit")
    for key in ("itemIndex", "pieceIndex"):
        if not isinstance(unit.get(key), int) or unit[key] < 0:
            raise ValueError(f"Translation unit {key} is invalid")
    if not isinstance(unit.get("originalText"), str):
        raise ValueError("Translation unit originalText must be a string")
    if "path" in unit and not isinstance(unit["path"], str):
        raise ValueError("Translation unit path must be a string")
    return unit


def _validate_translation(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Translation result item must be an object")
    for key in ("targetLanguage", "translationText", "originalText"):
        if not isinstance(item.get(key), str):
            raise ValueError(f"Translation result {key} must be a string")
    for key in ("itemIndex", "pieceIndex"):
        if not isinstance(item.get(key), int) or item[key] < 0:
            raise ValueError(f"Translation result {key} is invalid")
    if "path" in item and not isinstance(item["path"], str):
        raise ValueError("Translation result path must be a string")
    return item


def _assemble_translation(
    translations: List[Dict[str, Any]], target_language: str, item_index: int
) -> Dict[str, Any]:
    pieces = sorted(
        (
            item
            for item in translations
            if item["targetLanguage"] == target_language
            and item["itemIndex"] == item_index
        ),
        key=lambda item: item["pieceIndex"],
    )
    if not pieces:
        raise ValueError("Translation reduce is missing an expected item")
    expected_indexes = list(range(len(pieces)))
    if [piece["pieceIndex"] for piece in pieces] != expected_indexes:
        raise ValueError("Translation reduce has missing or duplicate pieces")
    result = {
        "translation_text": " ".join(
            piece["translationText"]
            for piece in pieces
            if piece["translationText"]
        ),
        "original_text": pieces[0]["originalText"],
    }
    if "path" in pieces[0]:
        result["path"] = pieces[0]["path"]
    return result
