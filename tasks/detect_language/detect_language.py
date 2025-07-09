from langdetect import detect
from utils.job_registry import job_handler

@job_handler("detect-language")
def detect_language(payload) -> dict:
    """Detect language of given text and return the language code."""
    try:
        results = []
        for sample in payload.get("samples", []):
            lang = detect(sample)
            results.append({"text": sample, "language": lang})
        return {"results": results}
    except Exception as e:
        return {"error": f"Error detecting language: {e}"}

