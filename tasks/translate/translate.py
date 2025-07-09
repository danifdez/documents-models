from transformers import pipeline
from utils.job_registry import job_handler



@job_handler("translate")
def translate(payload) -> dict:
    """Translate texts from source language to target language"""
    try:
        translation = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-{payload['sourceLanguage']}-{payload['targetLanguage']}")
        translated_texts = []
        texts = payload["texts"]
        chunk_size = 32
        for i in range(0, len(texts), chunk_size):
            chunk = [item['text'] if isinstance(item, dict) and 'text' in item else item for item in texts[i:i+chunk_size]]
            output = translation(chunk)
            for idx, item in enumerate(output):
                translated_texts.append({
                    "translation_text": item['translation_text'],
                    "original_text": chunk[idx] if idx < len(chunk) else "",
                    "path": texts[i + idx]['path'] if isinstance(texts[i + idx], dict) and 'path' in texts[i + idx] else None
                })
    except Exception as e:
        return f"Error during translation: {e}"

    return {"response": translated_texts}