from utils.job_registry import job_handler
from services.llm_service import get_llm_service
from services.text import normalize_text
from services.prompts import get_prompt
from services.model_config import get_llm_params
import re
from typing import List


def clean_sentence(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^\d+\.|^-|^\*', '', s).strip()
    return s


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


@job_handler("key-point")
def key_points(payload) -> dict:
    try:
        text = payload.get("content") or ""
        if not text or not str(text).strip():
            return {"key_points": []}

        text = normalize_text(str(text))

        target_lang = payload.get("targetLanguage") or payload.get("target_language") or "en"

        prompt = get_prompt("key-point").format(target_lang=target_lang, text=text)

        try:
            params = get_llm_params("key-point")
            generated = get_llm_service(**params).generate(prompt, max_tokens=1000)
        except Exception:
            generated = ""

        candidates: List[str] = []
        if generated:
            for line in generated.splitlines():
                line = clean_sentence(line)
                if line:
                    candidates.append(line)

            if not candidates:
                candidates = [clean_sentence(s) for s in re.split(
                    r'(?<=[.!?])\s+', generated) if s.strip()]

        selected: List[str] = []
        seen = set()
        for s in candidates:
            wc = word_count(s)
            if 3 <= wc <= 10:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    selected.append(s)
            if len(selected) >= 5:
                break

        if len(selected) < 5:
            orig_sentences = [s.strip() for s in re.split(
                r'(?<=[.!?])\s+', text) if s.strip()]
            for s in orig_sentences:
                cs = clean_sentence(s)
                wc = word_count(cs)
                if 3 <= wc <= 10 and cs.lower() not in seen:
                    seen.add(cs.lower())
                    selected.append(cs)
                if len(selected) >= 5:
                    break

        return {"key_points": selected}
    except Exception as e:
        return {"error": f"Error extracting key points: {e}"}
