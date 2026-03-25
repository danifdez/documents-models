from utils.job_registry import job_handler
from services.llm_service import get_llm_service
from services.text import normalize_text
from services.prompts import get_prompt
from services.model_config import get_llm_params, get_task_config
import re
from typing import List


def split_and_clean(generated: str) -> List[str]:
    # Split by commas or newlines and clean bullets/numbering
    parts = re.split(r'[\n,]+', generated)
    cleaned = []
    for p in parts:
        it = re.sub(r'^\s*[-\d\.\)]+\s*', '', p).strip()
        if it:
            cleaned.append(it)
    return cleaned


def enforce_constraints(items: List[str], max_items: int = 10, max_words: int = 3) -> List[str]:
    out = []
    seen = set()
    for t in items:
        t2 = ' '.join(t.split()[:max_words]).strip()
        if not t2:
            continue
        key = t2.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t2)
        if len(out) >= max_items:
            break
    return out


@job_handler("keywords")
def keywords(payload) -> dict:
    try:
        text = payload.get("content") or ""
        if not text or not str(text).strip():
            return {"keywords": []}

        text = normalize_text(str(text))

        target_lang = payload.get("targetLanguage") or payload.get(
            "target_language") or "auto"

        prompt = get_prompt("keywords").format(target_lang=target_lang, text=text)

        task_config = get_task_config("keywords")

        generated = ""
        try:
            params = get_llm_params("keywords")
            llm_service = get_llm_service(**params)
            max_tokens = task_config.get("max_tokens", 500)
            try:
                generated = llm_service.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
            except Exception:
                generated = llm_service.generate(prompt, max_tokens=max_tokens)
        except Exception:
            generated = ""

        candidates: List[str] = []
        if generated:
            candidates = split_and_clean(generated)

        if not candidates:
            sentences = [s.strip() for s in re.split(
                r'(?<=[.!?])\s+', text) if s.strip()]
            heur = []
            for s in sentences:
                heur.append(' '.join(s.split()[:3]))
            candidates = heur

        keywords_list = enforce_constraints(
            candidates,
            max_items=task_config.get("max_items", 10),
            max_words=task_config.get("max_words_per_item", 3),
        )

        return {"keywords": keywords_list}

    except Exception as e:
        return {"error": f"Error extracting keywords: {e}"}
