from utils.job_registry import job_handler
try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - llama_cpp is optional in some environments
    Llama = None
from config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_BATCH,
)
import re
from typing import List
import html


def normalize_text(text: str) -> str:
    try:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return text


def split_and_clean(generated: str) -> List[str]:
    # Split by commas or newlines and clean bullets/numbering
    parts = re.split(r'[\n,]+', generated)
    cleaned = []
    for p in parts:
        it = re.sub(r'^\s*[-\d\.\)]+\s*', '', p).strip()
        if it:
            cleaned.append(it)
    return cleaned


def enforce_constraints(items: List[str], max_items: int = 10) -> List[str]:
    out = []
    seen = set()
    for t in items:
        # truncate to 3 words
        t2 = ' '.join(t.split()[:3]).strip()
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
    """Extract up to 10 keywords/topics (1-3 words each) from input content.

    Payload expected keys:
      - content: HTML or text to extract keywords from
      - targetLanguage or target_language: language code (optional)

    Returns: {"keywords": [list of strings]}
    """

    try:
        text = payload.get("content") or ""
        if not text or not str(text).strip():
            return {"keywords": []}

        text = normalize_text(str(text))

        target_lang = payload.get("targetLanguage") or payload.get(
            "target_language") or "auto"

        prompt = (
            "You are an assistant that extracts up to 10 topics (keywords or short phrases) from a text. "
            "Return only the most relevant topics, ordered from most to least important, and NEVER exceed 10 topics. "
            "If there are fewer than 10 relevant topics, return fewer. "
            "EACH TOPIC MUST BE BETWEEN 1 AND 3 WORDS. If a natural topic would be longer, truncate it to the FIRST 3 WORDS.\n\n"
            "OUTPUT FORMAT: RETURN A SINGLE LINE WITH TOPICS SEPARATED BY COMMAS AND NOTHING ELSE. "
            "RETURN TOPICS IN THE SAME LANGUAGE AS THE INPUT TEXT; DO NOT TRANSLATE. "
            f"The language hint is: {target_lang}.\n"
            "Text:\n" + text
        )

        generated = ""
        try:
            if Llama is not None:
                llm = Llama(
                    model_path=LLM_MODEL_PATH,
                    n_ctx=LLM_N_CTX,
                    n_threads=LLM_N_THREADS,
                    n_batch=LLM_N_BATCH,
                )

                # Use chat completion if available, otherwise call as completion
                try:
                    resp = llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                    )
                    # llama_cpp chat returns choices with message/content or text depending on version
                    if resp and "choices" in resp and resp["choices"]:
                        c = resp["choices"][0]
                        if "message" in c and "content" in c["message"]:
                            generated = c["message"]["content"].strip()
                        elif "text" in c:
                            generated = c["text"].strip()
                except Exception:
                    # fallback to simple completion call
                    resp = llm(prompt, max_tokens=500, echo=False)
                    generated = resp["choices"][0]["text"].strip()
            else:
                # llama_cpp not available in this environment
                generated = ""
        except Exception:
            # LLM not available or failed -> leave generated empty to use heuristics
            generated = ""

        candidates: List[str] = []
        if generated:
            candidates = split_and_clean(generated)

        # If LLM didn't produce usable output, derive keywords heuristically from text
        if not candidates:
            # simple heuristic: take nouns/phrases by splitting on punctuation and picking first few words of sentences
            sentences = [s.strip() for s in re.split(
                r'(?<=[.!?])\s+', text) if s.strip()]
            heur = []
            for s in sentences:
                # take first 3 words of the sentence as a candidate
                heur.append(' '.join(s.split()[:3]))
            candidates = heur

        keywords_list = enforce_constraints(candidates, max_items=10)

        return {"keywords": keywords_list}

    except Exception as e:
        return {"error": f"Error extracting keywords: {e}"}
