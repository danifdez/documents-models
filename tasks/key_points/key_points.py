from utils.job_registry import job_handler
from llama_cpp import Llama
from config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_BATCH,
)
import re
from typing import List
import html


def clean_sentence(s: str) -> str:
    # Remove bullet markers and excess whitespace
    s = s.strip()
    s = re.sub(r'^\d+\.|^-|^\*', '', s).strip()
    return s


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


@job_handler("key-point")
def key_points(payload) -> dict:
    """Generate up to 5 key points using a seq2seq transformer (like the notebook).

    The model is prompted to extract key points as full sentences. After generation
    we split and filter results to ensure each sentence has between 3 and 10 words
    and return up to 5 items.
    """

    try:
        text = payload["content"] or ""
        if not text or not text.strip():
            return {"key_points": []}

        # Remove HTML tags and unescape HTML entities to get plain text
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = html.unescape(text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception:
            # If cleaning fails, keep original text
            pass

        # Read target language from payload (language code, e.g. 'en', 'es')
        target_lang = payload["targetLanguage"] or payload["target_language"] or "en"

        # Construct a concise prompt for the instruct model
        prompt = (
            f"You are an assistant that extracts up to 5 concise key points from a text. "
            f"Each point should be a complete sentence between 3 and 10 words. "
            f"Return them as separate lines in the language specified: {target_lang}.\n\nText: "
            + text
        )

        # Try using the local LLM via llama_cpp; if it fails, raise and fallback to heuristics below
        try:
            llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_N_CTX,
                n_threads=LLM_N_THREADS,
                n_batch=LLM_N_BATCH,
            )

            # Limit tokens for key point generation
            response = llm(prompt, max_tokens=1000, echo=False)
            generated = response["choices"][0]["text"].strip()
        except Exception as e:
            # If the local LLM is not available or errors, fall back to a simple heuristic
            generated = ""

        # Split generated text into candidate lines/sentences
        candidates: List[str] = []
        if generated:
            for line in generated.splitlines():
                line = clean_sentence(line)
                if line:
                    candidates.append(line)

            if not candidates:
                # fallback: split by sentence-ending punctuation
                candidates = [clean_sentence(s) for s in re.split(
                    r'(?<=[.!?])\s+', generated) if s.strip()]

        # Filter by required word count and deduplicate preserving order
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

        # If model output didn't satisfy constraints or model wasn't available, try a simple heuristic using the original text sentences
        if len(selected) < 5:
            # simple sentence split from original content
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
