from utils.job_registry import job_handler
from utils.device import get_device
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

_summarize_model = None
_summarize_tokenizer = None


def _get_summarize_model():
    global _summarize_model, _summarize_tokenizer
    if _summarize_model is None:
        from services.model_config import get_task_config
        model_name = get_task_config("summarize").get("model", "facebook/mbart-large-50-one-to-many-mmt")
        device = get_device()
        try:
            _summarize_tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            _summarize_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        _summarize_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return _summarize_model, _summarize_tokenizer


@job_handler("summarize")
def summarize_text(payload) -> dict:
    device = get_device()
    model, tokenizer = _get_summarize_model()
    tokenizer.src_lang = payload["sourceLanguage"] + "_XX"
    es_forced_bos_token_id = tokenizer.lang_code_to_id[payload["targetLanguage"] + "_XX"]

    plain_text = re.sub(r"<[^>]+>", "", payload["content"])

    inputs = tokenizer(
        plain_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        max_length=200,
        min_length=30,
        forced_bos_token_id=es_forced_bos_token_id,
        no_repeat_ngram_size=3
    )

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"response": summary_text.strip()}