from utils.job_registry import job_handler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


@job_handler("summarize")
def summarize_text(payload) -> dict:
    model_name = "facebook/mbart-large-50-one-to-many-mmt"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.src_lang = payload["sourceLanguage"] + "_XX"
    es_forced_bos_token_id = tokenizer.lang_code_to_id[payload["targetLanguage"] + "_XX"]

    plain_text = re.sub(r"<[^>]+>", "", payload["content"])

    inputs = tokenizer(
        plain_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

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