import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dateparser
import spacy

from utils.job_registry import job_handler
from utils.device import HAS_CUDA, get_spacy_model

logger = logging.getLogger(__name__)

_nlp = None

_RANGE_SEPARATORS_RE = re.compile(
    r"\s+(?:-|–|—|to|al|hasta|a|y|and|e|until|till)\s+|\s*-\s*|\s*–\s*|\s*—\s*",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
_MONTH_NAME_RE = re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"ene(ro)?|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
    r"janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre)\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b")


def _get_nlp():
    global _nlp
    if _nlp is None:
        if HAS_CUDA:
            spacy.prefer_gpu()
        from services.model_config import get_task_config
        task_config = get_task_config("date-extraction")
        model_name = task_config.get("model") or get_spacy_model()
        _nlp = spacy.load(model_name)
        logger.info("spaCy loaded model for date-extraction: %s", model_name)
    return _nlp


def _is_absolute(expression: str) -> bool:
    """True if the expression can be resolved without an anchor date."""
    if _YEAR_RE.search(expression):
        return True
    if _NUMERIC_DATE_RE.search(expression):
        return True
    return False


def _infer_precision(expression: str, parsed: datetime) -> str:
    """Infer day/month/year precision from the presence of day/month/year tokens."""
    has_day_number = bool(re.search(r"\b(0?[1-9]|[12]\d|3[01])\b", expression))
    has_month = bool(_MONTH_NAME_RE.search(expression)) or bool(_NUMERIC_DATE_RE.search(expression))
    has_year = bool(_YEAR_RE.search(expression))

    if _NUMERIC_DATE_RE.search(expression):
        return "day"
    if has_day_number and has_month:
        return "day"
    if has_month and has_year and not has_day_number:
        return "month"
    if has_year and not has_month:
        return "year"
    return "day"


def _parse_anchor(anchor_date: Optional[str]) -> Optional[datetime]:
    if not anchor_date:
        return None
    try:
        return datetime.strptime(anchor_date[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.warning("Invalid anchorDate: %s", anchor_date)
        return None


def _parse_with_dateparser(
    expression: str,
    language: str,
    anchor_dt: Optional[datetime],
    is_relative: bool,
) -> Optional[datetime]:
    settings: Dict[str, Any] = {"PREFER_DATES_FROM": "past"}
    if is_relative and anchor_dt is not None:
        settings["RELATIVE_BASE"] = anchor_dt

    languages = [language] if language else None
    try:
        return dateparser.parse(expression, languages=languages, settings=settings)
    except Exception:
        logger.debug("dateparser failed for expression: %s", expression, exc_info=True)
        return None


def _try_parse_range(
    expression: str,
    language: str,
    anchor_dt: Optional[datetime],
    is_relative: bool,
) -> Optional[Tuple[datetime, datetime]]:
    parts = _RANGE_SEPARATORS_RE.split(expression, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right:
        return None

    start = _parse_with_dateparser(left, language, anchor_dt, is_relative)
    end = _parse_with_dateparser(right, language, anchor_dt, is_relative)

    if start is None and end is not None:
        start = _parse_with_dateparser(f"{left} {right}", language, anchor_dt, is_relative)
    if end is None and start is not None:
        end = _parse_with_dateparser(f"{right} {left}", language, anchor_dt, is_relative)

    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def _build_context_snippet(text: str, start: int, end: int, window: int = 60) -> str:
    snippet_start = max(0, start - window)
    snippet_end = min(len(text), end + window)
    snippet = text[snippet_start:snippet_end]
    return snippet.strip()


def _llm_fallback(
    expression: str,
    context: str,
    language: str,
    anchor_date: Optional[str],
) -> Optional[Dict[str, Any]]:
    from services.model_config import get_task_config, get_llm_params
    from services.llm_service import get_llm_service

    task_config = get_task_config("date-extraction")
    if not task_config.get("enable_llm_fallback", True):
        return None

    try:
        params = get_llm_params("date-extraction")
        if not params.get("model_path"):
            return None
        llm = get_llm_service(**params)
    except Exception:
        logger.exception("LLM fallback unavailable for date-extraction")
        return None

    anchor_str = anchor_date or "UNKNOWN"
    prompt = (
        f'Anchor date: {anchor_str}. Text language: {language or "unknown"}.\n'
        f'Expression: "{expression}"\n'
        f'Context: "{context}"\n'
        'Respond ONLY with JSON. If the expression refers to a resolvable date or range, respond with '
        '{"date":"YYYY-MM-DD","endDate":"YYYY-MM-DD" or null,"precision":"day" or "month" or "year"}. '
        'If it is a relative expression (like "yesterday") and the anchor is UNKNOWN, respond with '
        '{"unresolved":true,"reason":"missing_anchor"}. '
        'If it cannot be resolved, respond with {"unresolved":true,"reason":"ambiguous"}. '
        'Do not include any other text.'
    )
    try:
        response = llm.chat([{"role": "user", "content": prompt}], max_tokens=120)
    except Exception:
        logger.exception("LLM fallback chat failed")
        return None

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if parsed.get("unresolved"):
        return {"unresolved": True, "reason": parsed.get("reason", "ambiguous")}

    date_str = parsed.get("date")
    end_str = parsed.get("endDate")
    precision = parsed.get("precision")
    if not date_str or precision not in {"day", "month", "year"}:
        return None
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        if end_str:
            datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        return None

    return {
        "date": date_str,
        "endDate": end_str,
        "precision": precision,
    }


@job_handler("date-extraction")
def extract_dates(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract explicit and implicit date expressions from a resource text.

    Payload:
        text: str — the document content (plain or HTML-ish text).
        language: str — ISO code (e.g. "en", "es"). Optional.
        anchorDate: str | null — YYYY-MM-DD, the date the document was authored.
            MUST be publicationDate — never uploadDate or the current date, or
            relative expressions will resolve against the wrong reference.

    Returns:
        {"dates": [ { rawExpression, date, endDate, precision, charOffset,
                      contextSnippet, resolver, isRelative, unresolvedReason } ]}
    """
    text = payload.get("text") or ""
    language = (payload.get("language") or "").strip() or None
    anchor_date = payload.get("anchorDate")
    anchor_dt = _parse_anchor(anchor_date)

    if not text:
        return {"dates": []}

    from services.model_config import get_task_config
    task_config = get_task_config("date-extraction")
    max_llm_fallbacks = int(task_config.get("max_llm_fallbacks", 10))

    nlp = _get_nlp()
    doc = nlp(text)

    results: List[Dict[str, Any]] = []
    seen_spans = set()
    llm_fallback_count = 0

    for ent in doc.ents:
        if ent.label_ != "DATE":
            continue
        raw = ent.text.strip()
        if len(raw) < 2:
            continue
        span_key = (ent.start_char, ent.end_char, raw.lower())
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        is_relative = not _is_absolute(raw)
        snippet = _build_context_snippet(text, ent.start_char, ent.end_char)

        entry: Dict[str, Any] = {
            "rawExpression": raw,
            "date": None,
            "endDate": None,
            "precision": None,
            "charOffset": ent.start_char,
            "contextSnippet": snippet,
            "resolver": "unresolved",
            "isRelative": is_relative,
            "unresolvedReason": None,
        }

        if is_relative and anchor_dt is None:
            entry["unresolvedReason"] = "missing_anchor"
            results.append(entry)
            continue

        range_result = _try_parse_range(raw, language or "", anchor_dt, is_relative)
        if range_result is not None:
            start_dt, end_dt = range_result
            entry.update({
                "date": start_dt.date().isoformat(),
                "endDate": end_dt.date().isoformat(),
                "precision": _infer_precision(raw, start_dt),
                "resolver": "dateparser",
            })
            results.append(entry)
            continue

        parsed = _parse_with_dateparser(raw, language or "", anchor_dt, is_relative)
        if parsed is not None:
            entry.update({
                "date": parsed.date().isoformat(),
                "precision": _infer_precision(raw, parsed),
                "resolver": "dateparser",
            })
            results.append(entry)
            continue

        if llm_fallback_count < max_llm_fallbacks:
            llm_fallback_count += 1
            llm_result = _llm_fallback(raw, snippet, language or "", anchor_date)
            if llm_result and not llm_result.get("unresolved"):
                entry.update({
                    "date": llm_result["date"],
                    "endDate": llm_result.get("endDate"),
                    "precision": llm_result["precision"],
                    "resolver": "llm",
                })
                results.append(entry)
                continue
            if llm_result and llm_result.get("unresolved"):
                entry["unresolvedReason"] = llm_result.get("reason") or "ambiguous"

        if entry["unresolvedReason"] is None:
            entry["unresolvedReason"] = "unparseable"
        results.append(entry)

    results.sort(key=lambda r: (r["date"] or "9999-12-31", r["charOffset"] or 0))
    return {"dates": results}
