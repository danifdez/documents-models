"""Self-contained steps for the durable date-extraction workflow."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dateparser

from common.execution_registry import execution_handler
from lib.llm.config import get_llm_params, get_task_config
from lib.llm.grammars import STRING_ARRAY_GBNF
from lib.llm.prompts import get_prompt
from services.llm_service import get_llm_service


_RANGE_SEPARATORS_RE = re.compile(
    r"\s+(?:-|–|—|to|al|hasta|a|y|and|e|until|till)\s+|"
    r"\s+[-–—]\s+",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
_MONTH_NAME_RE = re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
    r"aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"ene(ro)?|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
    r"octubre|noviembre|diciembre|janvier|f[eé]vrier|mars|avril|mai|juin|"
    r"juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre|januar|"
    r"februar|m[aä]rz|juni|juli|okt(ober)?|dez(ember)?|gennaio|febbraio|"
    r"aprile|maggio|giugno|luglio|settembre|ottobre|dicembre)\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|"
    r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"
)
_WEEKDAY_RE = re.compile(
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"\b(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)\b|"
    r"\b(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\b|"
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b|"
    r"\b(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|"
    r"domenica)\b",
    re.IGNORECASE,
)
_RELATIVE_MARKER_RE = re.compile(
    r"\bhace\b|\bdentro de\b|\bque viene\b|\bpr[oó]xim\w*\b|\bpasad\w*\b|"
    r"\bago\b|\bin\b|\bnext\b|\blast\b|\bwithin\b|\bvor\b|"
    r"\bn[aä]chst\w*\b|\bletzt\w*\b|\bil y a\b|\bdans\b|"
    r"\bprochain\w*\b|\bdernier\w*\b|\bfa\b|\btra\b|\bfra\b|"
    r"\bprossim\w*\b|\bscors\w*\b",
    re.IGNORECASE,
)
_STANDALONE_RELATIVE_RE = re.compile(
    r"\b(ayer|hoy|ma[nñ]ana|anteayer|anoche)\b|"
    r"\b(yesterday|today|tomorrow|tonight)\b|"
    r"\b(gestern|heute|morgen|vorgestern)\b|"
    r"\b(hier|aujourd'hui|demain|avant-hier)\b|"
    r"\b(ieri|oggi|domani|stanotte)\b",
    re.IGNORECASE,
)
_TIME_UNIT_RE = re.compile(
    r"\b(d[ií]as?|semanas?|meses|mes|a[nñ]os?|horas?|minutos?|d[eé]cadas?|"
    r"siglos?)\b|\b(days?|weeks?|months?|years?|hours?|minutes?|decades?|"
    r"centur(y|ies))\b|\b(tage?n?|wochen?|monate?n?|jahre?n?|stunden?|"
    r"minuten?|jahrzehnte?n?)\b|\b(jours?|semaines?|mois|ann[eé]es?|"
    r"heures?|minutes?|d[eé]cennies?|si[eè]cles?)\b|\b(giorni?|settimane?|"
    r"mesi|anni?|ore|ora|minuti?|decenni?|secoli?)\b",
    re.IGNORECASE,
)
_DECOR_RE = re.compile(r"^[\s_*`~@#]+|[\s_*`~@#]+$")
_DETECT_PROMPT = get_prompt("date-extraction-map", "detect_prompt.md")
_LANGUAGE_SKIP_TOKENS = {
    "it": ["il"],
}


def _parse_expression_array(response: str) -> List[str]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("date-extraction-map returned invalid JSON") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("date-extraction-map must return a string array")
    return parsed


def _detect_expressions(text: str, config: Dict[str, Any]) -> List[str]:
    if not _DETECT_PROMPT:
        raise RuntimeError("date-extraction-map prompt is unavailable")
    response = get_llm_service(**get_llm_params("date-extraction-map")).chat(
        [{"role": "user", "content": _DETECT_PROMPT.format(text=text)}],
        max_tokens=int(config.get("max_tokens", 800)),
        grammar=STRING_ARRAY_GBNF,
        temperature=0.0,
    )
    return [
        cleaned
        for item in _parse_expression_array(response)
        if len(cleaned := _DECOR_RE.sub("", item)) >= 2
    ]


def _locate(text: str, expression: str, cursor: int) -> int:
    lowered_text = text.casefold()
    lowered_expression = expression.casefold()
    for start in (cursor, 0):
        position = text.find(expression, start)
        if position >= 0:
            return position
        position = lowered_text.find(lowered_expression, start)
        if position >= 0:
            return position
    return -1


def _classify(expression: str) -> Optional[str]:
    if _YEAR_RE.search(expression) or _NUMERIC_DATE_RE.search(expression):
        return "absolute"
    if _MONTH_NAME_RE.search(expression) or _WEEKDAY_RE.search(expression):
        return "absolute"
    if _STANDALONE_RELATIVE_RE.search(expression):
        return "relative"
    if _RELATIVE_MARKER_RE.search(expression) and _TIME_UNIT_RE.search(expression):
        return "relative"
    return None


def _infer_precision(expression: str) -> str:
    has_day_number = bool(re.search(r"\b(0?[1-9]|[12]\d|3[01])\b", expression))
    has_month = bool(_MONTH_NAME_RE.search(expression)) or bool(
        _NUMERIC_DATE_RE.search(expression)
    )
    has_year = bool(_YEAR_RE.search(expression))
    if _NUMERIC_DATE_RE.search(expression) or (has_day_number and has_month):
        return "day"
    if has_month and has_year:
        return "month"
    if has_year:
        return "year"
    return "day"


def _parse_anchor(anchor_date: Any) -> Optional[datetime]:
    if anchor_date is None:
        return None
    if not isinstance(anchor_date, str):
        raise ValueError("date-extraction-map anchorDate must be an ISO date or null")
    try:
        return datetime.strptime(anchor_date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("date-extraction-map anchorDate must use YYYY-MM-DD") from error


def _parse_with_dateparser(
    expression: str,
    language: Optional[str],
    anchor: Optional[datetime],
    is_relative: bool,
) -> Optional[datetime]:
    settings: Dict[str, Any] = {"PREFER_DATES_FROM": "past"}
    if language in _LANGUAGE_SKIP_TOKENS:
        settings["SKIP_TOKENS"] = _LANGUAGE_SKIP_TOKENS[language]
    if is_relative and anchor is not None:
        settings["RELATIVE_BASE"] = anchor
    try:
        return dateparser.parse(
            expression,
            languages=[language] if language else None,
            settings=settings,
        )
    except (TypeError, ValueError):
        return None


def _try_parse_range(
    expression: str,
    language: Optional[str],
    anchor: Optional[datetime],
    is_relative: bool,
) -> Optional[Tuple[datetime, datetime]]:
    parts = _RANGE_SEPARATORS_RE.split(expression, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (part.strip() for part in parts)
    if not left or not right:
        return None
    start = _parse_with_dateparser(left, language, anchor, is_relative)
    end = _parse_with_dateparser(right, language, anchor, is_relative)
    if start is None and end is not None:
        start = _parse_with_dateparser(f"{left} {right}", language, anchor, is_relative)
    if end is None and start is not None:
        end = _parse_with_dateparser(f"{right} {left}", language, anchor, is_relative)
    if start is None or end is None:
        return None
    return (end, start) if end < start else (start, end)


def _context(text: str, start: int, end: int, window: int = 60) -> str:
    return text[max(0, start - window):min(len(text), end + window)].strip()


def _extract_dates(
    text: str,
    language: Optional[str],
    anchor: Optional[datetime],
    char_offset: int,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_spans = set()
    cursor = 0
    for raw in _detect_expressions(text, config):
        start = _locate(text, raw, cursor)
        if start < 0:
            continue
        end = start + len(raw)
        cursor = end
        span = (start, end, raw.casefold())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        kind = _classify(raw)
        if kind is None:
            continue

        relative = kind == "relative"
        entry: Dict[str, Any] = {
            "rawExpression": raw,
            "date": None,
            "endDate": None,
            "precision": None,
            "charOffset": char_offset + start,
            "contextSnippet": _context(text, start, end),
            "unresolvedReason": None,
        }
        if relative and anchor is None:
            entry["unresolvedReason"] = "missing_anchor"
            entries.append(entry)
            continue

        date_range = _try_parse_range(raw, language, anchor, relative)
        if date_range is not None:
            range_start, range_end = date_range
            entry.update({
                "date": range_start.date().isoformat(),
                "endDate": range_end.date().isoformat(),
                "precision": _infer_precision(raw),
            })
            entries.append(entry)
            continue

        parsed = _parse_with_dateparser(raw, language, anchor, relative)
        if parsed is None:
            entry["unresolvedReason"] = "unparseable"
        else:
            entry.update({
                "date": parsed.date().isoformat(),
                "precision": _infer_precision(raw),
            })
        entries.append(entry)
    return entries


def _dedupe_and_sort(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry.get("charOffset") or 0),
            str(entry.get("rawExpression") or ""),
        ),
    )
    kept: List[Dict[str, Any]] = []
    for entry in ordered:
        raw = str(entry.get("rawExpression") or "").casefold()
        offset = int(entry.get("charOffset") or 0)
        end = offset + len(raw)
        duplicate = False
        for existing in kept:
            existing_raw = str(existing.get("rawExpression") or "").casefold()
            existing_offset = int(existing.get("charOffset") or 0)
            existing_end = existing_offset + len(existing_raw)
            if existing_offset == offset and existing_raw == raw:
                duplicate = True
                break
            same_date = (
                entry.get("date") is not None
                and existing.get("date") == entry.get("date")
                and existing.get("endDate") == entry.get("endDate")
                and existing.get("precision") == entry.get("precision")
            )
            if same_date and (
                existing_raw == raw
                or not (existing_end <= offset or end <= existing_offset)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(entry)
    return sorted(
        kept,
        key=lambda entry: (
            str(entry.get("date") or "9999-12-31"),
            int(entry.get("charOffset") or 0),
        ),
    )


@execution_handler("date-extraction-map")
def date_extraction_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("date-extraction-map requires non-empty content")
    config = get_task_config("date-extraction-map")
    if len(content.split()) > int(config.get("max_input_words", 1500)):
        raise ValueError("date-extraction-map content exceeds its word budget")
    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("date-extraction-map language must be a string or null")
    char_offset = payload.get("charOffset", 0)
    if not isinstance(char_offset, int) or isinstance(char_offset, bool) or char_offset < 0:
        raise ValueError("date-extraction-map charOffset must be a non-negative integer")
    return {
        "dates": _extract_dates(
            content.strip(),
            language.strip() if language else None,
            _parse_anchor(payload.get("anchorDate")),
            char_offset,
            config,
        )
    }


@execution_handler("date-extraction-reduce")
def date_extraction_reduce(payload: Dict[str, Any]) -> Dict[str, Any]:
    partials = payload.get("partials")
    if not isinstance(partials, list):
        raise ValueError("date-extraction-reduce partials must be an array")
    entries: List[Dict[str, Any]] = []
    for partial in partials:
        if not isinstance(partial, list) or any(
            not isinstance(entry, dict) for entry in partial
        ):
            raise ValueError("date-extraction-reduce partials must contain date arrays")
        entries.extend(partial)
    return {"dates": _dedupe_and_sort(entries)}
