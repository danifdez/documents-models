"""Agentic date-extraction task.

Runs through the shared inline map-reduce helper (`lib.llm.map_reduce`):

- The invocation cleans and filters the text before chunking it.
- Each chunk receives its global character offset and returns dated entries;
  the reduce step concatenates, deduplicates and sorts them.

The worker extracts **in the document's own language**. The backend detects the
language and enqueues the original text with it (see `detect-language-processor`:
"Original language is preserved — no translation"), so `payload["language"]` is
authoritative and is threaded down to dateparser. Hardcoding "en" here silently
loses every relative expression and every spelled-out month in a non-English
document.

Per-chunk LLM fallback budget (`chunk_max_llm_fallbacks`, default 5) replaces
the previous global `max_llm_fallbacks=10`, so long documents no longer get
silently truncated.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dateparser

from lib.llm.config import get_llm_params, get_task_config
from lib.llm.grammars import DATE_RESOLUTION_GBNF, STRING_ARRAY_GBNF
from lib.llm.map_reduce import (
    InlineListMapReduceSpec,
    run_inline_list_map_reduce,
)
from lib.llm.prompts import get_prompt
from services.llm_service import get_llm_service
from services.relevance import select_relevant_units
from services.text import (
    chunk_units,
    extract_section_units,
    html_to_markdown,
    strip_dense_blobs,
)
from common.execution_registry import execution_handler

logger = logging.getLogger(__name__)


_RANGE_SEPARATORS_RE = re.compile(
    r"\s+(?:-|–|—|to|al|hasta|a|y|and|e|until|till)\s+|\s*-\s*|\s*–\s*|\s*—\s*",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
# All five supported languages: a month this misses degrades the entry to
# `precision: "year"`, so "20. Juli 1969" would land as a bare year.
_MONTH_NAME_RE = re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"ene(ro)?|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
    r"janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre|"
    r"januar|februar|m[aä]rz|juni|juli|okt(ober)?|dez(ember)?|"
    r"gennaio|febbraio|aprile|maggio|giugno|luglio|settembre|ottobre|dicembre)\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"
)

_WEEKDAY_RE = re.compile(
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"\b(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)\b|"
    r"\b(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\b|"
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b|"
    r"\b(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|venerd[ìi]|sabato|domenica)\b",
    re.IGNORECASE,
)

# Deictic markers: they point at a moment from an anchor. Paired with a time
# unit below, they turn a quantity into a date ("hace 3 días"); alone, a unit
# is only a duration.
_RELATIVE_MARKER_RE = re.compile(
    r"\bhace\b|\bdentro de\b|\bque viene\b|\bpr[oó]xim\w*\b|\bpasad\w*\b|"
    r"\bago\b|\bin\b|\bnext\b|\blast\b|\bwithin\b|"
    r"\bvor\b|\bn[aä]chst\w*\b|\bletzt\w*\b|"
    r"\bil y a\b|\bdans\b|\bprochain\w*\b|\bdernier\w*\b|"
    r"\bfa\b|\btra\b|\bfra\b|\bprossim\w*\b|\bscors\w*\b",
    re.IGNORECASE,
)

# Self-contained relative dates: no unit needed, they already name a day.
_STANDALONE_RELATIVE_RE = re.compile(
    r"\b(ayer|hoy|ma[nñ]ana|anteayer|anoche)\b|"
    r"\b(yesterday|today|tomorrow|tonight)\b|"
    r"\b(gestern|heute|morgen|vorgestern)\b|"
    r"\b(hier|aujourd'hui|demain|avant-hier)\b|"
    r"\b(ieri|oggi|domani|stanotte)\b",
    re.IGNORECASE,
)

_TIME_UNIT_RE = re.compile(
    r"\b(d[ií]as?|semanas?|meses|mes|a[nñ]os?|horas?|minutos?|d[eé]cadas?|siglos?)\b|"
    r"\b(days?|weeks?|months?|years?|hours?|minutes?|decades?|centur(y|ies))\b|"
    r"\b(tage?n?|wochen?|monate?n?|jahre?n?|stunden?|minuten?|jahrzehnte?n?)\b|"
    r"\b(jours?|semaines?|mois|ann[eé]es?|heures?|minutes?|d[eé]cennies?|si[eè]cles?)\b|"
    r"\b(giorni?|settimane?|mesi|anni?|ore|ora|minuti?|decenni?|secoli?)\b",
    re.IGNORECASE,
)


_DETECT_PROMPT = get_prompt("date-extraction", "detect_prompt.md")
_RESOLVE_PROMPT = get_prompt("date-extraction", "resolve_prompt.md")
_TITLE_PROMPT = get_prompt("date-extraction", "title_prompt.md")


# The model decorates the span it copies (`_July 20, 1969_`, `@3 días`) even
# when told to quote verbatim, and `_locate` matches against the undecorated
# document: left alone, the markers make the one expression we actually care
# about unlocatable, so it gets dropped while the undecorated noise survives.
# Edges only — a marker inside the span is part of what the model copied.
_DECOR_RE = re.compile(r"^[\s_*`~@#]+|[\s_*`~@#]+$")


_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_expression_array(response: str) -> List[str]:
    """Parse the detector's array, salvaging a truncated one.

    Greedy decoding with no repetition penalty makes the model loop over the
    same handful of spans until `max_tokens` cuts it off mid-string. Letting
    `json.loads` fail there would throw away the dozens of complete, correct
    expressions that precede the cut — including, in a chunked document, the
    only date in that chunk. Every closed string before the truncation is
    valid output and is kept; the dangling one has no closing quote and is
    naturally skipped.
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, str)]
        return []
    except json.JSONDecodeError:
        pass

    salvaged: List[str] = []
    for match in _JSON_STRING_RE.findall(response):
        try:
            salvaged.append(json.loads(f'"{match}"'))
        except json.JSONDecodeError:
            continue
    if salvaged:
        logger.warning(
            "date-extraction detection was truncated; salvaged %d complete "
            "expressions out of a malformed array", len(salvaged),
        )
    return salvaged


def _detect_expressions(text: str, cfg: Dict[str, Any]) -> List[str]:
    if not text.strip() or not _DETECT_PROMPT:
        return []
    try:
        response = get_llm_service(**get_llm_params("date-extraction")).chat(
            [{"role": "user", "content": _DETECT_PROMPT.format(text=text)}],
            max_tokens=int(cfg.get("detect_max_tokens", cfg.get("max_tokens", 800))),
            grammar=STRING_ARRAY_GBNF, temperature=0.0,
        )
    except Exception:
        logger.exception("date-extraction detection failed")
        return []
    cleaned = [_DECOR_RE.sub("", item) for item in _parse_expression_array(response)]
    # Exact duplicates only: the model loops and repeats spans verbatim. Do
    # NOT drop a span because a longer one contains it — the detector often
    # returns both "hace 3 días" and the whole sentence around it, and the
    # date is the short one.
    return list(dict.fromkeys(item for item in cleaned if len(item) >= 2))


def _assign_titles(text: str, entries: List[Dict[str, Any]]) -> None:
    if not entries or not _TITLE_PROMPT:
        return
    try:
        response = get_llm_service(**get_llm_params("date-extraction")).chat(
            [{"role": "user", "content": _TITLE_PROMPT.format(
                text=text, expressions=json.dumps([entry["rawExpression"] for entry in entries]),
            )}], max_tokens=400, temperature=0.0,
        )
        titles = json.loads(response)
    except Exception:
        logger.exception("date-extraction titling failed")
        return
    pending = {}
    for entry in entries:
        pending.setdefault(entry["rawExpression"].lower(), []).append(entry)
    for title in titles if isinstance(titles, list) else []:
        if not isinstance(title, dict) or not title.get("title"):
            continue
        matches = pending.get(str(title.get("expression") or "").lower(), [])
        if matches:
            matches.pop(0)["title"] = str(title["title"]).strip()


def _locate(text: str, expression: str, cursor: int) -> int:
    """First occurrence at or after `cursor`, falling back to the whole text.

    The cursor stops repeated expressions from collapsing onto one offset, but
    it must not make a span that IS in the document unfindable. The detector
    does not answer in document order — it may return a truncated copy of a
    date before the date itself — and scanning forward only meant the better
    expression, sitting behind the cursor, was silently dropped.
    """
    lowered_text, lowered_expr = text.lower(), expression.lower()
    for start in (cursor, 0):
        position = text.find(expression, start)
        if position >= 0:
            return position
        position = lowered_text.find(lowered_expr, start)
        if position >= 0:
            return position
    return -1


def _classify(expression: str) -> Optional[str]:
    """"absolute", "relative", or None when the span is not a date at all.

    The detector is an 8B model told to skip money, durations and counts; it
    does so unreliably, so this is where "is this even a date?" is decided.
    Answering it by elimination — anything without a year must be relative —
    is what let "149,90 euros" and "8 sesiones" through as relative dates
    awaiting an anchor, and a timeline is not the place for a price.

    A date names a point on the calendar: either directly (a year, a numeric
    date, a month name) or by pointing at one from an anchor ("hace 3 días",
    "yesterday"). A bare quantity of time is a DURATION, not a date — that is
    why a time unit alone ("90 minutos", "dos años") is not enough; it needs a
    deictic marker to become a point in time.
    """
    if _YEAR_RE.search(expression) or _NUMERIC_DATE_RE.search(expression):
        return "absolute"
    if _MONTH_NAME_RE.search(expression) or _WEEKDAY_RE.search(expression):
        return "absolute"
    if _STANDALONE_RELATIVE_RE.search(expression):
        return "relative"
    if _RELATIVE_MARKER_RE.search(expression) and _TIME_UNIT_RE.search(expression):
        return "relative"
    return None


def _infer_precision(expression: str, parsed: datetime) -> str:
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
    task_config = get_task_config("date-extraction")
    if not task_config.get("enable_llm_fallback", True):
        return None

    llm_model_name = task_config.get("llm_model")
    if not llm_model_name:
        return None

    try:
        params = get_llm_params("date-extraction", model_name=llm_model_name)
        model_path = params.get("model_path")
        if not model_path or not os.path.isfile(model_path):
            logger.warning("LLM fallback model not found: %s", model_path)
            return None
        llm = get_llm_service(**params)
    except Exception:
        logger.exception("LLM fallback unavailable for date-extraction")
        return None

    anchor_str = anchor_date or "UNKNOWN"
    prompt = get_prompt("date-extraction", "resolve_prompt.md").format(
        anchor_str=anchor_str,
        language=language or "unknown",
        expression=expression,
        context=context,
    )
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=120,
            grammar=DATE_RESOLUTION_GBNF,
            temperature=0.0,
        )
    except Exception:
        logger.exception("LLM fallback chat failed")
        return None

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Grammar-constrained output should always be valid JSON; only an
        # empty/aborted generation lands here.
        logger.warning("date-extraction fallback returned unparseable output: %r", response[:200])
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk extraction core (NER + dateparser + LLM fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_from_text(
    text: str,
    language: Optional[str],
    anchor_dt: Optional[datetime],
    anchor_date_str: Optional[str],
    char_offset: int,
    cfg: Dict[str, Any],
    *,
    llm_budget_remaining: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Run NER + dateparser + LLM fallback over a single (already cleaned)
    text blob. Returns (entries, llm_fallbacks_consumed). All charOffset
    values in the returned entries are global (caller's `char_offset` plus
    the local position within `text`).
    """
    if not text:
        return [], 0

    entries: List[Dict[str, Any]] = []
    consumed = 0
    seen_local = set()
    cursor = 0

    for raw in _detect_expressions(text, cfg):
        local_start = _locate(text, raw, cursor)
        if local_start < 0:
            continue
        local_end = local_start + len(raw)
        cursor = local_end
        span_key = (local_start, local_end, raw.lower())
        if span_key in seen_local:
            continue
        seen_local.add(span_key)

        kind = _classify(raw)
        if kind is None:
            # Detector noise (a price, a count, a duration). Dropping it here
            # also saves the LLM fallback it would otherwise burn.
            continue

        global_start = char_offset + local_start
        is_relative = kind == "relative"
        snippet = _build_context_snippet(text, local_start, local_end)

        entry: Dict[str, Any] = {
            "rawExpression": raw,
            "date": None,
            "endDate": None,
            "precision": None,
            "charOffset": global_start,
            "contextSnippet": snippet,
            "resolver": "unresolved",
            "isRelative": is_relative,
            "unresolvedReason": None,
        }

        if is_relative and anchor_dt is None:
            entry["unresolvedReason"] = "missing_anchor"
            entries.append(entry)
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
            entries.append(entry)
            continue

        parsed = _parse_with_dateparser(raw, language or "", anchor_dt, is_relative)
        if parsed is not None:
            entry.update({
                "date": parsed.date().isoformat(),
                "precision": _infer_precision(raw, parsed),
                "resolver": "dateparser",
            })
            entries.append(entry)
            continue

        if consumed < llm_budget_remaining:
            consumed += 1
            llm_result = _llm_fallback(raw, snippet, language or "", anchor_date_str)
            if llm_result and not llm_result.get("unresolved"):
                entry.update({
                    "date": llm_result["date"],
                    "endDate": llm_result.get("endDate"),
                    "precision": llm_result["precision"],
                    "resolver": "llm",
                })
                entries.append(entry)
                continue
            if llm_result and llm_result.get("unresolved"):
                entry["unresolvedReason"] = llm_result.get("reason") or "ambiguous"

        if entry["unresolvedReason"] is None:
            entry["unresolvedReason"] = "unparseable"
        entries.append(entry)

    _assign_titles(text, entries)
    return entries, consumed


# ─────────────────────────────────────────────────────────────────────────────
# Dedup + sort for the merge phase
# ─────────────────────────────────────────────────────────────────────────────


def _dedupe_and_sort(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicates and overlap-near-duplicates, then sort.

    Two entries are considered duplicates when:
      - same `(charOffset, raw.lower())`, OR
      - same resolved `(date, endDate, precision)` AND either the raw text
        matches case-insensitively OR the character ranges overlap.

    Among duplicates we keep the one with the smallest `charOffset` to give
    the timeline UI a stable anchor.
    """
    if not entries:
        return entries

    sorted_entries = sorted(
        entries, key=lambda e: (e.get("charOffset") or 0, e.get("rawExpression") or "")
    )

    kept: List[Dict[str, Any]] = []
    for e in sorted_entries:
        raw = (e.get("rawExpression") or "").lower()
        off = int(e.get("charOffset") or 0)
        end = off + len(e.get("rawExpression") or "")
        d = e.get("date")
        ed = e.get("endDate")
        prec = e.get("precision")
        is_dup = False
        for k in kept:
            k_raw = (k.get("rawExpression") or "").lower()
            k_off = int(k.get("charOffset") or 0)
            k_end = k_off + len(k.get("rawExpression") or "")
            if k_off == off and k_raw == raw:
                is_dup = True
                break
            if d is not None and (k.get("date") == d
                                  and k.get("endDate") == ed
                                  and k.get("precision") == prec):
                if k_raw == raw:
                    is_dup = True
                    break
                # range overlap
                if not (k_end <= off or end <= k_off):
                    is_dup = True
                    break
        if not is_dup:
            kept.append(e)

    kept.sort(key=lambda r: (r.get("date") or "9999-12-31", r.get("charOffset") or 0))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Map-reduce spec
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_offsets(cleaned: str, chunks: List[str]) -> List[int]:
    """Best-effort mapping from chunk index to its offset within `cleaned`.
    `chunk_units` joins units with `\\n\\n`; `find` is sufficient when the
    chunks are reasonably long. Falls back to the running cursor when an
    exact match isn't located (e.g. due to whitespace normalization)."""
    offsets: List[int] = []
    cursor = 0
    for c in chunks:
        head = c[:200] if len(c) > 200 else c
        pos = cleaned.find(head, cursor) if head else -1
        if pos < 0:
            pos = cursor
        offsets.append(pos)
        cursor = pos + len(c)
    return offsets


def _language_of(payload: Dict[str, Any]) -> Optional[str]:
    return (payload.get("language") or "").strip() or None


def _parse_anchor_quiet(anchor_date: Optional[str]) -> Optional[datetime]:
    """`_parse_anchor` without the warning: the plan phase (`_chunks`) already
    logged it once per invocation; leaves must not repeat it per chunk."""
    if not anchor_date:
        return None
    try:
        return datetime.strptime(anchor_date[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _chunks(payload: Dict[str, Any], cfg: Dict[str, Any], is_child: bool) -> List[str]:
    text = payload.get("text") or ""
    language = _language_of(payload)
    # Called for its warning only, so an invalid anchorDate is still logged
    # exactly once per invocation; the value is re-parsed quietly in each leaf.
    _parse_anchor(payload.get("anchorDate"))

    if not language:
        # dateparser can autodetect, but it is slower and more ambiguous than
        # being told; the backend normally resolves this before enqueueing.
        logger.warning("date-extraction got no language; falling back to autodetect")

    if not text:
        return []

    cleaned = strip_dense_blobs(html_to_markdown(text))
    units = extract_section_units(cleaned)
    if not units:
        return []

    if not is_child and cfg.get("relevance_filter_enabled", True):
        units = select_relevant_units(
            units, cfg, task_label="date extraction", target_lang="en",
        ) or units

    return chunk_units(units, int(cfg.get("chunk_word_budget", 1500)), joiner="\n\n")


def _leaf(chunk: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the entries of one chunk. Pass the declared language through
    rather than asserting "en": dateparser is multilingual, and hardcoding
    English makes it miss every non-English expression ("marzo de 2020",
    "hace 3 días") when the backend did not translate.

    Children (and the in-process fallback) carry the chunk's global offset in
    `_chunk_offset`; a top-level single chunk starts at 0. When the cleaning
    re-chunks a child's input further (rare), every piece is processed against
    the same base offset; the loss of precision is acceptable for retries.
    """
    anchor_date = payload.get("anchorDate")
    entries, _ = _extract_from_text(
        chunk,
        _language_of(payload),
        _parse_anchor_quiet(anchor_date),
        anchor_date,
        char_offset=int(payload.get("_chunk_offset", 0)),
        cfg=cfg,
        llm_budget_remaining=int(cfg.get("chunk_max_llm_fallbacks", 5)),
    )
    return entries


def _reduce(partials: List[Any], payload: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_entries: List[Dict[str, Any]] = []
    for entries in partials:
        for d in (entries or []):
            if isinstance(d, dict):
                all_entries.append(d)
    return _dedupe_and_sort(all_entries)


def _leaf_payload_extras(
    chunks: List[str], payload: Dict[str, Any], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    # Re-cleans the text (cheap and deterministic) because the chunk offsets
    # are measured against the cleaned document, not the raw payload.
    cleaned = strip_dense_blobs(html_to_markdown(payload.get("text") or ""))
    chunk_offsets = _chunk_offsets(cleaned, chunks)
    return [{"_chunk_offset": off} for off in chunk_offsets]


_SPEC = InlineListMapReduceSpec(
    leaf_fn=_leaf,
    reduce_fn=_reduce,
    chunks_fn=_chunks,
    result_key="dates",
    leaf_payload_extras_fn=_leaf_payload_extras,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@execution_handler("date-extraction")
def extract_dates(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract dates from every inline chunk and merge the results.

    Payload (top-level):
        text: str — the document content (HTML or plain).
        language: str — the document's own language, as detected by the
            backend. Used verbatim for parsing; not translated.
        anchorDate: str | null — YYYY-MM-DD, the resource's publication date.
    """
    try:
        # Module-level binding on purpose: a function-local re-import would
        # dodge any patch a test harness applies to this module to override
        # the task config (chunk_word_budget & friends).
        cfg = get_task_config("date-extraction")
        return run_inline_list_map_reduce(payload, spec=_SPEC, cfg=cfg)
    except Exception as e:
        logger.exception("date-extraction failed")
        return {"error": f"date-extraction failed: {e}"}
