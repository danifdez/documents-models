"""Agentic date-extraction task.

Mirrors the state-machine pattern of `summarize` and `key-point`:

- Top-level invocation cleans the text (HTML → markdown, strip dense blobs),
  splits it into section units, optionally drops auxiliary sections via the
  shared relevance filter, and chunks the survivors. If the result is one
  chunk, the full pipeline runs inline. Otherwise, one child `date-extraction`
  job per chunk is fanned out and the parent waits.
- Each child receives a single chunk plus its global character offset and
  returns a list of dated entries with global `charOffset` already applied.
- Once all children finish, the dispatcher re-invokes the handler with the
  persisted state; `_phase_merge` concatenates, dedupes, and sorts.

The worker is **English-only**. The backend is responsible for passing
`workingContent` (the translated-to-English text) for non-English resources
and `language="en"`. If the payload arrives in another language we still run
spaCy in English and emit a warning.

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
import spacy

from services.grammars import DATE_RESOLUTION_GBNF
from services.relevance import select_relevant_units
from services.text import (
    chunk_units,
    extract_section_units,
    html_to_markdown,
    strip_dense_blobs,
)
from utils.device import HAS_CUDA, get_spacy_model
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


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
_NUMERIC_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"
)


_nlp = None


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
    if _YEAR_RE.search(expression):
        return True
    if _NUMERIC_DATE_RE.search(expression):
        return True
    return False


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
    from services.model_config import get_task_config, get_llm_params
    from services.llm_service import get_llm_service

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

    nlp = _get_nlp()
    # spaCy default max_length is 1_000_000. Bump it generously rather than
    # crash on a fat chunk; the call site already chunks by word budget so this
    # is mostly a safety belt.
    needed = len(text) + 100_000
    if getattr(nlp, "max_length", 0) < needed:
        nlp.max_length = needed

    doc = nlp(text)
    entries: List[Dict[str, Any]] = []
    consumed = 0
    seen_local = set()

    for ent in doc.ents:
        if ent.label_ != "DATE":
            continue
        raw = ent.text.strip()
        if len(raw) < 2:
            continue
        local_start = ent.start_char
        local_end = ent.end_char
        span_key = (local_start, local_end, raw.lower())
        if span_key in seen_local:
            continue
        seen_local.add(span_key)

        global_start = char_offset + local_start
        is_relative = not _is_absolute(raw)
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
# Phases
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


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    text = payload.get("text") or ""
    language = (payload.get("language") or "").strip() or None
    anchor_date = payload.get("anchorDate")
    anchor_dt = _parse_anchor(anchor_date)
    is_child = "_chunk_idx" in payload
    chunk_offset_in = int(payload.get("_chunk_offset", 0)) if is_child else 0

    if language and language.lower() != "en":
        logger.warning(
            "date-extraction worker is English-only but received language=%s; "
            "the backend should pass the translated workingContent.", language,
        )

    if not text:
        return {"dates": []}

    cleaned = strip_dense_blobs(html_to_markdown(text))
    units = extract_section_units(cleaned)
    if not units:
        return {"dates": []}

    if not is_child and cfg.get("relevance_filter_enabled", True):
        units = select_relevant_units(
            units, cfg, task_label="date extraction", target_lang="en",
        ) or units

    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    chunks = chunk_units(units, chunk_word_budget, joiner="\n\n")
    if not chunks:
        return {"dates": []}

    chunk_max_llm = int(cfg.get("chunk_max_llm_fallbacks", 5))

    # CHILD: receive a single chunk plus its global offset.
    if is_child:
        # If the cleaning re-chunks the input further (rare — child payloads
        # are already a single chunk), process every piece against the same
        # base offset; the loss of precision is acceptable for retries.
        all_entries: List[Dict[str, Any]] = []
        for c in chunks:
            entries, _ = _extract_from_text(
                c, language, anchor_dt, anchor_date,
                char_offset=chunk_offset_in, cfg=cfg,
                llm_budget_remaining=chunk_max_llm,
            )
            all_entries.extend(entries)
        return {"dates": all_entries}

    # TOP-LEVEL with a single chunk: full pipeline inline.
    if len(chunks) == 1:
        entries, _ = _extract_from_text(
            chunks[0], "en", anchor_dt, anchor_date,
            char_offset=0, cfg=cfg,
            llm_budget_remaining=chunk_max_llm,
        )
        return {"dates": _dedupe_and_sort(entries)}

    chunk_offsets = _chunk_offsets(cleaned, chunks)

    # No DB context (unit tests / fallback): in-process serial.
    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        all_entries = []
        for c, off in zip(chunks, chunk_offsets):
            entries, _ = _extract_from_text(
                c, "en", anchor_dt, anchor_date,
                char_offset=off, cfg=cfg,
                llm_budget_remaining=chunk_max_llm,
            )
            all_entries.extend(entries)
        return {"dates": _dedupe_and_sort(all_entries)}

    # FAN-OUT: one child per chunk, each carrying its global offset.
    pending: Dict[str, int] = {}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    retries: Dict[str, int] = {}
    for i, (chunk, off) in enumerate(zip(chunks, chunk_offsets)):
        child_payload = {
            "text": chunk,
            "language": "en",
            "anchorDate": anchor_date,
            "_chunk_idx": i,
            "_chunk_offset": off,
        }
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id,
            "date-extraction",
            payload=child_payload,
            agent_max_steps=1,
        )
        if child_id is None:
            return {"error": f"failed to enqueue child for chunk {i}"}
        pending[str(child_id)] = i
        results[str(i)] = None
        retries[str(i)] = 0

    state = {
        "phase": "merging",
        "chunks_count": len(chunks),
        "pending": pending,
        "results": results,
        "retries": retries,
        "chunks": chunks,
        "chunk_offsets": chunk_offsets,
        "language": "en",
        "anchorDate": anchor_date,
        "chunk_field": "text",
        "chunk_payload_template": {
            "language": "en",
            "anchorDate": anchor_date,
        },
    }

    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _phase_merge(state: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return {
            "error": (
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            )
        }

    n = int(state.get("chunks_count", 0))
    results = state.get("results") or {}
    all_entries: List[Dict[str, Any]] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            for d in (r.get("dates") or []):
                if isinstance(d, dict):
                    all_entries.append(d)

    return {"dates": _dedupe_and_sort(all_entries)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@job_handler("date-extraction")
def extract_dates(
    payload: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    ctx=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete.

    Payload (top-level):
        text: str — the document content (HTML or plain).
        language: str — should be "en"; backend resolves workingContent for
            non-English resources before enqueueing.
        anchorDate: str | null — YYYY-MM-DD, the resource's publication date.

    Children additionally carry `_chunk_idx` and `_chunk_offset`.
    """
    try:
        from services.model_config import get_task_config
        cfg = get_task_config("date-extraction")
        if state and state.get("phase") == "merging":
            return _phase_merge(state, cfg, ctx)
        return _phase_plan_or_leaf(payload, cfg, ctx)
    except Exception as e:
        logger.exception("date-extraction failed")
        return {"error": f"date-extraction failed: {e}"}
