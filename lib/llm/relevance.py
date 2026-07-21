"""Drop the section units of a document that don't carry substantive content.

Espejo de `documents-dev/models/services/relevance.py`. Única desviación: el
modelo se resuelve con `llm_params_for(cfg)` (la convención de core) en vez de
construir un `ModelSpec`. Como `cfg` es la propia config de la tarea, las
funciones solo reciben `cfg`.

Two independent filters, each registered as its own unit-filter entity (see
`lib.llm.unit_filters`) so a task turns them on by naming them, not with a flag:

1. `heuristic_relevance` (name `relevance`): regex against each unit's heading
   drops sections that look auxiliary (References, Bibliography, Appendix,
   Acknowledgements, Glossary, Index, Copyright, …). Deterministic and free.

2. `llm_relevance` (name `relevance_llm`): one call per `relevance_batch_size`
   units receives a compact `[idx] heading | preview` listing and returns
   `{"keep": [idx, ...]}`. The model only chooses what to discard; it can't add
   or paraphrase. Compose it after `relevance` (`["relevance", "relevance_llm"]`)
   to run the free regex first and the LLM only on survivors.

Fail-open throughout: any parse error, empty result, or LLM unavailability keeps
the input. Neither filter ever returns an empty list for a non-empty input.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from lib.llm.config import llm_params_for
from lib.llm.prompts import load_prompt
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Shown to the model in the filter prompt ("filtering sections for a … task").
# Neutral by default since the filter is now task-agnostic; a task can override
# it with `relevance_task_label` in its config.
_DEFAULT_TASK_LABEL = "content extraction"

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompt_templates")
_RELEVANCE_SYSTEM_PROMPT = load_prompt(_PROMPTS_DIR, "relevance_system.md").strip()
_RELEVANCE_FILTER_PROMPT = load_prompt(_PROMPTS_DIR, "relevance_filter.md")


# Headings that are almost always auxiliary content, regardless of document type.
# Allows an optional short identifier (e.g. "Appendix A", "Annex II") and an
# optional short colon-delimited title ("Appendix B: Notation"). Won't match
# sentences like "References to recent literature" because the trailing words
# don't fit the optional-identifier / colon-title shape.
_AUX_HEADING_RE = re.compile(
    r"^\s*("
    r"references?|bibliograph(y|ies)|works\s+cited|literature\s+cited|"
    r"acknowledgements?|acknowledgments?|"
    r"appendix(es)?|appendices|annex(es)?|"
    r"about\s+the\s+authors?|author\s+bio(graph(y|ies))?|biograph(y|ies)|"
    r"index|glossary|nomenclature|abbreviations?|"
    r"funding|conflicts?\s+of\s+interest|disclosures?|"
    r"copyright|licen[sc]e|disclaimer|legal\s+notice|"
    r"footnotes?|endnotes?|notes?|citations?|"
    r"further\s+reading|see\s+also|"
    r"table\s+of\s+contents|toc|contents|"
    r"publication\s+details?|colophon|imprint|"
    r"erratum|errata|corrigend(um|a)|retraction"
    r")(\s+[A-Za-z0-9][A-Za-z0-9\.\-]*)?\s*([:\-–—].{0,60})?\s*$",
    re.IGNORECASE,
)

# Markdown ATX heading marker (`## `), stripped before the aux regex because the
# units reach here as markdown (html_to_markdown runs upstream).
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")

# Strip leading numbering / "Chapter N:" prefixes before applying the aux regex.
_NUMBERING_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+|chapter\s+\d+\s*[:\-–—]?\s*|section\s+\d+\s*[:\-–—]?\s*)",
    re.IGNORECASE,
)


def _heading_of(unit: str) -> str:
    if not unit:
        return ""
    for line in unit.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _looks_auxiliary_heading(unit: str) -> bool:
    head = _heading_of(unit)
    if not head:
        return False
    # The section heading arrives as markdown (`## References`) because the input
    # pipeline runs html_to_markdown first; strip the ATX marker before matching.
    head = _MD_HEADING_RE.sub("", head, count=1)
    head = _NUMBERING_RE.sub("", head, count=1)
    return bool(_AUX_HEADING_RE.match(head[:80]))


def _heuristic_keep_indices(units: List[str]) -> List[int]:
    return [i for i, u in enumerate(units) if not _looks_auxiliary_heading(u)]


def _preview_for_judgement(unit: str, char_budget: int = 240) -> str:
    lines = [ln.strip() for ln in (unit or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    head = lines[0]
    body = " ".join(lines[1:])
    body_excerpt = body[:char_budget]
    if body_excerpt:
        return f"{head} | {body_excerpt}"
    return head[:char_budget]


def _parse_keep_indices(raw: str, allowed: List[int]) -> Optional[List[int]]:
    """Parse `{"keep": [int, ...]}` from possibly noisy LLM output.

    Returns None if no keep list could be extracted (caller should fail-open).
    Returns an empty list only if the model explicitly returned `keep: []` —
    the caller must treat that as a "model rejected everything" signal.
    """
    if not raw:
        return None
    text = raw.strip()
    # Trim accidental code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    allowed_set = set(allowed)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("keep"), list):
            return [int(x) for x in data["keep"] if isinstance(x, (int, float)) and int(x) in allowed_set]
    except Exception:
        pass
    m = re.search(r'"keep"\s*:\s*\[([\s\S]*?)\]', raw)
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        return [n for n in nums if n in allowed_set]
    return None


def heuristic_relevance(units: List[str], cfg: Dict[str, Any]) -> List[str]:
    """Drop units whose heading looks auxiliary (References, Appendix, …). Free
    and deterministic.

    `relevance_filter_enabled` (default True) disables it; fail-open if it would
    drop everything.
    """
    if not units:
        return units
    if not cfg.get("relevance_filter_enabled", True):
        return units
    # Nothing meaningful to filter on a single unit.
    if len(units) < 2:
        return units

    keep_idx = _heuristic_keep_indices(units)
    if not keep_idx:
        # Dropped everything (e.g. a doc that is only a "References" heading).
        return units
    return [units[i] for i in keep_idx]


def llm_relevance(units: List[str], cfg: Dict[str, Any]) -> List[str]:
    """Ask the LLM which units carry the substantive content and keep those.

    Operates on whatever `units` it receives (compose it after `relevance` to run
    the free regex first). `cfg` resolves the LLM (`llm_params_for`) and tunes it:
    `relevance_filter_enabled` (default True), `relevance_batch_size` (default
    30), `relevance_max_tokens` (default 300), `relevance_task_label` (label shown
    to the model, default `content extraction`). Fail-open on every error path.
    """
    if not units:
        return units
    if not cfg.get("relevance_filter_enabled", True):
        return units
    if len(units) < 2:
        return units
    if not cfg.get("model"):
        return units

    try:
        llm = get_llm_service(**llm_params_for(cfg))
    except Exception as e:
        logger.warning("Relevance filter: LLM unavailable, keeping units (%s)", e)
        return units

    batch_size = int(cfg.get("relevance_batch_size", 30))
    max_tokens = int(cfg.get("relevance_max_tokens", 300))
    task_label = cfg.get("relevance_task_label", _DEFAULT_TASK_LABEL)

    accepted: List[int] = []
    any_batch_succeeded = False
    for start in range(0, len(units), batch_size):
        batch = list(range(start, min(start + batch_size, len(units))))
        listing = "\n".join(
            f"[{i}] {_preview_for_judgement(units[i])}" for i in batch
        )
        instruction = _RELEVANCE_FILTER_PROMPT.format(task_label=task_label, listing=listing)
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("Relevance filter: LLM batch failed, keeping batch (%s)", e)
            accepted.extend(batch)
            continue

        parsed = _parse_keep_indices(raw, batch)
        if parsed is None:
            # Could not parse; keep the batch (fail-open per batch).
            logger.warning("Relevance filter: unparseable LLM reply, keeping batch")
            accepted.extend(batch)
            continue
        any_batch_succeeded = True
        accepted.extend(parsed)

    if not accepted:
        if any_batch_succeeded:
            # Model explicitly rejected every unit — almost certainly wrong.
            logger.warning("Relevance filter: LLM rejected all units; keeping them")
        return units

    return [units[i] for i in sorted(set(accepted))]
