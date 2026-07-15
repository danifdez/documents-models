"""Pre-filter the section units of a document to keep only what's relevant
for summarization-style tasks (summarize, key-point).

Two-stage pipeline:

1. **Heuristic**: regex against the first non-empty line of each unit drops
   sections whose heading looks auxiliary (References, Bibliography, Appendix,
   Acknowledgements, Glossary, Index, Copyright, etc.). Deterministic and free.

2. **LLM (Phi)**: a single Phi call (batched if many units) receives a compact
   `[idx] heading | preview` listing of each surviving unit and returns
   `{"keep": [idx, ...]}`. The model only chooses what to discard; it can't
   add or paraphrase.

Fail-open: any parse error, empty result, or LLM unavailability returns the
heuristic result (or, in the worst case, the original list). The function
must never return an empty list when given a non-empty input.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from agent.llm import get_llm_for_spec
from agent.types import ModelSpec
from lib.llm.prompts import load_prompt

logger = logging.getLogger(__name__)

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


def select_relevant_units(
    units: List[str],
    cfg: Dict[str, Any],
    *,
    task_label: str,
    target_lang: str = "en",
) -> List[str]:
    """Return a subset of `units` keeping only sections relevant to `task_label`.

    `cfg` is the task config dict (must contain `model` for the LLM step).
    Recognized keys: `relevance_filter_enabled` (default True),
    `relevance_batch_size` (default 30), `relevance_max_tokens` (default 300).
    """
    if not units:
        return units
    if not cfg.get("relevance_filter_enabled", True):
        return units
    # Nothing meaningful to filter on a single unit.
    if len(units) < 2:
        return units

    heuristic_idx = _heuristic_keep_indices(units)
    if not heuristic_idx:
        # Heuristic dropped everything (e.g. a doc consisting only of a "References"
        # heading because the structure is unusual). Fail-open.
        return units

    model_path = cfg.get("model")
    if not model_path:
        return [units[i] for i in heuristic_idx]

    try:
        llm = get_llm_for_spec(ModelSpec(path=str(model_path)))
    except Exception as e:
        logger.warning("Relevance filter: LLM unavailable, using heuristic only (%s)", e)
        return [units[i] for i in heuristic_idx]

    batch_size = int(cfg.get("relevance_batch_size", 30))
    max_tokens = int(cfg.get("relevance_max_tokens", 300))

    accepted: List[int] = []
    any_batch_succeeded = False
    for start in range(0, len(heuristic_idx), batch_size):
        batch = heuristic_idx[start:start + batch_size]
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

    # Safety nets.
    if not accepted:
        if any_batch_succeeded:
            # Model explicitly rejected every unit across all batches. That's
            # almost certainly wrong — fail-open to the heuristic result.
            logger.warning("Relevance filter: LLM rejected all units; reverting to heuristic")
        return [units[i] for i in heuristic_idx]

    # Preserve original order; dedupe in case overlapping batches ever occurred.
    seen = set()
    ordered = []
    for i in sorted(set(accepted)):
        if i in seen:
            continue
        seen.add(i)
        ordered.append(units[i])
    return ordered
