"""Composable pre-chunking unit filters for content tasks.

A *unit filter* drops section units before chunking (see `build_chunks`), so the
LLM never spends context on auxiliary or junk sections. Each filter is a named
entity in `_REGISTRY`; a task declares the ones it wants as a list of names in
its `MapReduceSpec.units_filters` and only those run — `["basic", "relevance"]`.

Two kinds of names:

- **filter**: a single entity (`dedup`, `min_length`, `bullet_lines`,
  `link_density`, `symbol_ratio`, `relevance`, `relevance_llm`).
- **bundle**: shorthand for an ordered list of filters (`basic`, `web`),
  expanded in place — `["basic", "relevance"]` == the bundle's filters followed
  by `relevance`.

The LLM relevance pass is just the `relevance_llm` entity: a task enables it by
naming it (`["relevance", "relevance_llm"]`), not with a config flag.

Every filter only ever DROPS whole units; it never edits or reorders them.
Returning an empty list is treated as fail-open by `build_chunks` (the input is
kept), so a filter can be aggressive without risking an empty document.

All filters here are deterministic, free and language-agnostic (they lean on
Unicode `isalnum`/`\\w`, never on English stopwords or word-length rules that
break on scripts without spaces) except `relevance`, which is the LLM/heuristic
pass in `relevance.py`.

The list can also come from config: when `cfg["units_filters"]` is present it
overrides the factory's default, so a task can be re-tuned — or filtering
disabled with `[]` — without touching code. Per-filter thresholds are read from
`cfg` too (`filter_min_words`, `filter_max_bullet_ratio`, …).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from lib.llm.text import word_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterContext:
    """Everything a filter may need beyond the units themselves: the task config
    (thresholds, model, …) and the job payload."""

    cfg: Dict[str, Any]
    payload: Dict[str, Any]


FilterFn = Callable[[List[str], FilterContext], List[str]]


@dataclass(frozen=True)
class Filter:
    name: str
    apply: FilterFn


_REGISTRY: Dict[str, Filter] = {}
_BUNDLES: Dict[str, Tuple[str, ...]] = {}


def _filter(name: str):
    def register(fn: FilterFn) -> FilterFn:
        _REGISTRY[name] = Filter(name=name, apply=fn)
        return fn

    return register


# ── Filters (each its own entity) ─────────────────────────────────────────────

# Common punctuation that must NOT count as a "symbol" for `symbol_ratio`,
# including the markdown markers our units carry (`#`, `*`, `>`, `|`, …) and
# CJK / fullwidth punctuation so non-Latin prose isn't flagged as junk.
_ALLOWED_PUNCT = set(".,;:!?'\"()[]{}<>#*_`~-–—/\\|@%&+=$€£¥…•·"
                     "。、，！？：；「」『』（）《》〈〉〔〕・")

# A markdown/plain bullet or an ordered-list marker at line start.
_BULLET_RE = re.compile(r"^\s*([-*+•·▪◦‣∙]|\d+[.)]|[A-Za-z][.)])\s+")
# Markdown link `[text](url)` or a bare URL.
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)|https?://\S+")
_WS_RE = re.compile(r"\s+")


def _norm(unit: str) -> str:
    return _WS_RE.sub(" ", unit).strip().lower()


def _alnum_len(unit: str) -> int:
    return sum(1 for c in unit if c.isalnum())


@_filter("dedup")
def _dedup(units: List[str], ctx: FilterContext) -> List[str]:
    """Drop units whose text (whitespace/case-normalized) already appeared —
    repeated headers, footers or nav blocks copied into every section."""
    seen = set()
    out = []
    for u in units:
        key = _norm(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


@_filter("min_length")
def _min_length(units: List[str], ctx: FilterContext) -> List[str]:
    """Drop units with too little content (stray separators, single labels).

    Word count is space-based and undercounts scripts without spaces (CJK, Thai,
    …), so a unit survives on EITHER enough words OR enough alphanumeric
    characters — keeping the filter language-agnostic."""
    min_words = int(ctx.cfg.get("filter_min_words", 4))
    min_chars = int(ctx.cfg.get("filter_min_chars", 16))
    return [
        u for u in units
        if word_count(u) >= min_words or _alnum_len(u) >= min_chars
    ]


@_filter("bullet_lines")
def _bullet_lines(units: List[str], ctx: FilterContext) -> List[str]:
    """Drop list-like units where almost every line is a bullet (a Gopher rule):
    tables of contents, link lists, indexes."""
    max_ratio = float(ctx.cfg.get("filter_max_bullet_ratio", 0.9))
    out = []
    for u in units:
        lines = [ln for ln in u.splitlines() if ln.strip()]
        if not lines:
            continue
        bullets = sum(1 for ln in lines if _BULLET_RE.match(ln))
        if bullets / len(lines) > max_ratio:
            continue
        out.append(u)
    return out


@_filter("link_density")
def _link_density(units: List[str], ctx: FilterContext) -> List[str]:
    """Drop units that are mostly links (nav menus, footers) — jusText's main
    signal for web boilerplate. Measured on non-space characters."""
    max_ratio = float(ctx.cfg.get("filter_max_link_ratio", 0.5))
    out = []
    for u in units:
        total = len(_WS_RE.sub("", u))
        if total == 0:
            continue
        link_chars = sum(len(_WS_RE.sub("", m)) for m in _LINK_RE.findall(u))
        if link_chars / total > max_ratio:
            continue
        out.append(u)
    return out


@_filter("symbol_ratio")
def _symbol_ratio(units: List[str], ctx: FilterContext) -> List[str]:
    """Drop units dominated by non-text symbols (raw tables, ascii art, leftover
    markup). `isalnum` is Unicode-aware, so any script's letters count as text —
    only genuinely odd symbols are penalized."""
    max_ratio = float(ctx.cfg.get("filter_max_symbol_ratio", 0.3))
    out = []
    for u in units:
        chars = [c for c in u if not c.isspace()]
        if not chars:
            continue
        symbols = sum(1 for c in chars if not c.isalnum() and c not in _ALLOWED_PUNCT)
        if symbols / len(chars) > max_ratio:
            continue
        out.append(u)
    return out


@_filter("relevance")
def _relevance(units: List[str], ctx: FilterContext) -> List[str]:
    """Free regex drop of auxiliary-heading sections (References, Appendix, …)."""
    from lib.llm.relevance import heuristic_relevance

    return heuristic_relevance(units, ctx.cfg)


@_filter("relevance_llm")
def _relevance_llm(units: List[str], ctx: FilterContext) -> List[str]:
    """LLM pass that drops sections which are auxiliary by MEANING, not by form.
    Enabled simply by naming it in the list; imported lazily so the cheap filters
    never pull in the LLM service. Resolves the model and label from `ctx.cfg`."""
    from lib.llm.relevance import llm_relevance

    return llm_relevance(units, ctx.cfg)


# ── Bundles ───────────────────────────────────────────────────────────────────

# Universal, safe on prose: never touches genuine content, only obvious junk.
_BUNDLES["basic"] = ("dedup", "min_length", "bullet_lines")
# For web-sourced documents: adds link/symbol density on top of `basic`.
_BUNDLES["web"] = ("dedup", "min_length", "bullet_lines", "link_density", "symbol_ratio")


def _resolve(names: List[str]) -> List[Filter]:
    """Expand bundles and map names to their entities, preserving order and
    dropping duplicates. Unknown names are logged and skipped (fail-open)."""
    resolved: List[Filter] = []
    seen = set()
    for name in names:
        for fname in _BUNDLES.get(name, (name,)):
            if fname in seen:
                continue
            filt = _REGISTRY.get(fname)
            if filt is None:
                logger.warning("Unit filter %r is not registered; skipping", fname)
                continue
            seen.add(fname)
            resolved.append(filt)
    return resolved


def build_units_filter(names, payload: Dict[str, Any], cfg: Dict[str, Any]):
    """Compose the named filters into a single `units -> units` callable for
    `build_chunks`, running each in order — a filter runs only because it was
    named.

    `names` is the task's default list of filter/bundle names (from
    `MapReduceSpec.units_filters`); `cfg["units_filters"]` overrides it when
    present (`[]` disables filtering). Returns `None` when the resolved list is
    empty so the orchestrator skips filtering entirely. Each filter reads
    whatever it needs (thresholds, model) straight from `cfg`.
    """
    names = cfg.get("units_filters", names)
    if not names:
        return None
    entities = _resolve(list(names))
    if not entities:
        return None
    ctx = FilterContext(cfg=cfg, payload=payload)

    def run(units: List[str]) -> List[str]:
        for filt in entities:
            units = filt.apply(units, ctx) or units
        return units

    return run
