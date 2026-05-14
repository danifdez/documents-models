"""Agentic summarization task.

Reentrant handler that drives a small state machine via the job system:

- First invocation: chunk the content. If a single chunk fits, summarize inline
  with Phi and finish. If not, fan-out one child `summarize` job per chunk
  (each child receives one chunk and falls into the leaf branch).
- Once all children finish, the dispatcher wakes the parent and re-invokes the
  handler with the persisted `state`. The handler merges all partial summaries
  with Phi. If the merge is still too large, it fans out again recursively.

Chunk failures are retried by `_maybe_resume_parent` in the dispatcher up to
`chunk_max_retries`. If retries are exhausted, the parent is woken and this
handler returns `{error: ...}`.
"""

from typing import Any, Callable, Dict, List, Optional

from agent.llm import get_llm_for_spec
from agent.types import ModelSpec
from services.model_config import get_llm_defaults, get_task_config
from services.relevance import select_relevant_units
from services.text import (
    chunk_units,
    extract_section_units,
    html_to_markdown,
    strip_dense_blobs,
)
from utils.job_registry import job_handler


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _char_budget(cfg: Dict[str, Any]) -> int:
    """Approximate max chars that fit alongside the prompt in the LLM context.

    Uses ~4 chars/token (English heuristic) and reserves room for the system
    prompt and output. Honours overrides via `summarize.input_char_budget`.
    """
    override = cfg.get("input_char_budget")
    if override is not None:
        return int(override)
    n_ctx = int(get_llm_defaults().get("n_ctx", 32768))
    out_tokens = int(cfg.get("chunk_max_tokens", 400))
    # Leave 512 tokens of headroom for the prompt boilerplate.
    available_tokens = max(512, n_ctx - out_tokens - 512)
    return available_tokens * 4


def _truncate_for_llm(text: str, cfg: Dict[str, Any]) -> str:
    cap = _char_budget(cfg)
    if len(text) <= cap:
        return text
    return text[:cap]


def _build_chunks(
    content: str,
    chunk_word_budget: int,
    *,
    units_filter: Optional[Callable[[List[str]], List[str]]] = None,
) -> List[str]:
    cleaned = strip_dense_blobs(html_to_markdown(content or ""))
    units = extract_section_units(cleaned)
    if not units:
        return []
    if units_filter is not None:
        units = units_filter(units) or units
    return chunk_units(units, chunk_word_budget, joiner="\n\n")


def _phi_summarize(text: str, target_language: str, cfg: Dict[str, Any]) -> str:
    spec = ModelSpec(path=cfg["model"])
    llm = get_llm_for_spec(spec)
    max_tokens = int(cfg.get("chunk_max_tokens", 400))
    safe_text = _truncate_for_llm(strip_dense_blobs(text), cfg)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. Produce a faithful summary that "
                "preserves key facts, names, and numbers. Do not add information "
                "that is not in the source. Output the summary directly, without "
                "preamble or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following text in {target_language}. "
                f"Keep it under {max_tokens} tokens.\n\n{safe_text}"
            ),
        },
    ]
    return llm.chat(messages, max_tokens=max_tokens).strip()


def _phi_merge(partials: List[str], target_language: str, cfg: Dict[str, Any]) -> str:
    spec = ModelSpec(path=cfg["model"])
    llm = get_llm_for_spec(spec)
    max_tokens = int(cfg.get("merge_max_tokens", 800))
    joined = "\n\n---\n\n".join(
        f"[part {i + 1}]\n{p}" for i, p in enumerate(partials) if p
    )
    joined = _truncate_for_llm(joined, cfg)
    messages = [
        {
            "role": "system",
            "content": (
                "You combine partial summaries of one document into a single "
                "coherent summary. Remove redundancy, keep all distinct facts, "
                "preserve order where it matters. Output the merged summary "
                "directly, without preamble or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Combine these partial summaries into a single coherent summary "
                f"in {target_language}. Keep it under {max_tokens} tokens.\n\n"
                f"{joined}"
            ),
        },
    ]
    return llm.chat(messages, max_tokens=max_tokens).strip()


def _phase_plan_or_leaf(payload: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    target_language = payload.get("targetLanguage") or "en"
    source_language = payload.get("sourceLanguage")
    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    is_child = "_chunk_idx" in payload

    units_filter = None
    if not is_child:
        units_filter = lambda us: select_relevant_units(
            us, cfg, task_label="summarization", target_lang=target_language,
        )

    chunks = _build_chunks(
        payload.get("content", ""), chunk_word_budget, units_filter=units_filter,
    )
    if not chunks:
        return {"response": ""}

    if len(chunks) == 1:
        return {"response": _phi_summarize(chunks[0], target_language, cfg)}

    if ctx is None or getattr(ctx, "db", None) is None or getattr(ctx, "job_id", None) is None:
        # Fallback: no fan-out possible; summarize chunks sequentially in-process
        # and merge. Keeps the handler usable in unit tests without a DB.
        partials = [_phi_summarize(c, target_language, cfg) for c in chunks]
        return {"response": _phi_merge(partials, target_language, cfg)}

    pending: Dict[str, int] = {}
    results: Dict[str, Optional[str]] = {}
    retries: Dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        child_payload = {
            "content": chunk,
            "targetLanguage": target_language,
            "_chunk_idx": i,
        }
        if source_language is not None:
            child_payload["sourceLanguage"] = source_language
        child_id = ctx.db.enqueue_child_job(
            ctx.job_id,
            "summarize",
            payload=child_payload,
            agent_max_steps=1,
        )
        if child_id is None:
            return {"error": f"failed to enqueue child for chunk {i}"}
        pending[str(child_id)] = i
        results[str(i)] = None
        retries[str(i)] = 0

    chunk_payload_template: Dict[str, Any] = {"targetLanguage": target_language}
    if source_language is not None:
        chunk_payload_template["sourceLanguage"] = source_language

    state = {
        "phase": "merging",
        "chunks_count": len(chunks),
        "pending": pending,
        "results": results,
        "retries": retries,
        "chunks": chunks,
        "targetLanguage": target_language,
        "chunk_field": "content",
        "chunk_payload_template": chunk_payload_template,
    }
    if source_language is not None:
        state["sourceLanguage"] = source_language

    return {
        "_sub_agent_pending_many": True,
        "_state": state,
        "pending_children": pending,
    }


def _phase_merge(state: Dict[str, Any], cfg: Dict[str, Any], ctx) -> Dict[str, Any]:
    n = int(state.get("chunks_count", 0))
    target_language = state.get("targetLanguage") or "en"
    source_language = state.get("sourceLanguage")

    results = state.get("results") or {}
    partials: List[str] = []
    for i in range(n):
        r = results.get(str(i))
        if isinstance(r, dict):
            partials.append(r.get("response") or "")
        elif isinstance(r, str):
            partials.append(r)
        else:
            partials.append("")

    failed_idx = state.get("failed_idx")
    if failed_idx is not None:
        return {
            "error": (
                f"chunk {failed_idx} failed after retries: "
                f"{state.get('failed_error') or 'unknown error'}"
            )
        }

    valid_partials = [p for p in partials if p]
    if not valid_partials:
        return {"error": "no chunk results available to merge"}

    merged = _phi_merge(valid_partials, target_language, cfg)

    chunk_word_budget = int(cfg.get("chunk_word_budget", 1500))
    factor = float(cfg.get("merge_recursion_factor", 2))
    if _word_count(merged) > chunk_word_budget * factor:
        return _phase_plan_or_leaf(
            {
                "content": merged,
                "sourceLanguage": source_language or target_language,
                "targetLanguage": target_language,
            },
            cfg,
            ctx,
        )

    return {"response": merged}


@job_handler("summarize")
def summarize_text(payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None, ctx=None) -> Dict[str, Any]:
    """Reentrant handler. `state` is None on first invocation; populated by the
    dispatcher when the parent is woken after all children complete."""
    cfg = get_task_config("summarize")

    if state and state.get("phase") == "merging":
        return _phase_merge(state, cfg, ctx)

    return _phase_plan_or_leaf(payload, cfg, ctx)
