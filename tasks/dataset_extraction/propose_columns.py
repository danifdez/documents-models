"""Worker handler for `dataset.propose-columns`.

Given excerpts of a handful of resources, ask the LLM to propose a list of
DatasetField objects ready to be used as a Dataset schema. Sister piece to
`dataset.extract-row`: the extraction worker fills cells, this one proposes
which cells to fill in the first place.

Payload (set by backend `DatasetController.proposeColumns` in T05):
    {
      "resources": [
        { "id": int, "title": str, "excerpt": str },  # excerpt = first ~2000 chars
        ...
      ]
    }

Result:
    {
      "columns": list[DatasetField] | None,   # null when error
      "error":   str | None
    }

The chosen DatasetField objects are guaranteed to carry a non-empty
`description`. Downstream extraction (T03) refuses to populate columns
without one, so the proposal is useless without it.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from agent.llm import get_llm_for_spec
from agent.types import ModelSpec
from services.llm_json import parse_json
from services.model_config import get_llm_defaults, get_task_config
from services.prompts import load_prompt
from utils.job_registry import job_handler

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_PROPOSE_PROMPT = load_prompt(_PROMPTS_DIR, "propose_columns.md")

logger = logging.getLogger(__name__)


_PROPOSE_GRAMMAR = r"""
root ::= "[" ws column ws ("," ws column ws)* "]"
column ::= "{" ws key-pair ws "," ws name-pair ws "," ws type-pair ws "," ws desc-pair (ws "," ws options-pair)? ws "}"
key-pair ::= "\"key\":" ws string
name-pair ::= "\"name\":" ws string
type-pair ::= "\"type\":" ws ("\"text\"" | "\"number\"" | "\"boolean\"" | "\"date\"" | "\"select\"")
desc-pair ::= "\"description\":" ws string
options-pair ::= "\"options\":" ws "[" ws string (ws "," ws string)* ws "]"
ws ::= ([ \t\n] ws)?
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
""".strip() + "\n"


def _build_prompt(resources: List[Dict[str, Any]]) -> str:
    excerpts = []
    for r in resources:
        title = r.get("title") or f"resource {r.get('id')}"
        excerpt = (r.get("excerpt") or "").strip()
        excerpts.append(f"--- Document: {title} ---\n{excerpt}")
    joined = "\n\n".join(excerpts)

    return _PROPOSE_PROMPT.format(joined=joined)


def _snake_case(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip()).strip("_").lower()
    if not s:
        return "column"
    if s[0].isdigit():
        s = "col_" + s
    return s


_VALID_TYPES = {"text", "number", "boolean", "date", "select"}


def _normalize_columns(raw: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = _snake_case(str(item.get("key") or item.get("name") or ""))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        ftype = str(item.get("type") or "text").lower()
        if ftype not in _VALID_TYPES:
            ftype = "text"

        description = str(item.get("description") or "").strip()
        if not description:
            # Skip columns the model didn't bother to describe — they would be
            # useless to the extraction worker downstream.
            continue

        col: Dict[str, Any] = {
            "key": key,
            "name": str(item.get("name") or key).strip()[:200],
            "type": ftype,
            "description": description,
            "required": False,
        }
        if ftype == "select":
            options_raw = item.get("options")
            if isinstance(options_raw, list):
                opts = [str(o) for o in options_raw if isinstance(o, (str, int, float))]
                if opts:
                    col["options"] = opts
        out.append(col)
    return out


@job_handler("dataset.propose-columns")
def propose_dataset_columns(payload: Dict[str, Any]) -> Dict[str, Any]:
    resources: List[Dict[str, Any]] = payload.get("resources") or []
    if not resources:
        return {"columns": None, "error": "needs at least 1 resource"}

    usable = [r for r in resources if (r.get("excerpt") or "").strip()]
    if not usable:
        return {
            "columns": None,
            "error": "no readable content in any of the given resources",
        }

    cfg = get_task_config("dataset.propose-columns") or {}
    model_name = cfg.get("model") or get_llm_defaults().get("model")
    if not model_name:
        return {"columns": None, "error": "No model configured for dataset.propose-columns"}

    max_tokens = int(cfg.get("max_tokens", 1500))
    temperature = float(cfg.get("temperature", 0.2))
    seed = cfg.get("seed", 42)
    seed = int(seed) if seed is not None else None

    spec = ModelSpec(path=model_name)
    llm = get_llm_for_spec(spec)
    prompt = _build_prompt(usable[:3])

    try:
        raw = llm.generate(
            prompt,
            max_tokens=max_tokens,
            grammar=_PROPOSE_GRAMMAR,
            temperature=temperature,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("dataset.propose-columns: LLM call failed")
        return {"columns": None, "error": f"LLM call failed: {exc}"}

    parsed = parse_json(raw, default=None)
    if not isinstance(parsed, list):
        # Defensive retry without grammar — surfaces grammar bugs.
        try:
            raw_retry = llm.generate(
                prompt + "\n\nReturn ONLY the JSON array, nothing else.",
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            parsed = parse_json(raw_retry, default=None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dataset.propose-columns: retry failed")
            return {"columns": None, "error": f"LLM retry failed: {exc}"}

    if not isinstance(parsed, list):
        return {"columns": None, "error": "LLM did not return a JSON array"}

    columns = _normalize_columns(parsed)
    if not columns:
        return {"columns": None, "error": "LLM proposal had no usable columns"}

    return {"columns": columns, "error": None}
