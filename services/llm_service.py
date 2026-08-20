"""Text generation against the shared llama-server.

This used to load the .gguf into this process with llama-cpp-python. It no
longer does: the model is loaded once, by one llama-server, and everything on
the machine — every worker plus the embedded browser — generates through it. See
`services.llama_server` for how that server is found or started.

The public surface is unchanged on purpose (`generate`, `chat`,
`chat_with_tools`, `chat_stream`, `get_llm_service`), so the fifteen-odd task
modules that call it did not have to change. What changed underneath:

  - `n_ctx`, `n_threads`, `n_batch` and `n_gpu_layers` no longer decide anything
    here — the server was started with its own. They are kept in the signature
    because callers pass them from the task config, and `n_ctx` is still worth
    comparing against what the server actually serves.
  - LoRA adapters are not applied. The server loads what it was told to load; a
    task asking for one gets a warning and the base model.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

import re

from lib.llm.config import get_inference_sampling
from services import llama_server

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Generation is slow by nature and a busy server queues requests behind each
# other, so this is a "something is wrong" timeout, not a deadline.
REQUEST_TIMEOUT_S = float(os.environ.get("LLAMA_TIMEOUT", "600"))


def _begin_inference(
    name: str,
    request: Dict[str, Any],
    trace_metadata: Optional[Dict[str, Any]] = None,
):
    from lib.execution import get_active_emitter
    emitter = get_active_emitter()
    return (
        (emitter, emitter.start_inference(name, request, trace_metadata))
        if emitter else (None, None)
    )


def _response_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage") if isinstance(response, dict) else None
    timings = response.get("timings") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    timings = timings if isinstance(timings, dict) else {}

    def integer(*values):
        for value in values:
            if isinstance(value, (int, float)) and value >= 0:
                return int(round(value))
        return "unknown"

    return {
        "promptTokens": integer(
            usage.get("prompt_tokens"),
            timings.get("prompt_n"),
            response.get("tokens_evaluated") if isinstance(response, dict) else None,
        ),
        "generatedTokens": integer(
            usage.get("completion_tokens"),
            timings.get("predicted_n"),
            response.get("tokens_predicted") if isinstance(response, dict) else None,
        ),
        "timeToFirstTokenMs": integer(
            timings.get("time_to_first_token_ms"),
            timings.get("ttft_ms"),
        ),
    }


def _finish_inference(
    emitter,
    handle,
    response: Any,
    *,
    outcome: str,
    raw_response: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    if not emitter or not handle:
        return
    emitter.finish_inference(
        handle,
        response,
        outcome=outcome,
        status="failed" if error else "succeeded",
        error=error,
        reason=reason,
        metrics=_response_metrics(raw_response or {}),
        raw_response=raw_response,
    )


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from a model response."""
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


def _post(url: str, payload: Dict[str, Any], stream: bool = False):
    """POST JSON to the engine. Returns the parsed reply, or the live response
    object when `stream` is set so the caller can read it as it arrives."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"llama-server rejected the request: {_error_detail(e)}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"llama-server at {url} is not answering: {e}") from e
    if stream:
        return resp
    with resp:
        raw = resp.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"llama-server returned something that isn't JSON: {raw[:200]}") from e


def _error_detail(e: urllib.error.HTTPError) -> str:
    """The message llama-server puts in the body of a 4xx, which says what is
    actually wrong (bad grammar, context overflow) far better than the status."""
    try:
        body = json.loads(e.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return f"HTTP {e.code}"
    error = body.get("error")
    if isinstance(error, dict):
        return f"HTTP {e.code}: {error.get('message') or error}"
    return f"HTTP {e.code}: {error or body}"


class LLMService:
    """Client for one model served by the shared engine."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_batch: int,
        n_gpu_layers: int,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.lora_path = lora_path
        self.lora_scale = lora_scale

        self.sampling = get_inference_sampling(model_path)
        logger.info(
            "LLM sampling for %s: %s", os.path.basename(model_path), self.sampling
        )

        self.url = llama_server.ensure_server(model_path)
        self._warn_on_mismatch()

        # The adapter is applied per request, citing the id the server gave it
        # when it loaded it (`--lora`). Resolving it here means a task whose
        # adapter never made it into the engine fails when its client is built,
        # loudly — before it produces a single answer that looks fine-tuned and
        # is really the base model.
        self._lora_id = None
        if lora_path:
            self._lora_id = llama_server.lora_adapter_id(self.url, lora_path)
            if self._lora_id is None:
                raise RuntimeError(
                    f"LoRA {os.path.basename(lora_path)} is not loaded by the "
                    f"engine at {self.url}. Adapters attach at startup: restart "
                    "it so the deployment reaches the command line."
                )
            logger.info(
                "LLM using LoRA %s (id=%s, scale=%s)",
                os.path.basename(lora_path), self._lora_id, self.lora_scale,
            )

    def _lora_field(self) -> Dict[str, Any]:
        """The `lora` field of a request, or nothing when there's no adapter.

        Sent on every request rather than applied once on the server: one engine
        serves every task, and the next request may belong to a task with a
        different adapter — or with none.
        """
        if self._lora_id is None:
            return {}
        return {"lora": [{"id": self._lora_id, "scale": self.lora_scale}]}

    def _warn_on_mismatch(self) -> None:
        """Say so when the engine isn't serving what this task asked for.

        Not an error: sharing one engine means whoever started it chose the
        model, and a task that wanted another one still gets an answer — just
        not from the model its config names.
        """
        served = llama_server.loaded_model(self.url)
        wanted = os.path.basename(self.model_path)
        if served and wanted and served != wanted:
            logger.warning(
                "Shared engine serves %s, this task asked for %s. Using %s.",
                served, wanted, served,
            )
        data = llama_server.props(self.url)
        settings = data.get("default_generation_settings")
        served_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        if isinstance(served_ctx, int) and self.n_ctx and served_ctx < self.n_ctx:
            logger.warning(
                "Shared engine has a %d-token context, this task asked for %d. "
                "Long prompts will be truncated.",
                served_ctx, self.n_ctx,
            )

    def _sampling_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Map the resolved per-model sampling defaults to llama-server fields.

        Config keys use `repetition_penalty`; the server expects
        `repeat_penalty`. Any non-None value in `overrides` (e.g. a
        caller-supplied temperature) wins over the model default.
        """
        s = self.sampling
        kwargs: Dict[str, Any] = {
            "temperature": s.get("temperature"),
            "top_p": s.get("top_p"),
            "top_k": s.get("top_k"),
            "min_p": s.get("min_p"),
            "repeat_penalty": s.get("repetition_penalty"),
        }
        if "presence_penalty" in s:
            kwargs["presence_penalty"] = s["presence_penalty"]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if overrides:
            kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        grammar: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        allow_thinking: bool = False,
    ) -> str:
        """Simple completion. Returns the generated text.

        When `grammar` is provided (a GBNF string), the output is constrained
        to match it — used by structured-extraction tasks to guarantee valid
        JSON. `temperature` and `seed` are forwarded when set; otherwise the
        model defaults apply. Unless `allow_thinking` is set, any <think>
        blocks are stripped from the result.
        """
        body: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            # Reusing the KV cache of a shared prefix is the difference between
            # answering now and reprocessing thousands of tokens; tasks that
            # loop over one document hit this on every call.
            "cache_prompt": True,
        }
        body.update(self._sampling_kwargs({"temperature": temperature, "seed": seed}))
        body.update(self._lora_field())
        if grammar is not None:
            body["grammar"] = grammar
        emitter, trace = _begin_inference("generate", body)
        try:
            resp = _post(f"{self.url}/completion", body)
        except Exception as error:
            _finish_inference(
                emitter, trace, {}, outcome="invalid", error=str(error),
            )
            raise
        text = (resp.get("content") or "").strip()
        _finish_inference(
            emitter,
            trace,
            text,
            outcome="final_text" if text else "invalid",
            raw_response=resp,
            reason=None if text else "empty_model_response",
        )
        return text if allow_thinking else strip_thinking(text)

    def ask(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: Optional[float] = None,
    ) -> str:
        """One instruction in, the answer out — no conversation to carry.

        Through `chat`, so the model's own template is rendered and the thinking
        is suppressed: a raw completion gives an instruct model no way to tell an
        instruction from text to continue, and it answers with a heading, repeats
        the block, or spills its reasoning into the output. Tasks that constrain
        the answer with a grammar don't need this and call `generate` directly.

        The fallback keeps an engine without the chat endpoint answering.
        """
        try:
            return self.chat([{"role": "user", "content": prompt}],
                             max_tokens=max_tokens, temperature=temperature)
        except Exception:  # noqa: BLE001 — engine without /v1/chat/completions
            return self.generate(prompt, max_tokens=max_tokens,
                                 temperature=temperature)

    def chat(
        self,
        messages: list,
        max_tokens: int = 1000,
        grammar: Optional[str] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        allow_thinking: bool = False,
        inference_name: str = "chat",
        trace_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Chat completion. Returns the assistant message content.

        Thinking is suppressed by default: a `/no_think` system message is
        appended (Qwen3 soft switch) and any <think> blocks are stripped, so
        the reasoning budget isn't spent inside `max_tokens`. Callers that
        want the model to reason (and handle stripping themselves) pass
        `allow_thinking=True`.

        `response_format` follows the OpenAI convention
        ({"type": "json_object", "schema": {...}}): the server compiles the
        schema to a grammar, so decoding is token-constrained and the output
        is guaranteed to be a conforming JSON object. Note that the schema is
        NOT shown to the model — callers must describe the expected output in
        the prompt themselves.
        """
        if not allow_thinking:
            messages = list(messages) + [{"role": "system", "content": "/no_think"}]
        body: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_prompt": True,
        }
        body.update(self._sampling_kwargs({"temperature": temperature, "seed": seed}))
        body.update(self._lora_field())
        if grammar is not None:
            body["grammar"] = grammar
        if response_format is not None:
            body["response_format"] = response_format
        emitter, trace = _begin_inference(inference_name, body, trace_metadata)
        try:
            resp = _post(f"{self.url}/v1/chat/completions", body)
        except Exception as error:
            _finish_inference(
                emitter, trace, {}, outcome="invalid", error=str(error),
            )
            raise
        text = _content_of(resp)
        _finish_inference(
            emitter,
            trace,
            text,
            outcome="final_text" if text else "invalid",
            raw_response=resp,
            reason=None if text else "empty_model_response",
        )
        return text if allow_thinking else strip_thinking(text)

    def chat_with_tools(
        self,
        messages: list,
        tools: List[dict],
        max_tokens: int = 1000,
        tool_choice: str = "auto",
        inference_name: str = "chat_with_tools",
        trace_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Chat completion with function/tool calling enabled.

        Returns the full first choice's `message` dict so the caller can
        inspect either `content` (plain reply) or `tool_calls` (a list of
        functions the model wants invoked). The caller is responsible for
        executing the tools and feeding the results back in a follow-up call.

        Non-streaming on purpose: the model decides whether to call a tool
        before producing user-visible text, so there's nothing useful to
        stream yet.

        Needs a server started with --jinja; without it llama-server has no
        chat template to render the tool calls with and answers 500.

        `inference_name` and `trace_metadata` distinguish semantic follow-up
        operations, such as a bounded output repair, without changing the
        request sent to the model.
        """
        body: Dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body.update(self._sampling_kwargs())
        body.update(self._lora_field())
        emitter, trace = _begin_inference(
            inference_name,
            body,
            trace_metadata,
        )
        try:
            resp = _post(f"{self.url}/v1/chat/completions", body)
        except Exception as error:
            _finish_inference(
                emitter, trace, {}, outcome="invalid", error=str(error),
            )
            raise
        choices = resp.get("choices") or [{}]
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        has_inline_tool_call = isinstance(content, str) and "<tool_call>" in content
        outcome = (
            "tool_requests"
            if tool_calls or has_inline_tool_call
            else ("final_text" if content else "invalid")
        )
        _finish_inference(
            emitter,
            trace,
            {"content": content, "tool_calls": tool_calls},
            outcome=outcome,
            raw_response=resp,
            reason="empty_model_response" if outcome == "invalid" else None,
        )
        return message

    def chat_stream(self, messages: list, max_tokens: int = 1000) -> Iterator[str]:
        """Chat completion as a token stream. Yields content chunks as the
        model produces them. The caller is responsible for accumulating the
        full reply if they need it. Each yielded value is a (possibly empty)
        string; consumers should ignore empties.

        A reasoning model's thinking no longer arrives mixed into the text: the
        server renders the chat template itself and puts it in a separate
        `reasoning_content` field, which this ignores. It still spends
        `max_tokens` though — a budget too small can be eaten whole by the
        thinking, and the stream then yields nothing at all.
        """
        body: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        body.update(self._sampling_kwargs())
        body.update(self._lora_field())
        emitter, trace = _begin_inference("chat_stream", body)
        parts: List[str] = []
        last_chunk: Dict[str, Any] = {}
        try:
            resp = _post(f"{self.url}/v1/chat/completions", body, stream=True)
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        last_chunk = chunk
                        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        piece = delta.get("content") or ""
                    except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
                        piece = ""
                    if piece:
                        parts.append(piece)
                        yield piece
        except Exception as error:
            _finish_inference(
                emitter, trace, "".join(parts), outcome="invalid", error=str(error),
            )
            raise
        _finish_inference(
            emitter,
            trace,
            "".join(parts),
            outcome="final_text" if parts else "invalid",
            raw_response=last_chunk,
            reason=None if parts else "empty_model_response",
        )


def _content_of(resp: Dict[str, Any]) -> str:
    """The assistant text of a non-streamed reply, whichever shape it came in."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or choice.get("text") or "").strip()


# Cache of clients keyed by (model_path, lora_path, lora_scale)
_llm_cache: dict[tuple, LLMService] = {}


def get_llm_service(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    n_gpu_layers: int,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
) -> LLMService:
    """Get or create a cached client for the given model (and optional LoRA)."""
    key = (model_path, lora_path, lora_scale)
    if key not in _llm_cache:
        _llm_cache[key] = LLMService(
            model_path, n_ctx, n_threads, n_batch, n_gpu_layers, lora_path, lora_scale
        )
    return _llm_cache[key]


def reset_cache() -> None:
    """Forget every cached client. Call it after restarting the engine.

    A client holds the engine URL and the id its adapter had in THAT server;
    both are meaningless once the server is replaced, and reusing them would
    send requests citing an adapter id that now belongs to another file — or to
    nothing at all.
    """
    _llm_cache.clear()
