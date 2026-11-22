"""Text generation client for the Models llama-server.

`services.llama_server` resolves or starts the configured endpoint. The public
surface (`generate`, `chat`, `chat_with_tools`, `chat_stream`,
`get_llm_service`) is used by task handlers. Runtime sizing belongs to the
server; request-level sampling and the model expected by each task remain here.
"""

import json
import logging
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

import re

from lib.llm.config import get_inference_sampling
from services import llama_server, managed_engine
from lib.telemetry import trace_post, inference_span, inference_response
from lib.execution.inference_capture import current_capture, CaptureError, observe_inference, engine_properties, safe_model

logger = logging.getLogger(__name__)


class _NoCaptureRedirect(urllib.request.HTTPRedirectHandler):
    # urllib invokes this hook when the captured endpoint redirects.
    def redirect_request(self, *_args, **_kwargs):  # noqa: V105
        return None


_capture_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoCaptureRedirect(),
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Generation is slow by nature and a busy server queues requests behind each
# other, so this is a "something is wrong" timeout, not a deadline.
REQUEST_TIMEOUT_S = float(os.environ.get("LLAMA_TIMEOUT", "600"))

# llama-server compiles a decoding grammar from the tool parameter JSON
# Schemas. A bounded string (`maxLength`) becomes a repeated-character rule,
# and above roughly 1800 the generated grammar is rejected at sampler init
# ("Failed to initialize samplers: failed to parse grammar"). Tool schemas
# legitimately declare large maxima (multi-thousand-char prompts, 1 MB file
# bodies) that trip this, so the grammar copy drops oversized bounds.
# Argument size stays bounded by `max_tokens`, and the Backend validates the
# resulting payload, so the bound is not load-bearing for correctness.
_MAX_GRAMMAR_STRING_LENGTH = 1000


def sanitize_tools_for_grammar(tools: List[dict]) -> List[dict]:
    """Return a copy of `tools` whose schemas llama-server can compile.

    Only `maxLength` values above `_MAX_GRAMMAR_STRING_LENGTH` are removed;
    everything else is preserved. The caller's definitions are not mutated.
    """
    sanitized = json.loads(json.dumps(tools))

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            max_length = node.get("maxLength")
            if (
                isinstance(max_length, int)
                and max_length > _MAX_GRAMMAR_STRING_LENGTH
            ):
                node.pop("maxLength", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(sanitized)
    return sanitized


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from a model response."""
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


@trace_post
def _post(url: str, payload: Dict[str, Any], stream: bool = False):
    """POST JSON to the engine. Returns the parsed reply, or the live response
    object when `stream` is set so the caller can read it as it arrives."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    capture = current_capture()
    exchange = capture.begin(url, data, stream) if capture else None
    resp = _open_inference_request(req, exchange)
    if stream:
        return resp
    try:
        with resp:
            raw = exchange.read(resp, resp.status) if exchange else resp.read()
    except CaptureError:
        raise
    except Exception:
        if exchange:
            exchange.finish("transport_error")
            raise CaptureError("Inference response transport failed") from None
        raise
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        if exchange:
            exchange.finish("invalid_response")
            raise CaptureError("Inference returned an invalid JSON response") from None
        if isinstance(e, UnicodeDecodeError):
            raise
        text = raw.decode("utf-8")
        raise RuntimeError(f"llama-server returned something that isn't JSON: {text[:200]}") from e


class _EndpointUnavailable(CaptureError):
    """The inference server does not provide the requested endpoint."""


def _http_error_type(status):
    return _EndpointUnavailable if status in (404, 405) else CaptureError


def _open_inference_request(req, exchange):
    try:
        return (_capture_opener.open(req, timeout=REQUEST_TIMEOUT_S) if exchange
                else urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S))
    except urllib.error.HTTPError as e:
        if exchange:
            _raise_captured_http_error(exchange, e)
        if e.code in (404, 405):
            raise _EndpointUnavailable(f"Inference endpoint unavailable (HTTP {e.code})") from e
        raise RuntimeError(f"llama-server rejected the request: {_error_detail(e)}") from e
    except Exception as e:
        if exchange:
            exchange.finish("transport_error")
            raise CaptureError("Inference transport failed") from None
        if isinstance(e, OSError):
            raise RuntimeError(f"llama-server at {req.full_url} is not answering: {e}") from e
        raise


def _raise_captured_http_error(exchange, error):
    try:
        exchange.read(error, error.code)
    except CaptureError:
        raise
    except Exception:
        exchange.finish("transport_error")
        raise CaptureError("Inference error response transport failed") from None
    finally:
        error.close()
    raise _http_error_type(error.code)(f"Inference request failed (HTTP {error.code})") from None


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


_INVALID_LORA_SCALE = "Invalid LoRA scale"


class LLMService:
    """Client for one model served by the configured Models engine."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_batch: int,
        n_gpu_layers: int,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        lora_sha256: str | None = None,
    ):
        self.model_path = model_path
        self.model_id = os.path.splitext(os.path.basename(model_path))[0]
        self.n_ctx = n_ctx
        self.lora_path = lora_path
        if type(lora_scale) is bool:
            raise ValueError(_INVALID_LORA_SCALE)
        try:
            self.lora_scale = float(lora_scale)
        except (ValueError, TypeError):
            raise ValueError(_INVALID_LORA_SCALE) from None
        if not math.isfinite(self.lora_scale):
            raise ValueError(_INVALID_LORA_SCALE)
        self.lora_sha256 = lora_sha256

        self.sampling = get_inference_sampling(model_path)
        logger.info(
            "LLM sampling for %s: %s", os.path.basename(model_path), self.sampling
        )

        self.url = llama_server.ensure_server(model_path)

    def _prepare_request(self, body):
        data = llama_server.props(self.url, model_id=self.model_id)
        self._warn_on_mismatch(data)
        adapter = None
        if self.lora_path:
            adapter = llama_server.verified_lora_adapter(
                self.url, self.lora_path, self.model_id, self.lora_sha256,
            )
            adapter['scale'] = self.lora_scale
            body['lora'] = [{'id': adapter['id'], 'scale': self.lora_scale}]
        engine = engine_properties(data)
        instance = managed_engine.inspect(self.url, data)
        if engine or instance['status'] == 'verified_local':
            engine = engine or {'source': 'server_props', 'properties': {}}
            engine['localInstance'] = instance
        capture = current_capture()
        if capture is not None:
            capture.record_templates(data, engine)
        return {'engine': engine, 'adapter': adapter}, data

    def _post(self, path, body):
        identity, properties = self._prepare_request(body)
        try:
            return _post(self.url + path, body, identity=identity)
        finally:
            managed_engine.confirm(self.url, properties, identity)

    def _model_field(self) -> Dict[str, Any]:
        return {"model": self.model_id}

    def _warn_on_mismatch(self, data) -> None:
        """Say so when the engine isn't serving what this task asked for.

        Asking props for this model also selects it when the engine is a router.
        A legacy single-model server ignores the selector, so the mismatch
        warning remains useful there.
        """
        path = data.get("model_path") or data.get("model") or ""
        if not path and isinstance(data.get("default_generation_settings"), dict):
            path = data["default_generation_settings"].get("model") or ""
        served = os.path.splitext(safe_model(path) or "")[0]
        if served and served != self.model_id:
            logger.warning(
                "Documents engine serves %s, this task asked for %s.",
                served, self.model_id,
            )
        settings = data.get("default_generation_settings")
        served_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        if isinstance(served_ctx, int) and self.n_ctx and served_ctx < self.n_ctx:
            logger.warning(
                "Documents engine has a %d-token context, this task asked for %d. "
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
        body.update(self._model_field())
        body.update(self._sampling_kwargs({"temperature": temperature, "seed": seed}))
        if grammar is not None:
            body["grammar"] = grammar
        resp = self._post("/completion", body)
        text = (resp.get("content") or "").strip()
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
        except _EndpointUnavailable:
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
        sampling_overrides: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Chat completion. Returns the assistant message content.

        Thinking is suppressed by default through the model chat template and
        any <think> blocks are stripped. This works for Qwen3 and Qwen3.8,
        whose template rejects system messages placed after the user message.

        `response_format` follows the OpenAI convention
        ({"type": "json_object", "schema": {...}}): the server compiles the
        schema to a grammar, so decoding is token-constrained and the output
        is guaranteed to be a conforming JSON object. Note that the schema is
        NOT shown to the model — callers must describe the expected output in
        the prompt themselves.
        """
        body: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_prompt": True,
        }
        body.update(self._model_field())
        overrides = dict(sampling_overrides or {})
        if temperature is not None:
            overrides["temperature"] = temperature
        if seed is not None:
            overrides["seed"] = seed
        body.update(self._sampling_kwargs(overrides))
        if grammar is not None:
            body["grammar"] = grammar
        if response_format is not None:
            body["response_format"] = response_format
        template_kwargs = dict(chat_template_kwargs or {})
        if not allow_thinking:
            template_kwargs.setdefault("enable_thinking", False)
            template_kwargs.setdefault("preserve_thinking", False)
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        resp = self._post("/v1/chat/completions", body)
        text = _content_of(resp)
        return text if allow_thinking else strip_thinking(text)

    def chat_with_tools(
        self,
        messages: list,
        tools: List[dict],
        max_tokens: int = 1000,
        tool_choice: str = "auto",
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

        """
        body: Dict[str, Any] = {
            "messages": messages,
            "tools": sanitize_tools_for_grammar(tools),
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body.update(self._model_field())
        body.update(self._sampling_kwargs())
        resp = self._post("/v1/chat/completions", body)
        choices = resp.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return message

    def chat_stream(
        self,
        messages: list,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
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
            "stream_options": {"include_usage": True},
            "cache_prompt": True,
        }
        body.update(self._model_field())
        body.update(self._sampling_kwargs())
        identity, properties = self._prepare_request(body)
        with inference_span(body) as current_span, observe_inference(body, identity) as observation:
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
                            inference_response(current_span, chunk, observation)
                            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                            piece = delta.get("content") or ""
                        except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
                            piece = ""
                        if piece:
                            yield piece

            finally:
                managed_engine.confirm(self.url, properties, identity)


def _content_of(resp: Dict[str, Any]) -> str:
    """The assistant text of a non-streamed reply, whichever shape it came in."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or choice.get("text") or "").strip()


# Cache of clients keyed by their complete requested configuration
_llm_cache: dict[tuple, LLMService] = {}


def get_llm_service(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    n_gpu_layers: int,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
    lora_sha256: str | None = None,
) -> LLMService:
    """Get or create a cached client for the given model (and optional LoRA)."""
    if lora_sha256 is not None and not isinstance(lora_sha256, str):
        raise ValueError("Invalid LoRA checksum")
    key = (model_path, n_ctx, n_threads, n_batch, n_gpu_layers, lora_path, lora_scale, lora_sha256)
    if key not in _llm_cache:
        _llm_cache[key] = LLMService(
            model_path, n_ctx, n_threads, n_batch, n_gpu_layers, lora_path, lora_scale, lora_sha256
        )
    return _llm_cache[key]


def reset_cache() -> None:
    """Forget the requested configurations; server properties and IDs are read per request."""
    _llm_cache.clear()
