import ctypes
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import logging

import llama_cpp
from llama_cpp import Llama, LlamaGrammar

from lib.llm.config import get_inference_sampling

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# `verbose=False` no silencia los avisos que llama.cpp emite por su propio log
# callback nativo (p. ej. "n_ctx_seq (8192) < n_ctx_train ..."). Registramos un
# callback que descarta todo lo que esté por debajo de ERROR (nivel ggml >= 4),
# preservando los errores reales. Guardamos la referencia para que ctypes no lo
# recolecte, y lo instalamos una sola vez.
_GGML_LOG_LEVEL_ERROR = 4
_log_cb = None


def _silence_llama_below_error():
    global _log_cb
    if _log_cb is not None:
        return

    @llama_cpp.llama_log_callback
    def _cb(level, text, user_data):  # noqa: ARG001
        if level >= _GGML_LOG_LEVEL_ERROR:
            print(text.decode("utf-8", "replace"), end="", flush=True)

    _log_cb = _cb
    llama_cpp.llama_log_set(_log_cb, ctypes.c_void_p(0))


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from a model response."""
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


class LLMService:
    """Service wrapping a Llama model instance."""

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
        self.lora_path = lora_path
        self.lora_scale = lora_scale

        self.sampling = get_inference_sampling(model_path)
        logger.info(
            "LLM sampling for %s: %s", os.path.basename(model_path), self.sampling
        )

        kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if lora_path:
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(
                    f"LoRA adapter file not found: {lora_path}. "
                    "Place the adapter .gguf in the models directory or set lora_path to its absolute path."
                )
            kwargs["lora_path"] = lora_path
            kwargs["lora_scale"] = lora_scale

        _silence_llama_below_error()
        try:
            self.llm = Llama(**kwargs)
        except Exception as e:
            logger.error("llama_cpp failed to load model %s: %s", model_path, e)
            # If a GPU-backed load was attempted, retry on CPU (n_gpu_layers=0)
            if kwargs.get("n_gpu_layers", 0) != 0:
                logger.info("Retrying model load on CPU (n_gpu_layers=0) for %s", model_path)
                kwargs["n_gpu_layers"] = 0
                try:
                    self.llm = Llama(**kwargs)
                except Exception as e2:
                    logger.error("CPU fallback also failed for model %s: %s", model_path, e2)
                    raise RuntimeError(f"llama_cpp failed to load model {model_path}: {e2}") from e2
            else:
                raise RuntimeError(f"llama_cpp failed to load model {model_path}: {e}") from e

    def _sampling_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Map the resolved per-model sampling defaults to llama-cpp-python kwargs.

        Config keys use `repetition_penalty`; llama-cpp-python expects `repeat_penalty`.
        Any non-None value in `overrides` (e.g. a caller-supplied temperature) wins over
        the model default for that parameter.
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
        JSON. `temperature` and `seed` are forwarded when set; otherwise
        llama-cpp's defaults apply. Unless `allow_thinking` is set, any
        <think> blocks are stripped from the result.
        """
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "echo": False}
        kwargs.update(self._sampling_kwargs({"temperature": temperature, "seed": seed}))
        if grammar is not None:
            kwargs["grammar"] = LlamaGrammar.from_string(grammar, verbose=False)
        response = self.llm(prompt, **kwargs)
        text = response["choices"][0]["text"].strip()
        return text if allow_thinking else strip_thinking(text)

    def chat(
        self,
        messages: list,
        max_tokens: int = 1000,
        grammar: Optional[str] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        allow_thinking: bool = False,
    ) -> str:
        """Chat completion. Returns the assistant message content.

        Thinking is suppressed by default: a `/no_think` system message is
        appended (Qwen3 soft switch) and any <think> blocks are stripped, so
        the reasoning budget isn't spent inside `max_tokens`. Callers that
        want the model to reason (and handle stripping themselves) pass
        `allow_thinking=True`.

        `response_format` follows the OpenAI convention
        ({"type": "json_object", "schema": {...}}): llama-cpp compiles the
        schema to a grammar internally, so decoding is token-constrained and
        the output is guaranteed to be a conforming JSON object. Note that
        the schema is NOT shown to the model — callers must describe the
        expected output in the prompt themselves.
        """
        if not allow_thinking:
            messages = list(messages) + [{"role": "system", "content": "/no_think"}]
        kwargs: Dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        kwargs.update(self._sampling_kwargs({"temperature": temperature, "seed": seed}))
        if grammar is not None:
            kwargs["grammar"] = LlamaGrammar.from_string(grammar, verbose=False)
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self.llm.create_chat_completion(**kwargs)
        choice = resp["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            text = choice["message"]["content"].strip()
        else:
            text = choice.get("text", "").strip()
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
        stream yet."""
        resp = self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            **self._sampling_kwargs(),
        )
        choice = resp["choices"][0]
        return choice.get("message") or {}

    def chat_stream(self, messages: list, max_tokens: int = 1000) -> Iterator[str]:
        """Chat completion as a token stream. Yields content chunks as the
        model produces them. The caller is responsible for accumulating the
        full reply if they need it. Each yielded value is a (possibly empty)
        string; consumers should ignore empties."""
        stream = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, stream=True,
            **self._sampling_kwargs(),
        )
        for chunk in stream:
            try:
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                piece = ""
            if piece:
                yield piece


# Cache of LLM instances keyed by (model_path, lora_path, lora_scale)
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
    """Get or create a cached LLM service for the given model (and optional LoRA)."""
    key = (model_path, lora_path, lora_scale)
    if key not in _llm_cache:
        _llm_cache[key] = LLMService(
            model_path, n_ctx, n_threads, n_batch, n_gpu_layers, lora_path, lora_scale
        )
    return _llm_cache[key]
