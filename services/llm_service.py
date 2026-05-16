import os
from typing import Any, Iterator, List, Optional

from llama_cpp import Llama


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

        self.llm = Llama(**kwargs)

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Simple completion. Returns the generated text."""
        response = self.llm(prompt, max_tokens=max_tokens, echo=False)
        return response["choices"][0]["text"].strip()

    def chat(self, messages: list, max_tokens: int = 1000) -> str:
        """Chat completion. Returns the assistant message content."""
        resp = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens
        )
        choice = resp["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"].strip()
        return choice.get("text", "").strip()

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
        )
        choice = resp["choices"][0]
        return choice.get("message") or {}

    def chat_stream(self, messages: list, max_tokens: int = 1000) -> Iterator[str]:
        """Chat completion as a token stream. Yields content chunks as the
        model produces them. The caller is responsible for accumulating the
        full reply if they need it. Each yielded value is a (possibly empty)
        string; consumers should ignore empties."""
        stream = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, stream=True
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
