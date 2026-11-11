"""LangChain chat models for each provider."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from .config import MAX_OUTPUT_TOKENS, PROVIDERS, SEED, Provider


def chat_model(provider: Provider, base_url: str | None = None) -> BaseChatModel:
    if provider.kind == "llamacpp":
        from langchain_openai import ChatOpenAI

        # llama-server speaks the OpenAI chat API. Greedy decoding and a fixed seed make the
        # local runs repeatable; SmolLM3's reasoning mode is switched off through its template.
        return ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-needed",
            model=os.environ.get("BANKRAG_LOCAL_MODEL", provider.model),
            temperature=0.0,
            seed=SEED,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=120,
            max_retries=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    if provider.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY (in the environment or in .env) to use Claude")
        # Claude Haiku 5.5 takes no sampling parameters, so its runs are not exactly repeatable.
        # Effort is set low: these are one-paragraph lookups, not reasoning tasks.
        return ChatAnthropic(
            model=provider.model,
            max_tokens=1024,
            output_config={"effort": os.environ.get("BANKRAG_CLAUDE_EFFORT", "low")},
            max_retries=3,
            default_request_timeout=120,
        )
    if provider.kind == "transformers":
        from . import hf_chat
        from .config import LORA_ADAPTER_DIR

        if provider.adapter and not LORA_ADAPTER_DIR.exists():
            raise RuntimeError(f"no LoRA adapter at {LORA_ADAPTER_DIR}: run notebooks/lora_experiment.ipynb first")
        return hf_chat.load(provider.model, adapter=str(LORA_ADAPTER_DIR) if provider.adapter else None)
    raise ValueError(f"unknown provider kind {provider.kind!r}")


def get_provider(key: str) -> Provider:
    try:
        return PROVIDERS[key]
    except KeyError:
        raise SystemExit(f"unknown provider {key!r}; choose from {', '.join(PROVIDERS)}") from None
