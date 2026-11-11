"""A LangChain chat model over a Hugging Face transformers model, with an optional LoRA adapter.

It lets the LoRA experiment run through the same RagPipeline, graders and run files as the
llama.cpp and Claude providers. Needs the optional packages: uv sync --group finetune.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from .config import MAX_OUTPUT_TOKENS

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def chat_prompt_ids(tokenizer, system: str, user: str) -> list[int]:
    """Token ids of a system + user turn and the assistant header, thinking switched off.

    Training and answering both use this, so the adapter sees exactly the prompt it was
    trained on.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class HFChatModel(BaseChatModel):
    model: Any
    tokenizer: Any
    max_new_tokens: int = MAX_OUTPUT_TOKENS
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "hf-transformers"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        import torch

        roles = [_ROLES[m.type] for m in messages]
        if roles != ["system", "user"]:
            raise ValueError(f"expected a system and a user message, got {roles}")
        ids = chat_prompt_ids(self.tokenizer, messages[0].content, messages[1].content)
        input_ids = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy, so runs repeat
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        text = _THINK.sub("", self.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
        usage = {"input_tokens": len(ids), "output_tokens": int(new_tokens.shape[0]), "total_tokens": len(ids) + int(new_tokens.shape[0])}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=usage))])


def load(model_id: str, adapter: str | None = None, dtype: str = "bfloat16") -> HFChatModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to("cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return HFChatModel(model=model, tokenizer=tokenizer)
