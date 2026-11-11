"""The RAG chain: retrieve from Qdrant, number the chunks, ask the model, time and count it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import PROMPTS_DIR, Provider
from .index import Index


@dataclass(frozen=True)
class PromptSpec:
    id: str
    description: str
    system: str
    human: str

    def template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([("system", self.system), ("human", self.human)])


def load_prompt(prompt_id: str) -> PromptSpec:
    path = PROMPTS_DIR / f"{prompt_id}.yaml"
    if not path.exists():
        names = sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
        raise SystemExit(f"no prompt {prompt_id!r} in {PROMPTS_DIR}; available: {', '.join(names)}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptSpec(spec["id"], spec.get("description", "").strip(), spec["system"], spec["human"])


def format_sources(docs: list[Document]) -> str:
    """[1] PSD2 Article 74: Payer's liability ...\n<chunk text>, one block per retrieved chunk."""
    return "\n\n".join(f"[{i}] {d.metadata['header']}\n{d.metadata['body']}" for i, d in enumerate(docs, 1))


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # article_id, chunk, score, per number
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @property
    def retrieved_ids(self) -> list[str]:
        return [s["article_id"] for s in self.sources]


class RagPipeline:
    def __init__(self, index: Index, llm: BaseChatModel, prompt: PromptSpec, provider: Provider, k: int | None = None):
        self.index = index
        self.prompt = prompt
        self.provider = provider
        self.k = k or index.settings.k
        # LCEL: the prompt template piped into the chat model.
        self.chain = prompt.template() | llm

    def ask(self, question: str) -> RagAnswer:
        t0 = time.perf_counter()
        hits = self.index.search(question, k=self.k)
        t1 = time.perf_counter()
        docs = [d for d, _ in hits]
        message = self.chain.invoke({"context": format_sources(docs), "question": question})
        t2 = time.perf_counter()
        usage = message.usage_metadata or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return RagAnswer(
            question=question,
            answer=message.text.strip(),
            sources=[
                {"n": i, "article_id": d.metadata["article_id"], "chunk": d.metadata["chunk"], "score": round(float(s), 4)}
                for i, (d, s) in enumerate(hits, 1)
            ],
            retrieval_ms=(t1 - t0) * 1000,
            generation_ms=(t2 - t1) * 1000,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self.provider.cost(tokens_in, tokens_out),
        )


def prompt_path(prompt_id: str) -> Path:
    return PROMPTS_DIR / f"{prompt_id}.yaml"
