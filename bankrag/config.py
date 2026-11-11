"""Paths, sources, retrieval settings, providers and prices, all in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
CACHE_DIR = ROOT / ".cache"
RAW_DIR = CACHE_DIR / "raw"
PROMPTS_DIR = ROOT / "prompts"
EVALS_DIR = ROOT / "evals"
QUESTIONS_FILE = EVALS_DIR / "questions.yaml"
# "main" is the gated eval set; "holdout" was written after the shipped prompt was frozen;
# "plain" asks main-set facts in everyday words, without the directives' own wording.
QUESTION_SETS = {"main": QUESTIONS_FILE, "holdout": EVALS_DIR / "holdout.yaml", "plain": EVALS_DIR / "plain.yaml"}
BASELINES_FILE = EVALS_DIR / "baselines.json"
GATES_FILE = EVALS_DIR / "gates.yaml"
RESULTS_DIR = ROOT / "results"

# Machine-specific settings (paths to llama.cpp, an API key) live in an untracked .env file.
load_dotenv(ROOT / ".env")

DEFAULT_PROMPT = "grounded-v2"


@dataclass(frozen=True)
class Source:
    key: str  # short name used in citations, e.g. "PSD2"
    celex: str  # the act's CELEX number in EUR-Lex
    title: str

    @property
    def download_url(self) -> str:
        # EUR-Lex itself answers scripts with a bot check; the Publications Office's
        # CELLAR endpoint serves the same Official Journal text as XHTML.
        return f"http://publications.europa.eu/resource/celex/{self.celex}"

    @property
    def human_url(self) -> str:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{self.celex}"


SOURCES = [
    Source("PSD2", "32015L2366", "Directive (EU) 2015/2366 on payment services in the internal market (PSD2)"),
    Source("DGSD", "32014L0049", "Directive 2014/49/EU on deposit guarantee schemes (DGSD)"),
    Source("GDPR", "32016R0679", "Regulation (EU) 2016/679, the General Data Protection Regulation (GDPR)"),
]


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # dense, 384 dimensions, runs on the CPU
    sparse_model: str = "Qdrant/bm25"  # keyword scores, for hybrid search
    mode: str = "hybrid"  # dense | sparse | hybrid
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150
    chunk_header: bool = True  # prefix each chunk with "PSD2 Article 74: <title>"
    k: int = 5  # chunks handed to the model

    def with_(self, **changes) -> "RetrievalSettings":
        return replace(self, **changes)


RETRIEVAL = RetrievalSettings()


@dataclass(frozen=True)
class Provider:
    key: str
    kind: str  # "llamacpp", "anthropic" or "transformers"
    model: str
    label: str
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    adapter: bool = False  # transformers only: load the LoRA adapter from LORA_ADAPTER_DIR

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_mtok_in + tokens_out * self.usd_per_mtok_out) / 1_000_000


PROVIDERS = {
    # SmolLM3 3B on llama.cpp, on this PC's GPU. No API cost; electricity isn't counted.
    "local": Provider("local", "llamacpp", "SmolLM3-3B-Q4_K_M", "SmolLM3 3B (Q4_K_M), llama.cpp, RTX 3060 Ti"),
    # Anthropic list prices for Claude Haiku 5.5 (prompts up to 100K tokens), November 2026.
    "claude-haiku": Provider(
        "claude-haiku", "anthropic", "claude-haiku-5-5", "Claude Haiku 5.5, Anthropic API",
        usd_per_mtok_in=0.10, usd_per_mtok_out=0.50,
    ),
    # The LoRA experiment (notebooks/lora_experiment.ipynb): Qwen3 0.6B through transformers,
    # without and with the adapter. Needs `uv sync --group finetune`.
    "qwen3-0.6b": Provider("qwen3-0.6b", "transformers", "Qwen/Qwen3-0.6B", "Qwen3 0.6B (bf16), transformers, RTX 3060 Ti"),
    "qwen3-0.6b-lora": Provider(
        "qwen3-0.6b-lora", "transformers", "Qwen/Qwen3-0.6B", "Qwen3 0.6B + LoRA adapter, transformers, RTX 3060 Ti",
        adapter=True,
    ),
}

# Where the notebook saves the trained adapter (not committed; the notebook rebuilds it).
LORA_ADAPTER_DIR = CACHE_DIR / "lora" / "qwen3-0.6b-contract"

# Generation settings shared by every provider.
MAX_OUTPUT_TOKENS = 400
SEED = 42


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None
