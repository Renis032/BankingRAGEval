"""Training data for the LoRA experiment (notebooks/lora_experiment.ipynb).

The adapter is meant to learn the answer contract: cite the source as [n], and reply with the
exact refusal sentence when the sources don't hold the answer. The data is built so that it
can't leak the eval:

1. Only chunks from articles that no eval question (main, holdout or plain) has as its gold
   article are used, and administrative articles (entry into force, repeals, ...) are skipped.
2. SmolLM3 writes one question, a short answer and a supporting quote per chunk, through
   llama-server with a JSON schema. Code rejects a question unless the quote appears word for
   word in the chunk and the question stands on its own ("the passage", article numbers).
3. Each kept question becomes up to two examples, retrieved with the same hybrid search as
   the pipeline:
   - answer: the source chunk is among the top 5, so the target is the answer plus [n];
   - refuse: every chunk of the source article (and any chunk containing the quote) is
     removed from the results, so the target is the refusal sentence.
   The two halves differ only in whether the evidence is there, so the model has to check
   the sources rather than learn which questions get refused.

Contexts are stored as chunk references, not text, and rebuilt from the corpus when needed.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from . import corpus
from .config import DATA_DIR, QUESTION_SETS
from .grading import REFUSAL, load_questions, normalise
from .index import Index, chunk_articles
from .rag import format_sources, load_prompt

FINETUNE_DIR = DATA_DIR / "finetune"
GENERATED_FILE = FINETUNE_DIR / "generated_qa.jsonl"
EXAMPLES_FILE = FINETUNE_DIR / "examples.jsonl"

SEED = 7
MAX_CHUNKS_PER_ARTICLE = 3
MAX_CHUNKS = 360
REFUSAL_SHARE = 0.6  # share of kept questions that also get a refusal example
VAL_SHARE = 0.1  # share of articles held back for validation loss
TRAIN_PROMPT = "grounded-v2"

# Articles about how the act itself works rather than about banking rules.
ADMIN_TITLES = re.compile(
    r"entry into force|addressees|transposition|repeal|amendments? to|review|exercise of the delegation"
    r"|committee procedure|transitional|correlation|reports?$|evaluation|full harmonisation",
    re.I,
)
BAD_QUESTION = re.compile(r"\bpassage\b|\bthis (text|article|provision|section)\b|\bthe text\b|\barticle \d", re.I)
BAD_ANSWER = re.compile(r"\bpassage\b|\bthe text\b", re.I)  # answers may cite other articles, as the law does


class GeneratedQA(BaseModel):
    question: str = Field(description="A question a bank employee might ask that the passage answers")
    answer: str = Field(description="One or two sentences answering it, using only facts from the passage")
    quote: str = Field(description="A phrase of 5 to 30 words copied word for word from the passage that supports the answer")


GEN_SYSTEM = (
    "You write training questions for an assistant that answers questions about EU banking regulation. "
    "Given one passage, write a question that the passage answers, a short answer that uses only the passage, "
    "and an exact quote from the passage that supports the answer."
)
GEN_HUMAN = """Passage from {header}:
{body}

Rules:
- The question must make sense on its own. Do not write "the passage", "this article" or "the text", and do not mention article numbers. Name the act or topic instead, for example "Under PSD2, ..." or "Under the GDPR, ...".
- Ask about a concrete rule, deadline, amount, condition or obligation, not about definitions of words.
- The answer is one or two sentences in your own words, using only facts from the passage.
- The quote is copied word for word from the passage, 5 to 30 words long."""


def eval_gold_articles() -> set[str]:
    return {g for name in QUESTION_SETS for q in load_questions(name) for g in q.gold}


def eligible_chunks(seed: int = SEED) -> list[Document]:
    """Chunks the generator may see: no eval gold article, no administrative article, not tiny."""
    excluded = eval_gold_articles()
    by_article: dict[str, list[Document]] = {}
    for d in chunk_articles(corpus.load()):
        m = d.metadata
        if m["article_id"] in excluded or ADMIN_TITLES.search(m["title"]) or len(m["body"]) < 300:
            continue
        by_article.setdefault(m["article_id"], []).append(d)
    rng = random.Random(seed)
    picked = []
    for article_id in sorted(by_article):
        docs = by_article[article_id]
        picked += rng.sample(docs, min(MAX_CHUNKS_PER_ARTICLE, len(docs)))
    rng.shuffle(picked)
    return picked[:MAX_CHUNKS]


def check_qa(qa: dict, body: str) -> str | None:
    """Why a generated question is rejected, or None if it is kept."""
    q, a, quote = qa["question"].strip(), qa["answer"].strip(), qa["quote"].strip()
    if not q.endswith("?") or not 20 <= len(q) <= 250:
        return "question shape"
    if BAD_QUESTION.search(q):
        return "question refers to the passage or an article number"
    if not 20 <= len(a) <= 500:
        return "answer length"
    if BAD_ANSWER.search(a):
        return "answer refers to the passage"
    if len(quote.split()) < 5:
        return "quote too short"
    if normalise(quote).strip(" .;,") not in normalise(body):
        return "quote not in the passage"
    return None


def generate(llm, chunks: list[Document], path: Path = GENERATED_FILE, progress: bool = True) -> list[dict]:
    """Ask the generator model for one question per chunk; keep every result with its verdict."""
    structured = llm.with_structured_output(GeneratedQA, method="json_schema")
    rows = []
    for i, d in enumerate(chunks, 1):
        m = d.metadata
        try:
            qa = structured.invoke([("system", GEN_SYSTEM), ("human", GEN_HUMAN.format(header=m["header"], body=m["body"]))])
            qa = qa.model_dump()
            reason = check_qa(qa, m["body"])
        except Exception as e:  # malformed output from the generator counts as a rejection
            qa, reason = {"question": "", "answer": "", "quote": ""}, f"generation failed: {type(e).__name__}"
        rows.append({"article_id": m["article_id"], "chunk": m["chunk"], **qa, "rejected": reason})
        if progress and i % 25 == 0:
            kept = sum(1 for r in rows if not r["rejected"])
            print(f"  {i}/{len(chunks)} chunks, {kept} questions kept")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def build_examples(index: Index, generated: list[dict], seed: int = SEED, path: Path = EXAMPLES_FILE) -> list[dict]:
    """Turn kept questions into answer and refusal examples with retrieved contexts."""
    rng = random.Random(seed)
    kept = [g for g in generated if not g["rejected"]]
    seen, unique = set(), []
    for g in kept:  # one example per distinct question
        key = normalise(g["question"])
        if key not in seen:
            seen.add(key)
            unique.append(g)
    articles = sorted({g["article_id"] for g in unique})
    rng.shuffle(articles)
    val_articles = set(articles[: max(1, round(len(articles) * VAL_SHARE))])

    examples = []
    for g in unique:
        split = "val" if g["article_id"] in val_articles else "train"
        source = (g["article_id"], g["chunk"])
        hits = [d for d, _ in index.search(g["question"], k=15)]
        top5 = hits[:5]
        refs = [(d.metadata["article_id"], d.metadata["chunk"]) for d in top5]
        if source in refs:
            n = refs.index(source) + 1
            examples.append({
                "id": f"{g['article_id']}-{g['chunk']}-answer", "kind": "answer", "split": split,
                "question": g["question"], "context": refs, "target": f"{g['answer'].rstrip()} [{n}]",
            })
        if rng.random() < REFUSAL_SHARE:
            quote = normalise(g["quote"]).strip(" .;,")
            others = [d for d in hits if d.metadata["article_id"] != g["article_id"] and quote not in normalise(d.metadata["body"])]
            if len(others) >= 5:
                examples.append({
                    "id": f"{g['article_id']}-{g['chunk']}-refuse", "kind": "refuse", "split": split,
                    "question": g["question"],
                    "context": [(d.metadata["article_id"], d.metadata["chunk"]) for d in others[:5]],
                    "target": REFUSAL,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return examples


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ChatExample:
    id: str
    kind: str
    split: str
    system: str
    user: str
    target: str


def to_chat(examples: list[dict], prompt_id: str = TRAIN_PROMPT) -> list[ChatExample]:
    """Render examples with the shipped prompt, exactly as the pipeline would at answer time."""
    prompt = load_prompt(prompt_id)
    chunks = {(d.metadata["article_id"], d.metadata["chunk"]): d for d in chunk_articles(corpus.load())}
    out = []
    for e in examples:
        docs = [chunks[tuple(ref)] for ref in e["context"]]
        messages = prompt.template().format_messages(context=format_sources(docs), question=e["question"])
        out.append(ChatExample(e["id"], e["kind"], e["split"], messages[0].content, messages[1].content, e["target"]))
    return out
