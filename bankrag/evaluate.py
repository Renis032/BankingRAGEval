"""Run the eval set, score it, record every answer, and compare with the accepted baseline."""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import BASELINES_FILE, GATES_FILE, RESULTS_DIR, RETRIEVAL, ROOT, Provider, RetrievalSettings
from .grading import Question, grade, load_questions
from .index import Index, build_index
from .llamaserver import LlamaServer, external_url
from .llm import chat_model, get_provider
from .rag import RagPipeline, load_prompt

RUNS_DIR = RESULTS_DIR / "runs"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarise(rows: list[dict]) -> dict:
    """Metrics for one run. Rates are 0..1; every one is computed from the per-answer rows."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    n_a, n_u = len(ans), len(una)

    def rate(rs, key):
        return sum(1 for r in rs if r[key]) / len(rs) if rs else 0.0

    correct_a = sum(1 for r in ans if r["correct"])
    correct_u = sum(1 for r in una if r["correct"])
    kinds = sorted({r["kind"] for r in rows})
    total_ms = [r["retrieval_ms"] + r["generation_ms"] for r in rows]
    return {
        "n_questions": len(rows),
        "n_answerable": n_a,
        "n_unanswerable": n_u,
        # gated metrics
        "retrieval_hit_rate": rate(ans, "retrieval_hit"),
        "answer_accuracy": correct_a / n_a if n_a else 0.0,
        "citation_accuracy": rate(ans, "cites_gold"),
        "refusal_rate": correct_u / n_u if n_u else 0.0,
        "false_refusal_rate": rate(ans, "refused"),
        # context
        "answer_accuracy_ci95": wilson(correct_a, n_a),
        "refusal_rate_ci95": wilson(correct_u, n_u),
        "grounded_accuracy": sum(1 for r in ans if r["correct"] and r["cites_gold"]) / n_a if n_a else 0.0,
        "accuracy_when_retrieved": (
            sum(1 for r in ans if r["correct"] and r["retrieval_hit"]) / max(1, sum(1 for r in ans if r["retrieval_hit"]))
        ),
        "by_kind": {
            k: {
                "n": sum(1 for r in rows if r["kind"] == k),
                "correct": sum(1 for r in rows if r["kind"] == k and r["correct"]),
            }
            for k in kinds
        },
        "latency_ms_p50": percentile(total_ms, 0.5),
        "latency_ms_p95": percentile(total_ms, 0.95),
        "retrieval_ms_mean": statistics.fmean(r["retrieval_ms"] for r in rows) if rows else 0.0,
        "generation_ms_mean": statistics.fmean(r["generation_ms"] for r in rows) if rows else 0.0,
        "tokens_in_mean": statistics.fmean(r["tokens_in"] for r in rows) if rows else 0.0,
        "tokens_out_mean": statistics.fmean(r["tokens_out"] for r in rows) if rows else 0.0,
        "cost_usd_total": sum(r["cost_usd"] for r in rows),
        "cost_usd_per_query": statistics.fmean(r["cost_usd"] for r in rows) if rows else 0.0,
    }


@dataclass
class Run:
    provider: Provider
    prompt_id: str
    settings: RetrievalSettings
    rows: list[dict]
    summary: dict
    started: str
    path: Path | None = None
    question_set: str = "main"


def run_eval(
    provider_key: str,
    prompt_id: str,
    index: Index | None = None,
    question_set: str = "main",
    save: bool = True,
    progress: bool = True,
    llm=None,
) -> Run:
    """Run one eval set. Pass `llm` to use a chat model that is already loaded."""
    provider = get_provider(provider_key)
    prompt = load_prompt(prompt_id)
    questions: list[Question] = load_questions(question_set)
    index = index or build_index(RETRIEVAL)
    started = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")

    server_cm = nullcontext(None)
    if llm is None and provider.kind == "llamacpp" and not external_url():
        server_cm = LlamaServer()
    rows = []
    with server_cm as server:
        base_url = external_url() or (server.base_url if server else None)
        pipeline = RagPipeline(index, llm or chat_model(provider, base_url), prompt, provider)
        for i, q in enumerate(questions, 1):
            a = pipeline.ask(q.question)
            g = grade(q, a.answer, a.retrieved_ids)
            row = {
                "id": q.id,
                "kind": q.kind,
                "answerable": q.answerable,
                "question": q.question,
                "answer": a.answer,
                "gold": q.gold,
                "retrieved": a.sources,
                "retrieval_hit": any(r in q.gold for r in a.retrieved_ids),
                **g.as_dict(),
                "retrieval_ms": round(a.retrieval_ms, 1),
                "generation_ms": round(a.generation_ms, 1),
                "tokens_in": a.tokens_in,
                "tokens_out": a.tokens_out,
                "cost_usd": a.cost_usd,
            }
            rows.append(row)
            if progress:
                mark = "ok  " if row["correct"] else "FAIL"
                print(f"  [{i:2}/{len(questions)}] {mark} {q.id:34} {a.generation_ms:7.0f} ms  {a.tokens_in:5}/{a.tokens_out:<4} tok")
    run = Run(provider, prompt_id, index.settings, rows, summarise(rows), started, question_set=question_set)
    if save:
        save_run(run)
    return run


def save_run(run: Run) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{run.started}_{run.provider.key}_{run.prompt_id}"
    if run.question_set != "main":
        stem += f"_{run.question_set}"
    with (RUNS_DIR / f"{stem}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in run.rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "started": run.started,
        "provider": run.provider.key,
        "model": run.provider.model,
        "model_label": run.provider.label,
        "prompt": run.prompt_id,
        "question_set": run.question_set,
        "retrieval": run.settings.__dict__,
        "summary": run.summary,
    }
    path = RUNS_DIR / f"{stem}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    run.path = path
    return path


# --- Regression gate ----------------------------------------------------------------------


def load_gates(provider_key: str) -> dict:
    spec = yaml.safe_load(GATES_FILE.read_text(encoding="utf-8"))
    gates = {name: dict(g) for name, g in spec["metrics"].items()}
    for name, override in (spec.get("providers", {}).get(provider_key) or {}).items():
        gates[name].update(override)
    return gates


def load_baselines() -> dict:
    return json.loads(BASELINES_FILE.read_text(encoding="utf-8")) if BASELINES_FILE.exists() else {}


@dataclass
class GateResult:
    metric: str
    baseline: float
    current: float
    margin: float
    higher_is_better: bool

    @property
    def passed(self) -> bool:
        eps = 1e-9
        if self.higher_is_better:
            return self.current >= self.baseline - self.margin - eps
        return self.current <= self.baseline + self.margin + eps

    def describe(self) -> str:
        direction = "drop" if self.higher_is_better else "rise"
        verdict = "ok" if self.passed else "REGRESSION"
        return (
            f"{self.metric:20} baseline {self.baseline:.3f}  now {self.current:.3f}  "
            f"(allowed {direction} {self.margin:.3f})  {verdict}"
        )


def check_against_baseline(provider_key: str, summary: dict) -> list[GateResult]:
    baselines = load_baselines()
    if provider_key not in baselines:
        raise SystemExit(f"no accepted baseline for {provider_key!r}; run `eval --provider {provider_key} --accept` first")
    base = baselines[provider_key]["metrics"]
    return [
        GateResult(name, base[name], summary[name], g["margin"], g["higher_is_better"])
        for name, g in load_gates(provider_key).items()
    ]


def accept_baseline(run: Run) -> None:
    if run.question_set != "main":
        raise SystemExit("only a run on the main eval set can become a baseline")
    baselines = load_baselines()
    baselines[run.provider.key] = {
        "prompt": run.prompt_id,
        "model": run.provider.model,
        "run": run.path.relative_to(ROOT).as_posix() if run.path else None,
        "metrics": {k: run.summary[k] for k in load_gates(run.provider.key)},
    }
    BASELINES_FILE.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8", newline="\n")
