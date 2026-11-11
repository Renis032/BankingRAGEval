"""Print the latest run for each provider and prompt as Markdown, for the README."""

from __future__ import annotations

import json

from .config import PROVIDERS
from .evaluate import RUNS_DIR


def latest_runs() -> dict[tuple[str, str, str], dict]:
    runs: dict[tuple[str, str, str], dict] = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["file"] = path.name
        runs[(meta["provider"], meta["prompt"], meta.get("question_set", "main"))] = meta  # sorted by time, so the last one wins
    return runs


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def print_report() -> None:
    runs = latest_runs()
    if not runs:
        print("no runs yet: python -m bankrag eval")
        return
    print("| Model | Prompt | Set | Answer accuracy (95% CI) | Cites the right article | Refuses when it should (95% CI) | Wrongly refuses | p50 / p95 latency | Tokens in / out | Cost per query |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (provider, prompt, qset), m in sorted(runs.items()):
        s = m["summary"]
        lo, hi = s["answer_accuracy_ci95"]
        rlo, rhi = s["refusal_rate_ci95"]
        refusal = f"**{pct(s['refusal_rate'])}** ({pct(rlo)}–{pct(rhi)})" if s["n_unanswerable"] else "–"
        cost = "$0 (local)" if PROVIDERS[provider].usd_per_mtok_in == 0 else f"${s['cost_usd_per_query']:.5f}"
        print(
            f"| {m['model_label']} | {prompt} | {qset} | **{pct(s['answer_accuracy'])}** ({pct(lo)}–{pct(hi)}) "
            f"| {pct(s['citation_accuracy'])} | {refusal} "
            f"| {pct(s['false_refusal_rate'])} | {s['latency_ms_p50'] / 1000:.2f} / {s['latency_ms_p95'] / 1000:.2f} s "
            f"| {s['tokens_in_mean']:.0f} / {s['tokens_out_mean']:.0f} | {cost} |"
        )
    print()
    print("| Model | Prompt | Set | " + " | ".join(["fact", "false_premise", "out_of_corpus", "injection"]) + " | Run file |")
    print("|---|---|---|---|---|---|---|---|")
    for (provider, prompt, qset), m in sorted(runs.items()):
        bk = m["summary"]["by_kind"]
        cells = [f"{bk[k]['correct']}/{bk[k]['n']}" if k in bk else "–" for k in ("fact", "false_premise", "out_of_corpus", "injection")]
        print(f"| {m['model_label']} | {prompt} | {qset} | " + " | ".join(cells) + f" | `{m['file']}` |")
