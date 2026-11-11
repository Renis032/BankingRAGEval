"""Retrieval on its own, with no model: does the gold article come back, and how high?"""

from __future__ import annotations

import json
import statistics

from .config import RESULTS_DIR, RETRIEVAL, RetrievalSettings
from .grading import Question, load_questions
from .index import MODES, Index, build_index

KS = (1, 3, 5, 10)


def score_retrieval(index: Index, questions: list[Question], mode: str) -> dict:
    """hit@k and MRR@10 over the answerable questions, plus top-1 scores for both groups."""
    answerable = [q for q in questions if q.answerable]
    hits = {k: 0 for k in KS}
    rr = []
    for q in answerable:
        ids = [d.metadata["article_id"] for d, _ in index.search(q.question, k=max(KS), mode=mode)]
        rank = next((i for i, a in enumerate(ids, 1) if a in q.gold), None)
        for k in KS:
            hits[k] += bool(rank and rank <= k)
        rr.append(1 / rank if rank else 0.0)
    result = {f"hit@{k}": hits[k] / len(answerable) for k in KS}
    result["mrr@10"] = statistics.fmean(rr)
    result["missed_at_5"] = [
        q.id for q in answerable
        if not any(d.metadata["article_id"] in q.gold for d, _ in index.search(q.question, k=5, mode=mode))
    ]
    return result


def top1_scores(index: Index, questions: list[Question]) -> dict:
    """Top-1 dense cosine similarity for answerable and unanswerable questions.

    If the two groups separated cleanly, a similarity threshold could refuse before calling
    the model at all. This measures whether they do.
    """
    out = {"answerable": [], "unanswerable": []}
    for q in questions:
        (_, score), = index.search(q.question, k=1, mode="dense")
        out["answerable" if q.answerable else "unanswerable"].append(round(score, 4))
    return out


def run(settings: RetrievalSettings = RETRIEVAL, save: bool = True) -> dict:
    questions = load_questions("main")
    sets = {name: load_questions(name) for name in ("main", "holdout", "plain")}
    report = {"settings": settings.__dict__, "configs": []}
    for header in (True, False):
        index = build_index(settings.with_(chunk_header=header))
        for set_name, set_questions in sets.items():
            for mode in MODES:
                row = {"set": set_name, "mode": mode, "chunk_header": header, **score_retrieval(index, set_questions, mode)}
                report["configs"].append(row)
                print(
                    f"  {set_name:7} {mode:6} header={'on ' if header else 'off'}  "
                    + "  ".join(f"hit@{k} {row[f'hit@{k}']:.3f}" for k in KS)
                    + f"  mrr {row['mrr@10']:.3f}"
                )
        if header == settings.chunk_header:
            report["n_chunks"] = index.n_chunks
            report["index_build_seconds"] = round(index.build_seconds, 1)
            report["top1_dense_scores"] = top1_scores(index, questions)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
