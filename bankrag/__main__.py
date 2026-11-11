"""Command line: python -m bankrag <command>."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext

from .config import DEFAULT_PROMPT, PROVIDERS, QUESTION_SETS


def cmd_build_corpus(args) -> None:
    from collections import Counter

    from . import corpus

    articles = corpus.build(refresh=args.refresh)
    counts = Counter(a.source for a in articles)
    print(f"wrote {len(articles)} articles ({', '.join(f'{k} {v}' for k, v in counts.items())})")


def cmd_retrieval(args) -> None:
    from . import retrieval_eval

    retrieval_eval.run()


def cmd_ask(args) -> None:
    from .index import build_index
    from .llamaserver import LlamaServer, external_url
    from .llm import chat_model, get_provider
    from .rag import RagPipeline, load_prompt

    provider = get_provider(args.provider)
    index = build_index()
    server_cm = LlamaServer() if provider.kind == "llamacpp" and not external_url() else nullcontext(None)
    with server_cm as server:
        base_url = external_url() or (server.base_url if server else None)
        pipeline = RagPipeline(index, chat_model(provider, base_url), load_prompt(args.prompt), provider)
        a = pipeline.ask(args.question)
    print(f"\n{a.answer}\n")
    for s in a.sources:
        print(f"  [{s['n']}] {s['article_id']} (chunk {s['chunk']}, score {s['score']})")
    print(
        f"\n  retrieval {a.retrieval_ms:.0f} ms, generation {a.generation_ms:.0f} ms, "
        f"{a.tokens_in} tokens in, {a.tokens_out} out, ${a.cost_usd:.6f}"
    )


def cmd_eval(args) -> int:
    from .evaluate import accept_baseline, check_against_baseline, run_eval

    if args.set != "main" and (args.accept or args.check):
        raise SystemExit("--check and --accept work on the main eval set only")
    run = run_eval(args.provider, args.prompt, question_set=args.set)
    s = run.summary
    refusal = (
        f"{s['refusal_rate']:.3f}  ({s['n_unanswerable']} unanswerable)"
        if s["n_unanswerable"]
        else "n/a  (no unanswerable questions in this set)"
    )
    print(
        f"\n{run.provider.label}, prompt {run.prompt_id}, {run.question_set} set\n"
        f"  answer accuracy     {s['answer_accuracy']:.3f}  ({s['n_answerable']} answerable)\n"
        f"  citation accuracy   {s['citation_accuracy']:.3f}\n"
        f"  refusal rate        {refusal}\n"
        f"  false refusal rate  {s['false_refusal_rate']:.3f}\n"
        f"  retrieval hit rate  {s['retrieval_hit_rate']:.3f}\n"
        f"  latency p50/p95     {s['latency_ms_p50']:.0f} / {s['latency_ms_p95']:.0f} ms\n"
        f"  tokens in/out       {s['tokens_in_mean']:.0f} / {s['tokens_out_mean']:.0f} per query\n"
        f"  cost                ${s['cost_usd_total']:.4f} total, ${s['cost_usd_per_query']:.6f} per query\n"
        f"  saved to            {run.path}"
    )
    if args.accept:
        accept_baseline(run)
        print(f"\naccepted as the new baseline for {run.provider.key}")
        return 0
    if args.check:
        results = check_against_baseline(run.provider.key, s)
        print("\nregression check against the accepted baseline:")
        for r in results:
            print("  " + r.describe())
        failed = [r for r in results if not r.passed]
        print("\nFAILED" if failed else "\npassed")
        return 1 if failed else 0
    return 0


def cmd_finetune_data(args) -> None:
    from collections import Counter

    from . import finetune_data as fd
    from .index import build_index
    from .llamaserver import LlamaServer
    from .llm import chat_model, get_provider

    if fd.GENERATED_FILE.exists() and not args.regenerate:
        generated = fd.load_jsonl(fd.GENERATED_FILE)
        print(f"reusing {len(generated)} generated questions from {fd.GENERATED_FILE.name} (--regenerate to redo)")
    else:
        chunks = fd.eligible_chunks()
        n_articles = len({d.metadata["article_id"] for d in chunks})
        print(f"generating one question for each of {len(chunks)} chunks from {n_articles} articles with SmolLM3")
        with LlamaServer() as server:
            generated = fd.generate(chat_model(get_provider("local"), server.base_url), chunks)
    for reason, n in Counter(r["rejected"] or "kept" for r in generated).most_common():
        print(f"  {n:4}  {reason}")
    examples = fd.build_examples(build_index(), generated)
    for (split, kind), n in sorted(Counter((e["split"], e["kind"]) for e in examples).items()):
        print(f"  {split:5} {kind:6} {n}")
    print(f"wrote {len(examples)} examples to {fd.EXAMPLES_FILE}")


def cmd_report(args) -> None:
    from .report import print_report

    print_report()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bankrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-corpus", help="download the three EU acts and split them into articles")
    p.add_argument("--refresh", action="store_true", help="download again even if cached")
    p.set_defaults(func=cmd_build_corpus)

    p = sub.add_parser("retrieval", help="score retrieval alone (dense, sparse, hybrid), no model")
    p.set_defaults(func=cmd_retrieval)

    for name, func, help_ in (("ask", cmd_ask, "answer one question"), ("eval", cmd_eval, "run the eval set")):
        p = sub.add_parser(name, help=help_)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--provider", default="local", choices=list(PROVIDERS))
        p.add_argument("--prompt", default=DEFAULT_PROMPT, help="a prompt id from prompts/")
        if name == "eval":
            p.add_argument("--set", default="main", choices=list(QUESTION_SETS), help="which eval set to run")
            group = p.add_mutually_exclusive_group()
            group.add_argument("--check", action="store_true", help="compare with the accepted baseline; exit 1 on a regression")
            group.add_argument("--accept", action="store_true", help="accept this run as the provider's new baseline")
        p.set_defaults(func=func)

    p = sub.add_parser("finetune-data", help="generate the LoRA experiment's training data (needs llama-server)")
    p.add_argument("--regenerate", action="store_true", help="ask the generator again even if data/finetune exists")
    p.set_defaults(func=cmd_finetune_data)

    p = sub.add_parser("report", help="print the latest results as Markdown tables")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
