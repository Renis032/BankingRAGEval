"""The LoRA experiment's training data: built only from articles the evals don't use."""

import re

import pytest

from bankrag import finetune_data as fd
from bankrag.grading import REFUSAL, load_questions, normalise

BODY = "1. The payer shall notify the payment service provider without undue delay on becoming aware of the loss."


def qa(question="Under PSD2, when must a payer report a lost card?", answer="Without undue delay.",
       quote="notify the payment service provider without undue delay"):
    return {"question": question, "answer": answer, "quote": quote}


def test_a_good_question_is_kept():
    assert fd.check_qa(qa(), BODY) is None


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"quote": "notify the bank within 24 hours of the loss"}, "quote not in the passage"),
        ({"quote": "without undue delay"}, "quote too short"),
        ({"question": "What does the passage say about lost cards?"}, "question refers to the passage or an article number"),
        ({"question": "What does Article 69 require when a card is lost?"}, "question refers to the passage or an article number"),
        ({"question": "Under PSD2, when must a payer report a lost card"}, "question shape"),
        ({"answer": "As the passage says, without undue delay."}, "answer refers to the passage"),
    ],
)
def test_bad_questions_are_rejected_with_a_reason(change, reason):
    assert fd.check_qa(qa(**change), BODY) == reason


def test_eligible_chunks_avoid_eval_articles_and_administrative_ones():
    gold = fd.eval_gold_articles()
    chunks = fd.eligible_chunks()
    assert chunks
    for d in chunks:
        assert d.metadata["article_id"] not in gold
        assert not fd.ADMIN_TITLES.search(d.metadata["title"])


needs_examples = pytest.mark.skipif(not fd.EXAMPLES_FILE.exists(), reason="run `python -m bankrag finetune-data` first")


@pytest.fixture(scope="module")
def examples():
    return fd.load_jsonl(fd.EXAMPLES_FILE)


@needs_examples
def test_no_example_comes_from_an_eval_gold_article(examples):
    gold = fd.eval_gold_articles()
    for e in examples:
        assert e["id"].rsplit("-", 2)[0] not in gold, e["id"]


@needs_examples
def test_no_training_question_repeats_an_eval_question(examples):
    eval_questions = {normalise(q.question) for s in ("main", "holdout", "plain") for q in load_questions(s)}
    assert not eval_questions & {normalise(e["question"]) for e in examples}


@needs_examples
def test_answer_targets_cite_the_source_chunk(examples):
    for e in (e for e in examples if e["kind"] == "answer"):
        article, chunk = e["id"].rsplit("-", 2)[0], int(e["id"].rsplit("-", 2)[1])
        n = int(re.search(r"\[(\d)\]$", e["target"]).group(1))
        assert tuple(e["context"][n - 1]) == (article, chunk), e["id"]


@needs_examples
def test_refusal_contexts_hold_nothing_from_the_source_article(examples):
    for e in (e for e in examples if e["kind"] == "refuse"):
        article = e["id"].rsplit("-", 2)[0]
        assert e["target"] == REFUSAL
        assert len(e["context"]) == 5
        assert all(ref[0] != article for ref in e["context"]), e["id"]


@needs_examples
def test_train_and_validation_come_from_different_articles(examples):
    def articles(split):
        return {e["id"].rsplit("-", 2)[0] for e in examples if e["split"] == split}

    assert articles("train") and articles("val")
    assert not articles("train") & articles("val")
