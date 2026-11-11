"""The eval set itself: every expected fact is in the gold article, every trap really is a trap."""

import re

import pytest

from bankrag.grading import normalise

KINDS = {"fact", "false_premise", "out_of_corpus", "injection"}


def test_kinds_and_answerability_agree(all_questions):
    for q in all_questions:
        assert q.kind in KINDS, q.id
        assert q.answerable == (q.kind in {"fact", "false_premise"}), q.id


def test_answerable_questions_are_fully_specified(all_questions, articles):
    for q in (q for q in all_questions if q.answerable):
        assert q.gold and q.expect and q.pass_example and q.fail_example, q.id
        for gold in q.gold:
            assert gold in articles, f"{q.id}: gold article {gold} is not in the corpus"


def test_every_expected_fact_is_in_a_gold_article(all_questions, articles):
    """An answer key the documents can't support would reward outside knowledge."""
    for q in (q for q in all_questions if q.answerable):
        gold_text = " ".join(normalise(articles[g].text) for g in q.gold)
        for pattern in q.expect:
            if pattern in (r"\b(no|not|none)\b",):  # a yes/no verdict, not a quoted fact
                continue
            assert re.search(pattern, gold_text), f"{q.id}: {pattern!r} is not in {q.gold}"


def test_unanswerable_questions_have_no_answer_in_the_corpus(all_questions, articles):
    corpus_text = " ".join(normalise(a.text) for a in articles.values())
    for q in (q for q in all_questions if not q.answerable):
        assert not q.gold, q.id
        assert q.absent, f"{q.id}: give at least one pattern that must be absent from the corpus"
        for pattern in q.absent:
            assert not re.search(pattern, corpus_text), f"{q.id}: {pattern!r} does appear in the corpus"


@pytest.mark.parametrize("kind,minimum", [("fact", 25), ("false_premise", 3), ("out_of_corpus", 10), ("injection", 2)])
def test_each_kind_has_enough_questions(questions, kind, minimum):
    assert sum(q.kind == kind for q in questions) >= minimum
