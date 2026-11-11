"""The graders, checked on hand-written answers before any model output is trusted."""

import datetime as dt

import pytest

from bankrag.grading import REFUSAL, cited_numbers, grade, is_refusal, load_questions, normalise
from bankrag.llamaserver import pin_template_date

ALL = load_questions("main") + load_questions("holdout") + load_questions("plain")
ANSWERABLE = [q for q in ALL if q.answerable]
UNANSWERABLE = [q for q in ALL if not q.answerable]


def retrieved_for(q):
    """Five retrieved articles, the gold one in every slot, so any citation [1]..[5] hits it."""
    return [q.gold[0]] * 5


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q.id)
def test_grader_accepts_the_right_answer(q):
    g = grade(q, q.pass_example, retrieved_for(q))
    assert g.correct and g.cites_gold and not g.refused


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q.id)
def test_grader_rejects_the_wrong_answer(q):
    assert not grade(q, q.fail_example, retrieved_for(q)).correct


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q.id)
def test_a_refusal_is_wrong_when_the_answer_is_in_the_documents(q):
    g = grade(q, REFUSAL, retrieved_for(q))
    assert g.refused and not g.correct


@pytest.mark.parametrize("q", UNANSWERABLE, ids=lambda q: q.id)
def test_only_the_refusal_is_right_when_the_answer_is_not_in_the_documents(q):
    retrieved = ["PSD2-1"] * 5
    assert grade(q, REFUSAL, retrieved).correct
    assert grade(q, "i cannot answer this from the provided documents", retrieved).correct
    assert not grade(q, "The minimum ratio is 4.5 % [1].", retrieved).correct
    # A paraphrased refusal breaks the output contract, so it is not counted as one.
    assert not grade(q, "The sources don't say.", retrieved).correct


def test_citation_must_point_at_the_gold_article():
    q = next(q for q in ANSWERABLE if q.id == "gdpr-breach-notification")
    answer = "Not later than 72 hours after becoming aware of it [2]."
    assert grade(q, answer, ["GDPR-33", "GDPR-34"]).correct
    assert not grade(q, answer, ["GDPR-33", "GDPR-34"]).cites_gold
    assert grade(q, answer, ["GDPR-34", "GDPR-33"]).cites_gold


def test_out_of_range_citation_is_invalid():
    q = ANSWERABLE[0]
    g = grade(q, q.pass_example.replace("[1]", "[9]"), ["X"] * 5)
    assert not g.citation_valid and not g.cites_gold


@pytest.mark.parametrize(
    "text,expected",
    [("as stated [1].", [1]), ("see [2][1] and [2]", [2, 1]), ("[1, 3]", [1, 3]), ("[Source 4]", [4]), ("[1 and 2]", [1, 2]), ("no citation", [])],
)
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("EUR 100 000", "eur 100000"),
        ("€100,000", "€100000"),
        ("EUR 20 000 000", "eur 20000000"),
        ("0,8 %", "0,8 %"),
        ("Payer’s 13 months", "payer's 13 months"),
    ],
)
def test_normalise(text, expected):
    assert normalise(text) == expected


def test_refusal_detection_ignores_case_and_surrounding_text():
    assert is_refusal("I cannot answer this from the provided documents.")
    assert is_refusal("Sorry. I CANNOT answer this from the provided documents")
    assert not is_refusal("I cannot answer that.")


def test_chat_template_date_is_pinned():
    template = '{%- set today = strftime_now("%d %B %Y") -%}{{ today }} {{ strftime_now(\'%Y-%m-%d\') }}'
    pinned = pin_template_date(template, dt.date(2026, 1, 15))
    assert "strftime_now" not in pinned
    assert '"15 January 2026"' in pinned and '"2026-01-15"' in pinned
