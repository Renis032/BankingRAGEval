"""Code graders: is an answer a refusal, is it correct, and does it cite the right source.

No model grades another model here. Every rule is a regular expression or a lookup, and
tests/test_grading.py checks each one against hand-written right and wrong answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import QUESTION_SETS, QUESTIONS_FILE

# The exact sentence the prompt asks for when the sources don't hold the answer. A refusal
# phrased any other way doesn't count: a fixed sentence is a contract the application
# around the model can rely on, and keeping to it is part of what is tested.
REFUSAL = "I cannot answer this from the provided documents."
_REFUSAL_KEY = "cannot answer this from the provided documents"

_THOUSANDS = re.compile(r"(?<=\d)[ ,.'](?=\d{3}\b)")
# [1], [1, 3], [1][2], [Source 2]
_CITATION = re.compile(r"\[(?:sources?\s*)?(\d+(?:\s*(?:,|and|&)\s*(?:sources?\s*)?\d+)*)\]", re.I)


def normalise(text: str) -> str:
    """Lower case, plain spaces and quotes, no thousands separators: 'EUR 100 000' -> 'eur 100000'."""
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = _THOUSANDS.sub("", text)
    return text.strip().lower()


def is_refusal(answer: str) -> bool:
    return _REFUSAL_KEY in normalise(answer)


def cited_numbers(answer: str) -> list[int]:
    """Source numbers cited in the answer, in order of first appearance."""
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for n in re.findall(r"\d+", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


@dataclass
class Question:
    id: str
    kind: str
    answerable: bool
    question: str
    gold: list[str] = field(default_factory=list)
    expect: list[str] = field(default_factory=list)
    reject: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    pass_example: str = ""
    fail_example: str = ""


def load_questions(path: Path | str = QUESTIONS_FILE) -> list[Question]:
    """Load an eval set, by path or by name ("main", "holdout")."""
    path = QUESTION_SETS.get(path, path) if isinstance(path, str) else path
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    questions = []
    for row in rows:
        row = dict(row)
        row["pass_example"] = row.pop("pass", "")
        row["fail_example"] = row.pop("fail", "")
        questions.append(Question(**row))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in the eval set")
    return questions


def facts_match(question: Question, answer: str) -> bool:
    """Every `expect` pattern matches and no `reject` pattern does."""
    text = normalise(answer)
    return all(re.search(p, text) for p in question.expect) and not any(
        re.search(p, text) for p in question.reject
    )


@dataclass
class Grade:
    refused: bool
    correct: bool  # the outcome this question wants: right facts, or a refusal
    cited: list[int]  # source numbers the answer cites
    citation_valid: bool  # it cites at least one source, and only numbers that exist
    cites_gold: bool  # one of the cited sources is a gold article

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def grade(question: Question, answer: str, retrieved_ids: list[str]) -> Grade:
    """Grade one answer. retrieved_ids[i] is the article id behind source number i + 1."""
    refused = is_refusal(answer)
    cited = cited_numbers(answer)
    valid = bool(cited) and all(1 <= n <= len(retrieved_ids) for n in cited)
    cites_gold = any(1 <= n <= len(retrieved_ids) and retrieved_ids[n - 1] in question.gold for n in cited)
    if question.answerable:
        correct = not refused and facts_match(question, answer)
    else:
        correct = refused
    return Grade(refused=refused, correct=correct, cited=cited, citation_valid=valid, cites_gold=cites_gold)
