import pytest

from bankrag.config import DEFAULT_PROMPT, PROVIDERS


def pytest_addoption(parser):
    parser.addoption("--provider", default="local", choices=list(PROVIDERS), help="model for the live regression run")
    parser.addoption("--prompt", default=DEFAULT_PROMPT, help="prompt id from prompts/ for the live regression run")


@pytest.fixture(scope="session")
def questions():
    from bankrag.grading import load_questions

    return load_questions("main")


@pytest.fixture(scope="session")
def all_questions():
    """Every eval set together, for checks on the eval data itself."""
    from bankrag.grading import load_questions

    questions = load_questions("main") + load_questions("holdout") + load_questions("plain")
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique across both sets"
    return questions


@pytest.fixture(scope="session")
def articles():
    from bankrag import corpus

    return {a.id: a for a in corpus.load()}


@pytest.fixture(scope="session")
def index():
    from bankrag.index import build_index

    return build_index()
