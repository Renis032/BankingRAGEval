"""Retrieval regression test: no model involved, so it runs anywhere in about 40 seconds.

Embedding and search have no randomness, so these numbers repeat exactly. Any drop means
chunking, the embedding model or the search mode changed, and should be looked at before the
prompt is blamed for worse answers.
"""

import pytest

from bankrag.retrieval_eval import score_retrieval

# Measured with the settings in bankrag/config.py (results/retrieval.json): hybrid search,
# 1,000-character chunks with article headers, bge-small-en-v1.5 + BM25.
FLOORS = {"hit@1": 32 / 33, "hit@5": 1.0, "mrr@10": 0.984}


@pytest.fixture(scope="module")
def hybrid(index, questions):
    return score_retrieval(index, questions, "hybrid")


@pytest.mark.parametrize("metric", FLOORS)
def test_hybrid_retrieval_has_not_regressed(hybrid, metric):
    assert hybrid[metric] >= FLOORS[metric] - 1e-9, f"{metric} fell to {hybrid[metric]:.3f} (floor {FLOORS[metric]:.3f})"


def test_every_answerable_question_retrieves_its_gold_article_in_the_top_5(hybrid):
    assert hybrid["missed_at_5"] == []


def test_search_returns_scored_chunks_with_metadata(index):
    hits = index.search("lost or stolen payment instrument liability", k=3)
    assert len(hits) == 3
    doc, score = hits[0]
    assert doc.metadata["article_id"] == "PSD2-74"
    assert doc.page_content.startswith("PSD2 Article 74:")
    assert score > 0
