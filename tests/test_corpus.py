"""The parsed corpus: every article of the three acts, with text and a title."""

from collections import Counter

from bankrag.corpus import LABEL, clean


def test_every_article_of_each_act_is_present(articles):
    counts = Counter(a.source for a in articles.values())
    assert counts == {"PSD2": 117, "DGSD": 23, "GDPR": 99}


def test_articles_are_numbered_without_gaps(articles):
    for source, last in (("PSD2", 117), ("DGSD", 23), ("GDPR", 99)):
        numbers = sorted(int(a.article) for a in articles.values() if a.source == source)
        assert numbers == list(range(1, last + 1))


def test_articles_have_titles_and_text(articles):
    for a in articles.values():
        assert a.title, a.id
        assert len(a.text) > 40, a.id


def test_point_labels_are_joined_to_their_text(articles):
    text = articles["PSD2-97"].text
    assert "(a) accesses its payment account online;" in text
    assert not any(LABEL.match(line) for line in text.splitlines())


def test_known_facts_survive_parsing(articles):
    assert "up to a maximum of EUR 50" in articles["PSD2-74"].text
    assert "EUR 100 000" in articles["DGSD-6"].text
    assert "not later than 72 hours" in articles["GDPR-33"].text


def test_clean_flattens_whitespace():
    assert clean("1.   By way\n of  derogation") == "1. By way of derogation"
