"""Download the three EU acts and split them into one record per article.

Only the articles are kept (the enacting terms), not the recitals or annexes. Each record
keeps the act, the article number and its title, so an answer can cite "PSD2 Article 74".
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass

import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .config import CORPUS_FILE, RAW_DIR, SOURCES, Source

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# A paragraph that holds only a point label, such as "(a)", "(iv)" or "2.", is joined to
# the paragraph after it: the Official Journal puts the label and its text in two cells.
LABEL = re.compile(r"^(\(?[a-z0-9]{1,4}\)|\d{1,2}\.)$")


@dataclass
class Article:
    id: str  # "PSD2-74"
    source: str  # "PSD2"
    article: str  # "74"
    title: str
    text: str
    act: str
    url: str


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def download(source: Source, refresh: bool = False) -> str:
    path = RAW_DIR / f"{source.celex}.xhtml"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = httpx.get(
        source.download_url,
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=60,
    )
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.text


def parse_articles(html: str, source: Source) -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    articles = []
    for div in soup.select("div.eli-subdivision[id^=art_]"):
        number = div["id"].removeprefix("art_")
        if not re.fullmatch(r"\d+[a-z]?", number):
            continue
        title_p = div.select_one("p.oj-sti-art")
        title = clean(title_p.get_text(" ")) if title_p else ""
        lines: list[str] = []
        pending_label = ""
        for p in div.find_all("p"):
            classes = p.get("class") or []
            if "oj-ti-art" in classes or "oj-sti-art" in classes or "oj-note" in classes:
                continue
            text = clean(p.get_text(" "))
            if not text:
                continue
            if LABEL.match(text):
                pending_label = f"{pending_label} {text}".strip()
                continue
            lines.append(f"{pending_label} {text}".strip() if pending_label else text)
            pending_label = ""
        articles.append(
            Article(
                id=f"{source.key}-{number}",
                source=source.key,
                article=number,
                title=title,
                text="\n".join(lines),
                act=source.title,
                url=source.human_url,
            )
        )
    return articles


def build(refresh: bool = False) -> list[Article]:
    """Download (or reuse) each act, parse it, and write data/corpus.jsonl."""
    articles: list[Article] = []
    for source in SOURCES:
        parsed = parse_articles(download(source, refresh), source)
        if not parsed:
            raise RuntimeError(f"no articles found in {source.key} ({source.download_url})")
        articles += parsed
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for a in articles:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    return articles


def load() -> list[Article]:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"{CORPUS_FILE} is missing: run `python -m bankrag build-corpus` first")
    with CORPUS_FILE.open(encoding="utf-8") as f:
        return [Article(**json.loads(line)) for line in f if line.strip()]
