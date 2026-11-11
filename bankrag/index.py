"""Chunk the articles, embed them, and keep them in Qdrant.

Qdrant runs in local mode inside this process (no server, no account). One collection holds
both a dense vector (bge-small, through FastEmbed) and a sparse BM25 vector per chunk, so the
same index can be searched three ways: dense, sparse (keywords) or hybrid (both, fused with
reciprocal rank fusion inside Qdrant).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from . import corpus
from .config import CACHE_DIR, RETRIEVAL, RetrievalSettings

COLLECTION = "eu_banking_regulation"
MODES = {"dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE, "hybrid": RetrievalMode.HYBRID}
FASTEMBED_CACHE = CACHE_DIR / "fastembed"


class FastEmbedDense(Embeddings):
    """LangChain's Embeddings interface over a FastEmbed (ONNX, CPU) text model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(FASTEMBED_CACHE))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def chunk_articles(articles: list[corpus.Article], settings: RetrievalSettings = RETRIEVAL) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n", "; ", ". ", " ", ""],
    )
    docs = []
    for a in articles:
        header = f"{a.source} Article {a.article}: {a.title}"
        for i, piece in enumerate(splitter.split_text(a.text)):
            docs.append(
                Document(
                    page_content=f"{header}\n{piece}" if settings.chunk_header else piece,
                    metadata={
                        "article_id": a.id,
                        "source": a.source,
                        "article": a.article,
                        "title": a.title,
                        "chunk": i,
                        "header": header,
                        "body": piece,
                    },
                )
            )
    return docs


@dataclass
class Index:
    client: QdrantClient
    dense: FastEmbedDense
    sparse: FastEmbedSparse
    settings: RetrievalSettings
    n_chunks: int
    build_seconds: float

    def store(self, mode: str | None = None) -> QdrantVectorStore:
        """A LangChain vector store over the shared collection, searching in the given mode."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION,
            embedding=self.dense,
            sparse_embedding=self.sparse,
            retrieval_mode=MODES[mode or self.settings.mode],
        )

    def search(self, question: str, k: int | None = None, mode: str | None = None) -> list[tuple[Document, float]]:
        return self.store(mode).similarity_search_with_score(question, k=k or self.settings.k)


def build_index(settings: RetrievalSettings = RETRIEVAL, articles: list[corpus.Article] | None = None) -> Index:
    """Chunk and embed the corpus into an in-memory Qdrant collection."""
    t0 = time.perf_counter()
    docs = chunk_articles(articles or corpus.load(), settings)
    dense = FastEmbedDense(settings.embedding_model)
    sparse = FastEmbedSparse(model_name=settings.sparse_model, cache_dir=str(FASTEMBED_CACHE))
    store = QdrantVectorStore.from_documents(
        docs,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        location=":memory:",
        collection_name=COLLECTION,
    )
    return Index(store.client, dense, sparse, settings, len(docs), time.perf_counter() - t0)
