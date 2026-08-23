from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    source_id: str
    text: str
    source_path: str
    media_type: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    source_path: str
    media_type: str
    ordinal: int
    title: str | None = None
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "rrf_rank": self.rrf_rank,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
        }
