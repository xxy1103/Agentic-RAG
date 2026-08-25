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


@dataclass(slots=True)
class EvidenceRequirement:
    requirement_id: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "description": self.description,
        }


@dataclass(slots=True)
class RouteDecision:
    route: str
    query: str
    reason: str
    requirements: list[EvidenceRequirement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "query": self.query,
            "reason": self.reason,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
        }


@dataclass(slots=True)
class RequirementAssessment:
    requirement_id: str
    status: str
    evidence_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
        }


@dataclass(slots=True)
class EvidenceDecision:
    verdict: str
    reason: str
    model_verdict: str | None = None
    next_query: str | None = None
    next_requirement_id: str | None = None
    failure_reason: str | None = None
    requirement_assessments: list[RequirementAssessment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "model_verdict": self.model_verdict or self.verdict,
            "reason": self.reason,
            "next_query": self.next_query,
            "next_requirement_id": self.next_requirement_id,
            "failure_reason": self.failure_reason,
            "requirement_assessments": [assessment.to_dict() for assessment in self.requirement_assessments],
        }


@dataclass(slots=True)
class HopTrace:
    hop_index: int
    is_correction: bool
    query: str
    hits: list[SearchHit]
    decision: EvidenceDecision | None
    elapsed_seconds: float
    stage_calls: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "is_correction": self.is_correction,
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "decision": self.decision.to_dict() if self.decision else None,
            "elapsed_seconds": self.elapsed_seconds,
            "stage_calls": dict(self.stage_calls),
        }


@dataclass(slots=True)
class AgenticRetrievalResult:
    question: str
    route_decision: RouteDecision
    final_hits: list[SearchHit]
    traces: list[HopTrace]
    termination_reason: str
    total_hops: int
    correction_count: int
    elapsed_seconds: float
    stage_calls: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "route_decision": self.route_decision.to_dict(),
            "final_hits": [hit.to_dict() for hit in self.final_hits],
            "traces": [trace.to_dict() for trace in self.traces],
            "termination_reason": self.termination_reason,
            "total_hops": self.total_hops,
            "correction_count": self.correction_count,
            "elapsed_seconds": self.elapsed_seconds,
            "stage_calls": dict(self.stage_calls),
        }
