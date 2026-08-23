from __future__ import annotations

from dataclasses import replace

from base_rag.models import SearchHit


def fuse_rrf(dense_hits: list[SearchHit], bm25_hits: list[SearchHit], top_k: int, rrf_k: int) -> list[SearchHit]:
    """Fuse independent rankings without comparing their incompatible raw scores."""
    merged: dict[str, SearchHit] = {}
    for hit in dense_hits + bm25_hits:
        current = merged.get(hit.chunk.chunk_id)
        if current is None:
            merged[hit.chunk.chunk_id] = hit
            continue
        merged[hit.chunk.chunk_id] = replace(
            current,
            dense_score=hit.dense_score if hit.dense_score is not None else current.dense_score,
            dense_rank=hit.dense_rank if hit.dense_rank is not None else current.dense_rank,
            bm25_score=hit.bm25_score if hit.bm25_score is not None else current.bm25_score,
            bm25_rank=hit.bm25_rank if hit.bm25_rank is not None else current.bm25_rank,
        )
    scored: list[SearchHit] = []
    for hit in merged.values():
        score = sum(1 / (rrf_k + rank) for rank in (hit.dense_rank, hit.bm25_rank) if rank is not None)
        scored.append(replace(hit, score=score, rrf_score=score))
    scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
    return [replace(hit, rank=rank, rrf_rank=rank) for rank, hit in enumerate(scored[:top_k], start=1)]
