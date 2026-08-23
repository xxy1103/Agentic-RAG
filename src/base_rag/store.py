from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from base_rag.models import Chunk, SearchHit


@dataclass(frozen=True)
class IndexMetadata:
    embedding_model: str
    embedding_dimensions: int
    corpus_hash: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int


class FaissStore:
    def __init__(self, index, chunks: list[Chunk], metadata: IndexMetadata) -> None:
        self.index = index
        self.chunks = chunks
        self.metadata = metadata

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk], metadata: IndexMetadata) -> "FaissStore":
        import faiss

        if len(vectors) != len(chunks):
            raise ValueError("向量数量必须与 Chunk 数量一致。")
        normalised = _normalise(vectors)
        index = faiss.IndexFlatIP(metadata.embedding_dimensions)
        index.add(normalised)
        return cls(index, chunks, metadata)

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float | None) -> list[SearchHit]:
        if not self.chunks:
            return []
        scores, indexes = self.index.search(_normalise(query_vector.reshape(1, -1)), min(top_k, len(self.chunks)))
        hits: list[SearchHit] = []
        for rank, (score, index) in enumerate(zip(scores[0], indexes[0]), start=1):
            if index < 0 or (min_score is not None and float(score) < min_score):
                continue
            hits.append(SearchHit(self.chunks[int(index)], float(score), rank, dense_score=float(score), dense_rank=rank))
        return hits

    def save(self, index_dir: Path) -> None:
        import faiss

        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent))
        try:
            faiss.write_index(self.index, str(temporary / "index.faiss"))
            (temporary / "chunks.jsonl").write_text("\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in self.chunks) + "\n", encoding="utf-8")
            (temporary / "metadata.json").write_text(json.dumps(asdict(self.metadata), ensure_ascii=False, indent=2), encoding="utf-8")
            backup = index_dir.with_name(f".{index_dir.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            if index_dir.exists():
                os.replace(index_dir, backup)
            os.replace(temporary, index_dir)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, index_dir: Path, expected_model: str, expected_dimensions: int) -> "FaissStore":
        import faiss

        metadata = IndexMetadata(**json.loads((index_dir / "metadata.json").read_text(encoding="utf-8")))
        if metadata.embedding_model != expected_model or metadata.embedding_dimensions != expected_dimensions:
            raise ValueError("当前 YAML 的 Embedding 模型或维度与 FAISS 索引不一致；请重新 ingest。")
        chunks = [Chunk.from_dict(json.loads(line)) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        return cls(faiss.read_index(str(index_dir / "index.faiss")), chunks, metadata)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding 含零向量，无法进行余弦检索。")
    return values / norms
