from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from base_rag.models import Chunk, SearchHit


_TOKEN_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese prose while preserving technical identifiers as one token."""
    terms: list[str] = []
    try:
        import jieba

        split_chinese = jieba.lcut
    except ImportError:  # keeps offline source inspection usable before dependencies are installed
        split_chinese = list
    for part in _TOKEN_PART.findall(text.lower()):
        if part[0].isascii():
            terms.append(part)
        else:
            terms.extend(token.strip() for token in split_chinese(part) if token.strip())
    return terms


def tokenizer_version() -> str:
    try:
        import jieba

        return f"jieba-{getattr(jieba, '__version__', 'unknown')}"
    except ImportError:
        return "character-fallback-no-jieba"


@dataclass(frozen=True)
class BM25Metadata:
    corpus_hash: str
    chunk_count: int
    k1: float
    b: float
    average_document_length: float
    tokenizer_version: str


class BM25Store:
    """Small, persisted BM25 implementation kept intentionally readable for learning."""

    def __init__(self, chunks: list[Chunk], term_frequencies: list[Counter[str]], document_lengths: list[int], document_frequency: Counter[str], metadata: BM25Metadata) -> None:
        self.chunks = chunks
        self.term_frequencies = term_frequencies
        self.document_lengths = document_lengths
        self.document_frequency = document_frequency
        self.metadata = metadata

    @classmethod
    def build(cls, chunks: list[Chunk], corpus_hash: str, k1: float = 1.5, b: float = 0.75) -> "BM25Store":
        frequencies = [Counter(tokenize(_embedding_text(chunk))) for chunk in chunks]
        lengths = [sum(freq.values()) for freq in frequencies]
        document_frequency: Counter[str] = Counter()
        for frequency in frequencies:
            document_frequency.update(frequency.keys())
        average = sum(lengths) / len(lengths) if lengths else 0.0
        metadata = BM25Metadata(corpus_hash, len(chunks), k1, b, average, tokenizer_version())
        return cls(chunks, frequencies, lengths, document_frequency, metadata)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []
        total = len(self.chunks)
        scores: list[tuple[float, int]] = []
        for index, (frequency, length) in enumerate(zip(self.term_frequencies, self.document_lengths)):
            score = 0.0
            for term in query_terms:
                tf = frequency.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + self.metadata.k1 * (1 - self.metadata.b + self.metadata.b * length / max(self.metadata.average_document_length, 1e-9))
                score += idf * tf * (self.metadata.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))
        return [
            SearchHit(self.chunks[index], score, rank, bm25_score=score, bm25_rank=rank)
            for rank, (score, index) in enumerate(scores[:top_k], start=1)
        ]

    def save(self, index_dir: Path) -> None:
        payload = {
            "metadata": asdict(self.metadata),
            "document_lengths": self.document_lengths,
            "document_frequency": dict(self.document_frequency),
            "term_frequencies": [dict(item) for item in self.term_frequencies],
        }
        (index_dir / "bm25.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, chunks: list[Chunk], expected_corpus_hash: str) -> "BM25Store":
        path = index_dir / "bm25.json"
        if not path.exists():
            raise ValueError("当前索引没有 BM25 产物；请重新执行 ingest 后再使用 Hybrid/BM25 Profile。")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = BM25Metadata(**payload["metadata"])
        if metadata.corpus_hash != expected_corpus_hash or metadata.chunk_count != len(chunks):
            raise ValueError("BM25 索引与 Dense 索引不一致；请重新执行 ingest。")
        return cls(
            chunks,
            [Counter(item) for item in payload["term_frequencies"]],
            list(payload["document_lengths"]),
            Counter(payload["document_frequency"]),
            metadata,
        )


def _embedding_text(chunk: Chunk) -> str:
    prefix = "\n".join(value for value in (chunk.title, chunk.section) if value)
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text
