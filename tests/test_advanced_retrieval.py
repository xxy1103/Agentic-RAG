from pathlib import Path

import numpy as np
import pytest

import base_rag.evaluation as evaluation
from base_rag.bm25 import BM25Store, tokenize
from base_rag.config import load_config
from base_rag.evaluation import _metrics
from base_rag.models import Chunk, SearchHit
from base_rag.pipeline import ask
from base_rag.pipeline import PipelineStageError
from base_rag.rerank import DashScopeReranker
from base_rag.retrieval import fuse_rrf
from base_rag.rewrite import _parse
from base_rag.store import FaissStore, IndexMetadata


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(identifier, text, identifier, f"{identifier}.md", "markdown", 0)


def _config(tmp_path: Path, advanced: bool = False):
    path = tmp_path / "config.yaml"
    raw = "\n".join([
        "paths:", "  corpus_dir: corpus", "  index_dir: index", "  runs_dir: runs",
        "models:", "  api_base: http://example.test", "  embedding_model: fake", "  embedding_dimensions: 2", "  llm_model: fake", "  timeout_seconds: 1", "  max_retries: 1", "  retry_backoff_seconds: 0",
        "ingestion:", "  allowed_extensions: ['.md']", "  chunk_size: 100", "  chunk_overlap: 10", "  embedding_batch_size: 1", "  fail_on_error: true",
        "retrieval:", "  top_k: 1", "  min_score: 0.0", "  max_context_characters: 1000",
        f"query_rewrite:\n  enabled: {str(advanced).lower()}",
        f"reranker:\n  enabled: {str(advanced).lower()}\n  model: qwen3-rerank\n  top_n: 1\n  instruct: test",
        "generation:", "  temperature: 0", "  max_tokens: 10",
        f"runtime:\n  save_runs: false\n  default_profile: {'advanced' if advanced else 'dense'}",
        "evaluation:\n  judge_enabled: false\n  judge_max_tokens: 10", "",
    ])
    path.write_text(raw, encoding="utf-8")
    return load_config(path)


def test_bm25_preserves_technical_identifier_and_round_trips(tmp_path: Path) -> None:
    assert "h.264" in tokenize("H.264 码流")
    chunks = [_chunk("a", "H.264 码流"), _chunk("b", "普通文本")]
    store = BM25Store.build(chunks, "hash")
    store.save(tmp_path)
    assert BM25Store.load(tmp_path, chunks, "hash").search("H.264", 2)[0].chunk.chunk_id == "a"


def test_rrf_merges_scores_and_rank_provenance() -> None:
    first, second = _chunk("a", "甲"), _chunk("b", "乙")
    dense = [SearchHit(first, 0.9, 1, dense_score=0.9, dense_rank=1)]
    sparse = [SearchHit(second, 2.0, 1, bm25_score=2.0, bm25_rank=1), SearchHit(first, 1.0, 2, bm25_score=1.0, bm25_rank=2)]
    fused = fuse_rrf(dense, sparse, 3, 60)
    assert fused[0].chunk.chunk_id == "a"
    assert fused[0].dense_rank == 1 and fused[0].bm25_rank == 2


def test_rewrite_json_validation() -> None:
    assert _parse('{"intent":"查询","rewritten_query":"查询 BM25","keywords":["BM25"]}').keywords == ["BM25"]
    with pytest.raises(ValueError):
        _parse('{"intent":"x","rewritten_query":"y","keywords":[]}')


def test_reranker_rejects_duplicate_indices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace")
    reranker = DashScopeReranker(_config(tmp_path, True).models, _config(tmp_path, True).reranker)
    assert reranker.endpoint == "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
    monkeypatch.setattr(reranker, "_request", lambda payload: {"results": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 0.5}]})
    with pytest.raises(ValueError, match="重复"):
        reranker.rerank("问题", [SearchHit(_chunk("a", "甲"), 1.0, 1)], 2)


class _Embedder:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _Generator:
    def generate(self, prompt, temperature, max_tokens):
        return "回答"


class _Rewriter:
    def rewrite(self, question):
        from base_rag.rewrite import QueryAnalysis
        return QueryAnalysis("意图", "改写问题", ["关键词"])


class _Reranker:
    def rerank(self, query, hits, top_n):
        return hits[:top_n]


def test_advanced_pipeline_records_all_stages(tmp_path: Path) -> None:
    config = _config(tmp_path, True)
    chunks = [_chunk("a", "关键词 证据")]
    FaissStore.build(np.array([[1.0, 0.0]], dtype=np.float32), chunks, IndexMetadata("fake", 2, "hash", 1, 100, 10)).save(config.paths.index_dir)
    BM25Store.build(chunks, "hash").save(config.paths.index_dir)
    result = ask(config, _Embedder(), _Generator(), "原问题", reranker=_Reranker(), rewriter=_Rewriter())
    assert result["profile"] == "advanced"
    assert result["query_analysis"]["rewritten_query"] == "改写问题"
    assert result["stage_calls"]["rewrite"] == 1 and result["stage_calls"]["rerank"] == 1


def test_evaluation_metrics_separate_answerable_questions() -> None:
    records = [
        {"answerable": True, "source_recall_at_6": True, "chunk_recall_at_6": True, "chunk_recall_at_20": True, "mrr_at_6": 1.0, "ndcg_at_6": 1.0, "category": "lexical", "result": {"elapsed_seconds": 0.2, "stage_calls": {"embedding": 1, "rewrite": 0, "rerank": 0, "generation": 0}}},
        {"answerable": False, "source_recall_at_6": None, "chunk_recall_at_6": None, "chunk_recall_at_20": None, "mrr_at_6": None, "ndcg_at_6": None, "category": "unanswerable", "result": {"elapsed_seconds": 0.1, "stage_calls": {"embedding": 0, "rewrite": 0, "rerank": 0, "generation": 0}}},
    ]
    metrics = _metrics(records)
    assert metrics["questions"] == 2 and metrics["answerable_questions"] == 1
    assert metrics["source_recall_at_6"] == 1.0


def test_evaluation_records_failure_writes_checkpoint_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    questions = tmp_path / "questions.yaml"
    questions.write_text("questions:\n  - id: ok\n    question: 第一题\n  - id: blocked\n    question: 第二题\n", encoding="utf-8")

    def fake_ask(config, embedder, generator, question, **kwargs):
        if question == "第二题":
            raise PipelineStageError("generation", ValueError("内容审核拒绝"))
        return {"retrieval": {"final": [], "candidates": []}, "elapsed_seconds": 0.1, "stage_calls": {"embedding": 1, "rewrite": 0, "rerank": 0, "generation": 0}}

    monkeypatch.setattr(evaluation, "ask", fake_ask)
    progress = []
    checkpoint = tmp_path / "checkpoint.json"
    result = evaluation.evaluate(_config(tmp_path), _Embedder(), _Generator(), questions, "dense", checkpoint_path=checkpoint, on_progress=lambda *args: progress.append(args))
    assert [record["status"] for record in result["records"]] == ["ok", "failed"]
    assert result["records"][1]["failure"]["stage"] == "generation"
    assert len(progress) == 2
    assert __import__("json").loads(checkpoint.read_text(encoding="utf-8"))["status"] == "complete"
