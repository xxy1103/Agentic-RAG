from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from base_rag.bm25 import BM25Store
from base_rag.chunking import chunk_documents
from base_rag.config import AppConfig
from base_rag.embedding import batches
from base_rag.loaders import load_path
from base_rag.models import Chunk, SearchHit
from base_rag.rerank import DashScopeReranker, Reranker
from base_rag.retrieval import fuse_rrf
from base_rag.rewrite import QueryAnalysis, QueryRewriter
from base_rag.store import FaissStore, IndexMetadata


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str: ...


class PipelineStageError(RuntimeError):
    """Adds a machine-readable stage name to external-service failures."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage} 阶段失败：{cause}")
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    dense: bool
    bm25: bool
    rewrite: bool
    rerank: bool


PROFILES = {
    "dense": RetrievalProfile("dense", True, False, False, False),
    "bm25": RetrievalProfile("bm25", False, True, False, False),
    "hybrid": RetrievalProfile("hybrid", True, True, False, False),
    "hybrid-rerank": RetrievalProfile("hybrid-rerank", True, True, False, True),
    "advanced": RetrievalProfile("advanced", True, True, True, True),
}


def profile_for(config: AppConfig, name: str | None) -> RetrievalProfile:
    selected = name or config.runtime.default_profile
    try:
        profile = PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"未知 Profile：{selected}。可用值：{', '.join(PROFILES)}") from exc
    if profile.rewrite and not config.query_rewrite.enabled:
        raise ValueError("当前 YAML 禁用了 query_rewrite，无法运行 advanced Profile。")
    if profile.rerank and not config.reranker.enabled:
        raise ValueError("当前 YAML 禁用了 reranker，无法运行 hybrid-rerank/advanced Profile。")
    return profile


def ingest(config: AppConfig, embedder: Embedder) -> dict[str, object]:
    files = sorted(path for path in config.paths.corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in config.ingestion.allowed_extensions)
    if not files:
        raise ValueError(f"语料目录没有支持的文件：{config.paths.corpus_dir}")
    documents = []
    errors: list[str] = []
    for path in files:
        try:
            documents.extend(load_path(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors and config.ingestion.fail_on_error:
        raise ValueError("文档解析失败：\n" + "\n".join(errors))
    chunks = chunk_documents(documents, config.ingestion.chunk_size, config.ingestion.chunk_overlap)
    if not chunks:
        raise ValueError("没有生成 Chunk。")
    vectors = np.vstack([embedder.embed(batch) for batch in batches([_embedding_text(chunk) for chunk in chunks], config.ingestion.embedding_batch_size)])
    corpus_hash = _corpus_hash(files)
    metadata = IndexMetadata(config.models.embedding_model, config.models.embedding_dimensions, corpus_hash, len(chunks), config.ingestion.chunk_size, config.ingestion.chunk_overlap)
    FaissStore.build(vectors, chunks, metadata).save(config.paths.index_dir)
    bm25 = BM25Store.build(chunks, corpus_hash)
    bm25.save(config.paths.index_dir)
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "files": len(files),
        "documents": len(documents),
        "chunks": len(chunks),
        "errors": errors,
        "lengths": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)},
        "corpus_hash": corpus_hash,
        "bm25": {"k1": bm25.metadata.k1, "b": bm25.metadata.b, "tokenizer_version": bm25.metadata.tokenizer_version},
    }


def ask(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    question: str,
    profile: str | None = None,
    reranker: Reranker | None = None,
    rewriter: QueryRewriter | None = None,
    generate: bool = True,
) -> dict[str, object]:
    started = time.perf_counter()
    selected = profile_for(config, profile)
    stage_seconds: dict[str, float] = {}
    stage_calls: dict[str, int] = {"embedding": 0, "generation": 0, "rewrite": 0, "rerank": 0}
    analysis: QueryAnalysis | None = None
    if selected.rewrite:
        rewrite_started = time.perf_counter()
        analysis = _run_stage("rewrite", lambda: (rewriter or QueryRewriter(generator, config.models, config.query_rewrite)).rewrite(question))
        stage_seconds["rewrite"] = round(time.perf_counter() - rewrite_started, 3)
        stage_calls["rewrite"] += 1
    dense_query = analysis.rewritten_query if analysis else question
    sparse_query = " ".join([analysis.rewritten_query, *analysis.keywords]) if analysis else question

    dense_store = _run_stage("index_load", lambda: FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions))
    bm25_store = _run_stage("bm25_load", lambda: BM25Store.load(config.paths.index_dir, dense_store.chunks, dense_store.metadata.corpus_hash)) if selected.bm25 else None
    dense_hits: list[SearchHit] = []
    if selected.dense:
        dense_started = time.perf_counter()
        dense_hits = _run_stage("dense_retrieval", lambda: dense_store.search(embedder.embed([dense_query])[0], config.retrieval.dense_candidate_k, config.retrieval.min_score))
        stage_seconds["dense"] = round(time.perf_counter() - dense_started, 3)
        stage_calls["embedding"] += 1
    bm25_hits: list[SearchHit] = []
    if bm25_store:
        sparse_started = time.perf_counter()
        bm25_hits = _run_stage("bm25_retrieval", lambda: bm25_store.search(sparse_query, config.retrieval.sparse_candidate_k))
        stage_seconds["bm25"] = round(time.perf_counter() - sparse_started, 3)

    if selected.dense and selected.bm25:
        fusion_started = time.perf_counter()
        candidates = _run_stage("rrf_fusion", lambda: fuse_rrf(dense_hits, bm25_hits, config.retrieval.fusion_candidate_k, config.retrieval.rrf_k))
        stage_seconds["rrf"] = round(time.perf_counter() - fusion_started, 3)
    elif selected.dense:
        candidates = dense_hits
    else:
        candidates = bm25_hits

    final_hits = candidates[: config.retrieval.top_k]
    if selected.rerank and candidates:
        rerank_started = time.perf_counter()
        final_hits = _run_stage("rerank", lambda: (reranker or DashScopeReranker(config.models, config.reranker)).rerank(question, candidates, config.reranker.top_n))
        stage_seconds["rerank"] = round(time.perf_counter() - rerank_started, 3)
        stage_calls["rerank"] += 1

    prompt = ""
    answer: str | None = None
    if generate:
        if not final_hits:
            answer = "证据不足，无法基于已检索文档回答。"
        else:
            context = _context(final_hits, config.retrieval.max_context_characters)
            prompt = _prompt(question, context)
            generation_started = time.perf_counter()
            answer = _run_stage("generation", lambda: generator.generate(prompt, config.generation.temperature, config.generation.max_tokens))
            stage_seconds["generation"] = round(time.perf_counter() - generation_started, 3)
            stage_calls["generation"] += 1
    citations = [_citation(hit) for hit in final_hits]
    if answer is not None and citations:
        answer = answer.rstrip() + "\n\n参考来源：\n" + "\n".join(f"- {citation}" for citation in citations)
    result = {
        "question": question,
        "profile": selected.name,
        "query_analysis": analysis.to_dict() if analysis else None,
        "queries": {"dense": dense_query, "bm25": sparse_query},
        "hits": [hit.to_dict() for hit in final_hits],
        "retrieval": {
            "dense": [hit.to_dict() for hit in dense_hits],
            "bm25": [hit.to_dict() for hit in bm25_hits],
            "fused": [hit.to_dict() for hit in candidates] if selected.dense and selected.bm25 else [],
            "candidates": [hit.to_dict() for hit in candidates],
            "final": [hit.to_dict() for hit in final_hits],
        },
        "prompt": prompt,
        "answer": answer,
        "citations": citations,
        "models": {"embedding": config.models.embedding_model, "llm": config.models.llm_model, "reranker": config.reranker.model if selected.rerank else None},
        "stage_seconds": stage_seconds,
        "stage_calls": stage_calls,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if config.runtime.save_runs:
        _save_run(config.paths.runs_dir, result, config.safe_dict())
    return result


def _context(hits: list[SearchHit], limit: int) -> str:
    blocks: list[str] = []
    used = 0
    for hit in hits:
        label = _citation(hit)
        block = f"[证据 {hit.rank}: {label}]\n{hit.chunk.text}"
        if used + len(block) > limit:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _run_stage(stage: str, operation):
    try:
        return operation()
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(stage, exc) from exc


def _embedding_text(chunk: Chunk) -> str:
    prefix = "\n".join(value for value in (chunk.title, chunk.section) if value)
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text


def _prompt(question: str, context: str) -> str:
    return f"""你是一个严格基于给定资料回答问题的助手。只能使用“检索证据”中的事实；资料不足或无法直接支持结论时，回答“证据不足，无法基于已检索文档回答”。不要使用外部常识，不要编造来源。请用中文简明回答。\n\n问题：{question}\n\n检索证据：\n{context}"""


def _citation(hit: SearchHit) -> str:
    location = f"第 {hit.chunk.page} 页" if hit.chunk.page else (hit.chunk.section or "正文")
    return f"{Path(hit.chunk.source_path).name}｜{location}"


def _save_run(runs_dir: Path, result: dict[str, object], config: dict[str, object]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    (runs_dir / f"{stamp}.json").write_text(json.dumps({"result": result, "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")


def _corpus_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(path.parents[1])).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
