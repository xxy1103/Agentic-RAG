from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from base_rag.chunking import chunk_documents
from base_rag.config import AppConfig
from base_rag.embedding import batches
from base_rag.loaders import load_path
from base_rag.models import Chunk, SearchHit
from base_rag.store import FaissStore, IndexMetadata


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str: ...


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
    lengths = [len(chunk.text) for chunk in chunks]
    return {"files": len(files), "documents": len(documents), "chunks": len(chunks), "errors": errors, "lengths": {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)}, "corpus_hash": corpus_hash}


def ask(config: AppConfig, embedder: Embedder, generator: Generator, question: str) -> dict[str, object]:
    started = time.perf_counter()
    store = FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    hits = store.search(embedder.embed([question])[0], config.retrieval.top_k, config.retrieval.min_score)
    if not hits:
        answer = "证据不足，无法基于已检索文档回答。"
        prompt = ""
    else:
        context = _context(hits, config.retrieval.max_context_characters)
        prompt = _prompt(question, context)
        answer = generator.generate(prompt, config.generation.temperature, config.generation.max_tokens)
    citations = [_citation(hit) for hit in hits]
    if citations:
        answer = answer.rstrip() + "\n\n参考来源：\n" + "\n".join(f"- {citation}" for citation in citations)
    result = {"question": question, "hits": [hit.to_dict() for hit in hits], "prompt": prompt, "answer": answer, "citations": citations, "models": {"embedding": config.models.embedding_model, "llm": config.models.llm_model}, "elapsed_seconds": round(time.perf_counter() - started, 3)}
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


def _embedding_text(chunk: Chunk) -> str:
    """把标题/章节一并编码，短小的代码或术语片段也保有语义上下文。"""
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
