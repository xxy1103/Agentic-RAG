"""Independent MultiHop-RAG benchmark preparation and retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from base_rag.bm25 import BM25Store
from base_rag.config import AppConfig
from base_rag.pipeline import Embedder, Generator, PROFILES, PipelineStageError, ask
from base_rag.store import FaissStore


BENCHMARK_NAME = "MultiHop-RAG"
DATASET_REVISION = "main"
DATASET_URLS = {
    "queries": "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json?download=true",
    "corpus": "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json?download=true",
}
MANIFEST_NAME = "multihoprag-manifest.json"

ProgressCallback = Callable[[str, int, int, int, int, str], None]
PrepareProgressCallback = Callable[[str, int, int, str], None]


def require_multihoprag_config(config: AppConfig) -> Path:
    if config.multihoprag is None:
        raise ValueError("eval-multihop/prepare-multihop 需要 YAML 中的 multihoprag.dataset_dir。")
    return config.multihoprag.dataset_dir


def prepare_multihoprag(
    config: AppConfig,
    *,
    force: bool = False,
    download: Callable[[str], bytes] | None = None,
    on_progress: PrepareProgressCallback | None = None,
) -> dict[str, Any]:
    """Download the pinned public files and convert their corpus into Markdown.

    The generated manifest is intentionally separate from the local YAML question
    set: its evidence identity is the public dataset's exact document metadata
    plus fact text, rather than a user-maintained section annotation.
    """
    dataset_dir = require_multihoprag_config(config)
    raw_dir = config.paths.corpus_dir
    manifest_path = dataset_dir / MANIFEST_NAME
    source_dir = dataset_dir / "source"
    if (manifest_path.exists() or raw_dir.exists()) and not force:
        raise ValueError(
            f"MultiHop-RAG 已准备在 {dataset_dir}。如需重新下载和转换，请显式传入 --force。"
        )
    if force:
        _remove_prepared_outputs(dataset_dir, raw_dir)

    download = download or _download
    source_dir.mkdir(parents=True, exist_ok=True)
    query_path = source_dir / "MultiHopRAG.json"
    corpus_path = source_dir / "corpus.json"
    query_bytes = download(DATASET_URLS["queries"])
    if on_progress:
        on_progress("下载数据", 1, 2, "MultiHopRAG.json")
    corpus_bytes = download(DATASET_URLS["corpus"])
    if on_progress:
        on_progress("下载数据", 2, 2, "corpus.json")
    query_path.write_bytes(query_bytes)
    corpus_path.write_bytes(corpus_bytes)
    queries = _load_array(query_path, "问题集")
    corpus = _load_array(corpus_path, "语料")

    raw_dir.mkdir(parents=True, exist_ok=False)
    documents = _write_markdown_corpus(raw_dir, corpus, on_progress)
    manifest = _build_manifest(queries, corpus, documents, on_progress)
    manifest.update(
        {
            "benchmark": BENCHMARK_NAME,
            "dataset_revision": DATASET_REVISION,
            "source_urls": DATASET_URLS,
            "source_sha256": {
                "queries": hashlib.sha256(query_bytes).hexdigest(),
                "corpus": hashlib.sha256(corpus_bytes).hexdigest(),
            },
            "document_count": len(documents),
            "query_count": len(manifest["queries"]),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if on_progress:
        on_progress("写入清单", 1, 1, manifest_path.name)
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_dir": str(dataset_dir),
        "corpus_dir": str(raw_dir),
        "manifest": str(manifest_path),
        "documents": len(documents),
        "queries": len(manifest["queries"]),
        "source_sha256": manifest["source_sha256"],
    }


def evaluate_multihoprag(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    *,
    profiles: tuple[str, ...] = tuple(PROFILES),
    limit: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    dataset_dir = require_multihoprag_config(config)
    manifest = _load_manifest(dataset_dir / MANIFEST_NAME)
    selected_queries = manifest["queries"][:limit] if limit is not None else manifest["queries"]
    queries = [query for query in selected_queries if query["question_type"] != "null_query"]
    if not queries:
        raise ValueError("选定范围内没有可评测的 MultiHop-RAG 问题。")
    invalid_profiles = [profile for profile in profiles if profile not in PROFILES]
    if invalid_profiles:
        raise ValueError(f"未知 Profile：{', '.join(invalid_profiles)}")
    batch_dir = config.paths.runs_dir / "evaluations" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    batch_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        profile_dir = batch_dir / "profiles" / profile
        profile_dir.mkdir(parents=True, exist_ok=False)
        result = _evaluate_profile(config, embedder, generator, queries, profile, profile_dir / "questions", on_progress)
        (profile_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (profile_dir / "REPORT.md").write_text(_profile_report(result), encoding="utf-8")
        results[profile] = result
    summary = {
        "kind": "multihoprag_retrieval_evaluation_batch",
        "benchmark": BENCHMARK_NAME,
        "dataset_revision": manifest["dataset_revision"],
        "dataset_sha256": manifest["source_sha256"],
        "source_query_count": len(selected_queries),
        "query_count": len(queries),
        "skipped_null_queries": len(selected_queries) - len(queries),
        "config": config.safe_dict(),
        "profiles": [{"profile": profile, "metrics": result["metrics"], "artifact_dir": f"profiles/{profile}"} for profile, result in results.items()],
    }
    (batch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "REPORT.md").write_text(_batch_report(summary), encoding="utf-8")
    return batch_dir, results


def _evaluate_profile(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    queries: list[dict[str, Any]],
    profile: str,
    run_log_dir: Path,
    on_progress: ProgressCallback | None,
) -> dict[str, Any]:
    selected = PROFILES[profile]
    dense_store = FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    bm25_store = BM25Store.load(config.paths.index_dir, dense_store.chunks, dense_store.metadata.corpus_hash) if selected.bm25 else None
    concurrency = max(1, min(config.evaluation.concurrency, len(queries)))
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"multihop-{profile}") as executor:
        futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {
            executor.submit(_evaluate_query, config, embedder, generator, query, profile, dense_store, bm25_store, run_log_dir): (index, query)
            for index, query in enumerate(queries)
        }
        for count, future in enumerate(as_completed(futures), start=1):
            index, query = futures[future]
            completed[index] = future.result()
            if on_progress:
                records = list(completed.values())
                on_progress(profile, count, len(queries), sum(record["status"] == "ok" for record in records), sum(record["status"] == "failed" for record in records), query["id"])
    records = [completed[index] for index in range(len(queries))]
    return {"benchmark": BENCHMARK_NAME, "profile": profile, "mode": "retrieval", "records": records, "metrics": _metrics(records)}


def _evaluate_query(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    query: dict[str, Any],
    profile: str,
    dense_store: FaissStore,
    bm25_store: BM25Store | None,
    run_log_dir: Path,
) -> dict[str, Any]:
    try:
        result = ask(config, embedder, generator, query["query"], profile=profile, generate=False, dense_store=dense_store, bm25_store=bm25_store, run_log_dir=run_log_dir)
        return _score_query(query, result)
    except Exception as exc:
        stage = exc.stage if isinstance(exc, PipelineStageError) else "unknown"
        return {"id": query["id"], "query": query["query"], "question_type": query["question_type"], "status": "failed", "failure": {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}}


def _score_query(query: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    hits = result["retrieval"]["final"]
    gold = query["gold"]
    coverage_at_4 = _coverage(hits[:4], gold)
    coverage_at_10 = _coverage(hits[:10], gold)
    first_rank = next((rank for rank, hit in enumerate(hits[:10], start=1) if _hit_matches_any(hit, gold)), None)
    relevant_seen: set[str] = set()
    precision_sum = 0.0
    for rank, hit in enumerate(hits[:10], start=1):
        matched = {item["fact_key"] for item in gold if _hit_matches_fact(hit, item)} - relevant_seen
        if matched:
            relevant_seen.update(matched)
            precision_sum += len(matched) / rank
    return {
        "id": query["id"], "query": query["query"], "answer": query["answer"], "question_type": query["question_type"], "status": "ok", "gold_count": len(gold), "result": result,
        "evidence_coverage_at_4": coverage_at_4,
        "evidence_coverage_at_10": coverage_at_10,
        "complete_evidence_at_4": coverage_at_4 == 1.0,
        "complete_evidence_at_10": coverage_at_10 == 1.0,
        "official_hits_at_4": _hit_matches_any_in(hits[:4], gold),
        "official_hits_at_10": _hit_matches_any_in(hits[:10], gold),
        "official_map_at_10": precision_sum / min(len(gold), 10),
        "official_mrr_at_10": 1 / first_rank if first_rank else 0.0,
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "ok"]
    metric_names = ("evidence_coverage_at_4", "evidence_coverage_at_10", "official_hits_at_4", "official_hits_at_10", "official_map_at_10", "official_mrr_at_10")
    metrics: dict[str, Any] = {"questions": len(records), "completed_questions": len(completed), "failed_questions": len(records) - len(completed)}
    metrics.update({name: round(mean(record[name] for record in completed), 4) if completed else 0.0 for name in metric_names})
    metrics["complete_evidence_at_4"] = _rate(completed, "complete_evidence_at_4")
    metrics["complete_evidence_at_10"] = _rate(completed, "complete_evidence_at_10")
    metrics["mean_latency_seconds"] = round(mean(record["result"]["elapsed_seconds"] for record in completed), 3) if completed else 0.0
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        groups[record["question_type"]].append(record)
    metrics["by_question_type"] = {kind: {name: round(mean(item[name] for item in items), 4) for name in metric_names} for kind, items in groups.items()}
    return metrics


def _coverage(hits: list[dict[str, Any]], gold: list[dict[str, str]]) -> float:
    return round(sum(_hit_matches_any_in(hits, [item]) for item in gold) / len(gold), 4) if gold else 0.0


def _hit_matches_any_in(hits: list[dict[str, Any]], gold: list[dict[str, str]]) -> bool:
    return any(_hit_matches_any(hit, gold) for hit in hits)


def _hit_matches_any(hit: dict[str, Any], gold: list[dict[str, str]]) -> bool:
    return any(_hit_matches_fact(hit, item) for item in gold)


def _hit_matches_fact(hit: dict[str, Any], fact: dict[str, str]) -> bool:
    chunk = hit["chunk"]
    return Path(chunk["source_path"]).name == fact["source"] and fact["normalised_fact"] in _normalise(chunk["text"])


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(record[key]) for record in records) / len(records), 4) if records else 0.0


def _write_markdown_corpus(raw_dir: Path, corpus: list[dict[str, Any]], on_progress: PrepareProgressCallback | None = None) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for index, item in enumerate(corpus, start=1):
        _require_fields(item, ("title", "source", "published_at", "category", "url", "body"), f"语料第 {index} 条")
        fingerprint = _document_key(item)
        filename = f"doc-{index:05d}-{fingerprint[:12]}.md"
        metadata = [
            f"- Author: {item.get('author') or 'Unknown'}", f"- Source: {item['source']}", f"- Published at: {item['published_at']}",
            f"- Category: {item['category']}", f"- URL: {item['url']}",
        ]
        text = f"# {item['title']}\n\n## Metadata\n\n" + "\n".join(metadata) + f"\n\n## Body\n\n{item['body'].strip()}\n"
        (raw_dir / filename).write_text(text, encoding="utf-8")
        documents.append({"key": fingerprint, "source": filename})
        if on_progress:
            on_progress("转换语料", index, len(corpus), filename)
    return documents


def _build_manifest(queries: list[dict[str, Any]], corpus: list[dict[str, Any]], documents: list[dict[str, str]], on_progress: PrepareProgressCallback | None = None) -> dict[str, Any]:
    document_by_key = {item["key"]: item["source"] for item in documents}
    corpus_by_key = {_document_key(item): item for item in corpus}
    prepared_queries: list[dict[str, Any]] = []
    for index, item in enumerate(queries, start=1):
        _require_fields(item, ("query", "answer", "question_type"), f"问题第 {index} 条")
        if not isinstance(item["evidence_list"], list):
            raise ValueError(f"问题第 {index} 条的 evidence_list 必须是数组。")
        if not item["evidence_list"]:
            if item["question_type"] != "null_query":
                raise ValueError(f"问题第 {index} 条的 evidence_list 为空但 question_type 不是 null_query。")
            prepared_queries.append({"id": f"multihop-{index:04d}", "query": item["query"], "answer": item["answer"], "question_type": item["question_type"], "gold": []})
            if on_progress:
                on_progress("映射 Gold", index, len(queries), f"multihop-{index:04d}")
            continue
        gold: list[dict[str, str]] = []
        for evidence in item["evidence_list"]:
            _require_fields(evidence, ("title", "source", "published_at", "category", "url", "fact"), f"问题第 {index} 条证据")
            key = _document_key(evidence)
            corpus_item = corpus_by_key.get(key)
            if corpus_item is None:
                raise ValueError(f"问题第 {index} 条的 Gold 文档不在公开语料中：{evidence['title']}")
            fact = evidence["fact"]
            if _normalise(fact) not in _normalise(str(corpus_item["body"])):
                raise ValueError(f"问题第 {index} 条的 Gold fact 未出现在对应文档正文中：{fact[:80]}")
            gold.append({"source": document_by_key[key], "fact": fact, "fact_key": hashlib.sha256(f"{key}|{fact}".encode("utf-8")).hexdigest()[:20], "normalised_fact": _normalise(fact)})
        prepared_queries.append({"id": f"multihop-{index:04d}", "query": item["query"], "answer": item["answer"], "question_type": item["question_type"], "gold": gold})
        if on_progress:
            on_progress("映射 Gold", index, len(queries), f"multihop-{index:04d}")
    return {"queries": prepared_queries}


def _document_key(item: dict[str, Any]) -> str:
    identity = {name: item.get(name) for name in ("title", "author", "source", "published_at", "category", "url")}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"未找到 MultiHop-RAG 准备清单：{path}。请先运行 prepare-multihop。")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("benchmark") != BENCHMARK_NAME or not isinstance(data.get("queries"), list):
        raise ValueError(f"MultiHop-RAG 清单格式无效：{path}")
    return data


def _load_array(path: Path, name: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{name}必须是对象数组：{path}")
    return data


def _require_fields(item: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [name for name in fields if not isinstance(item.get(name), str) or not item[name].strip()]
    if missing:
        raise ValueError(f"{label}缺少非空字段：{', '.join(missing)}")


def _normalise(text: str) -> str:
    return "".join(text.split())


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "base-rag-multihoprag-adapter/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _remove_prepared_outputs(dataset_dir: Path, raw_dir: Path) -> None:
    dataset_root = dataset_dir.resolve()
    raw_root = raw_dir.resolve()
    if raw_root.name != "raw" or raw_root.parent != dataset_root:
        raise ValueError("为避免删除错误目录，MultiHop-RAG 语料目录必须是 dataset_dir/raw。")
    if raw_root.exists():
        shutil.rmtree(raw_root)
    manifest = dataset_root / MANIFEST_NAME
    if manifest.exists():
        manifest.unlink()


def _profile_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join([
        f"# {BENCHMARK_NAME} 检索评测报告", "", f"- Profile：`{result['profile']}`", "- 模式：仅检索；未调用 LLM 生成最终答案。", "",
        "| 完成/总题 | 失败 | Evidence Coverage@4 | Evidence Coverage@10 | Complete Evidence@4 | Complete Evidence@10 | Hits@4 | Hits@10 | MAP@10 | MRR@10 | 平均延迟(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['completed_questions']}/{metrics['questions']} | {metrics['failed_questions']} | {metrics['evidence_coverage_at_4']:.4f} | {metrics['evidence_coverage_at_10']:.4f} | {metrics['complete_evidence_at_4']:.4f} | {metrics['complete_evidence_at_10']:.4f} | {metrics['official_hits_at_4']:.4f} | {metrics['official_hits_at_10']:.4f} | {metrics['official_map_at_10']:.4f} | {metrics['official_mrr_at_10']:.4f} | {metrics['mean_latency_seconds']:.3f} |", "",
        "说明：Coverage/Complete Evidence 要求覆盖全部 Gold 事实；Hits/MAP/MRR 按公开仓库的字符串包含式检索口径计算。",
    ]) + "\n"


def _batch_report(summary: dict[str, Any]) -> str:
    lines = [f"# {BENCHMARK_NAME} 跨文档检索基准", "", f"- 可评测题数：{summary['query_count']}（跳过 null_query：{summary['skipped_null_queries']}）", "- 模式：仅检索；与本地中文题集完全分离。", "", "| Profile | 完成/总题 | 失败 | Coverage@4 | Coverage@10 | Complete@4 | Complete@10 | Hits@4 | Hits@10 | MAP@10 | MRR@10 | 平均延迟(s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary["profiles"]:
        metric = item["metrics"]
        lines.append(f"| {item['profile']} | {metric['completed_questions']}/{metric['questions']} | {metric['failed_questions']} | {metric['evidence_coverage_at_4']:.4f} | {metric['evidence_coverage_at_10']:.4f} | {metric['complete_evidence_at_4']:.4f} | {metric['complete_evidence_at_10']:.4f} | {metric['official_hits_at_4']:.4f} | {metric['official_hits_at_10']:.4f} | {metric['official_map_at_10']:.4f} | {metric['official_mrr_at_10']:.4f} | {metric['mean_latency_seconds']:.3f} |")
    return "\n".join(lines) + "\n"
