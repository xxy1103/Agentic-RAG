"""Evaluation routines and stratified dataset split for Phase 3 Agentic Multi-Hop RAG."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from base_rag.agentic import run_agentic_retrieval
from base_rag.bm25 import BM25Store
from base_rag.config import AppConfig
from base_rag.models import AgenticRetrievalResult, SearchHit
from base_rag.multihoprag import (
    BENCHMARK_NAME,
    MANIFEST_NAME,
    _coverage,
    _hit_matches_any,
    _hit_matches_any_in,
    _hit_matches_fact,
    _rate,
    _score_query,
    require_multihoprag_config,
)
from base_rag.pipeline import Embedder, Generator, PipelineStageError, ask
from base_rag.rerank import Reranker
from base_rag.store import FaissStore

SPLITS_FILENAME = "splits_dev_test.json"
ProgressCallback = Callable[[str, int, int, int, int, str], None]


def get_or_create_multihop_splits(
    dataset_dir: Path,
    split_file: Path | None = None,
    seed: int = 42,
    dev_count_per_type: int = 20,
    test_count_per_type: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"未找到 MultiHop-RAG 清单：{manifest_path}。请先运行 prepare-multihop。")
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = manifest_data.get("queries", [])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    save_path = split_file or (dataset_dir / SPLITS_FILENAME)
    if save_path.is_file():
        saved = json.loads(save_path.read_text(encoding="utf-8"))
        if saved.get("manifest_sha256") == manifest_hash and saved.get("seed") == seed:
            query_map = {q["id"]: q for q in queries}
            return {
                "dev": [query_map[qid] for qid in saved["dev_ids"] if qid in query_map],
                "test": [query_map[qid] for qid in saved["test_ids"] if qid in query_map],
            }

    valid_types = ("inference_query", "comparison_query", "temporal_query")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        if q.get("question_type") in valid_types:
            by_type[q["question_type"]].append(q)

    rng = random.Random(seed)
    dev_ids: list[str] = []
    test_ids: list[str] = []
    dev_queries: list[dict[str, Any]] = []
    test_queries: list[dict[str, Any]] = []

    for qtype in valid_types:
        category_queries = sorted(by_type[qtype], key=lambda x: x["id"])
        needed = dev_count_per_type + test_count_per_type
        if len(category_queries) < needed:
            raise ValueError(f"类别 {qtype} 样本数不足（现有 {len(category_queries)}，需要 {needed}）。")
        sampled = rng.sample(category_queries, needed)
        dev_sample = sampled[:dev_count_per_type]
        test_sample = sampled[dev_count_per_type:needed]
        for item in dev_sample:
            dev_ids.append(item["id"])
            dev_queries.append(item)
        for item in test_sample:
            test_ids.append(item["id"])
            test_queries.append(item)

    split_payload = {
        "benchmark": BENCHMARK_NAME,
        "manifest_sha256": manifest_hash,
        "seed": seed,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "counts": {
            "dev": len(dev_ids),
            "test": len(test_ids),
            "per_type_dev": dev_count_per_type,
            "per_type_test": test_count_per_type,
        },
    }
    save_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dev": dev_queries, "test": test_queries}


def score_agentic_query(query: dict[str, Any], result: AgenticRetrievalResult) -> dict[str, Any]:
    hits = [h.to_dict() for h in result.final_hits]
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
        "id": query["id"],
        "query": query["query"],
        "answer": query["answer"],
        "question_type": query["question_type"],
        "status": "ok",
        "gold_count": len(gold),
        "total_hops": result.total_hops,
        "correction_count": result.correction_count,
        "termination_reason": result.termination_reason,
        "route": result.route_decision.route,
        "elapsed_seconds": result.elapsed_seconds,
        "stage_calls": result.stage_calls,
        "result": result.to_dict(),
        "evidence_coverage_at_4": coverage_at_4,
        "evidence_coverage_at_10": coverage_at_10,
        "complete_evidence_at_4": coverage_at_4 == 1.0,
        "complete_evidence_at_10": coverage_at_10 == 1.0,
        "official_hits_at_4": _hit_matches_any_in(hits[:4], gold),
        "official_hits_at_10": _hit_matches_any_in(hits[:10], gold),
        "official_map_at_10": precision_sum / min(len(gold), 10) if gold else 0.0,
        "official_mrr_at_10": 1 / first_rank if first_rank else 0.0,
    }


def evaluate_agentic_multihop(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    *,
    split: str = "dev",
    system: str = "both",
    reranker: Reranker | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    if split not in {"dev", "test"}:
        raise ValueError("split 只能是 'dev' 或 'test'。")
    if system not in {"baseline", "agentic", "both"}:
        raise ValueError("system 只能是 'baseline'、'agentic' 或 'both'。")

    dataset_dir = require_multihoprag_config(config)
    splits = get_or_create_multihop_splits(dataset_dir)
    queries = splits[split]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    batch_dir = config.paths.runs_dir / "agentic_evaluations" / f"{split}-{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)

    dense_store = FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    bm25_store = BM25Store.load(config.paths.index_dir, dense_store.chunks, dense_store.metadata.corpus_hash)

    systems_to_run = ["baseline", "agentic"] if system == "both" else [system]
    eval_results: dict[str, Any] = {}

    for sys_name in systems_to_run:
        sys_dir = batch_dir / sys_name
        sys_dir.mkdir(parents=True, exist_ok=False)
        questions_dir = sys_dir / "questions"
        questions_dir.mkdir(parents=True, exist_ok=False)

        checkpoint_file = sys_dir / "checkpoint.json"
        records_map: dict[str, dict[str, Any]] = {}
        if checkpoint_file.is_file():
            try:
                saved_records = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                for rec in saved_records:
                    records_map[rec["id"]] = rec
            except Exception:
                records_map = {}

        pending_queries = [q for q in queries if q["id"] not in records_map]
        concurrency = max(1, min(config.evaluation.concurrency, len(queries)))

        def _run_one(q: dict[str, Any]) -> dict[str, Any]:
            try:
                if sys_name == "baseline":
                    res = ask(
                        config=config,
                        embedder=embedder,
                        generator=generator,
                        question=q["query"],
                        profile=config.agentic.base_profile,
                        reranker=reranker,
                        generate=False,
                        dense_store=dense_store,
                        bm25_store=bm25_store,
                        run_log_dir=questions_dir,
                    )
                    return _score_query(q, res)
                else:
                    res_agentic = run_agentic_retrieval(
                        config=config,
                        embedder=embedder,
                        generator=generator,
                        question=q["query"],
                        reranker=reranker,
                        dense_store=dense_store,
                        bm25_store=bm25_store,
                        run_log_dir=questions_dir,
                    )
                    return score_agentic_query(q, res_agentic)
            except Exception as exc:
                stage = exc.stage if isinstance(exc, PipelineStageError) else "unknown"
                return {
                    "id": q["id"],
                    "query": q["query"],
                    "question_type": q["question_type"],
                    "status": "failed",
                    "failure": {
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }

        if pending_queries:
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"eval-{sys_name}") as executor:
                futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
                    executor.submit(_run_one, q): q for q in pending_queries
                }
                for count, future in enumerate(as_completed(futures), start=1):
                    rec = future.result()
                    records_map[rec["id"]] = rec
                    checkpoint_file.write_text(
                        json.dumps(list(records_map.values()), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if on_progress:
                        completed_list = list(records_map.values())
                        ok_cnt = sum(r["status"] == "ok" for r in completed_list)
                        fail_cnt = sum(r["status"] == "failed" for r in completed_list)
                        on_progress(sys_name, len(records_map), len(queries), ok_cnt, fail_cnt, rec["id"])

        final_records = [records_map[q["id"]] for q in queries if q["id"] in records_map]
        sys_metrics = _calc_system_metrics(final_records, is_agentic=(sys_name == 'agentic'))
        sys_result = {
            "system": sys_name,
            "split": split,
            "metrics": sys_metrics,
            "records": final_records,
        }
        (sys_dir / "summary.json").write_text(json.dumps(sys_result, ensure_ascii=False, indent=2), encoding="utf-8")
        eval_results[sys_name] = sys_result

    paired_analysis = None
    if system == "both" and "baseline" in eval_results and "agentic" in eval_results:
        paired_analysis = _calc_paired_deltas(eval_results["baseline"]["records"], eval_results["agentic"]["records"])

    summary_payload = {
        "kind": "multihoprag_agentic_evaluation",
        "benchmark": BENCHMARK_NAME,
        "split": split,
        "system": system,
        "query_count": len(queries),
        "config": config.safe_dict(),
        "results": {k: v["metrics"] for k, v in eval_results.items()},
        "paired_analysis": paired_analysis,
    }
    (batch_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_text = _generate_markdown_report(summary_payload, eval_results, paired_analysis)
    (batch_dir / "REPORT.md").write_text(report_text, encoding="utf-8")

    return batch_dir, summary_payload


def _calc_system_metrics(records: list[dict[str, Any]], is_agentic: bool = False) -> dict[str, Any]:
    completed = [r for r in records if r.get("status") == "ok"]
    metrics: dict[str, Any] = {
        "questions": len(records),
        "completed_questions": len(completed),
        "failed_questions": len(records) - len(completed),
    }
    metric_names = (
        "evidence_coverage_at_4",
        "evidence_coverage_at_10",
        "official_hits_at_4",
        "official_hits_at_10",
        "official_map_at_10",
        "official_mrr_at_10",
    )
    metrics.update({name: round(mean(r[name] for r in completed), 4) if completed else 0.0 for name in metric_names})
    metrics["complete_evidence_at_4"] = _rate(completed, "complete_evidence_at_4")
    metrics["complete_evidence_at_10"] = _rate(completed, "complete_evidence_at_10")
    metrics["mean_latency_seconds"] = round(mean(r.get("elapsed_seconds", 0.0) for r in completed), 3) if completed else 0.0

    if is_agentic and completed:
        metrics["mean_hops"] = round(mean(r.get("total_hops", 1) for r in completed), 2)
        metrics["correction_rate"] = round(sum(r.get("correction_count", 0) > 0 for r in completed) / len(completed), 4)
        metrics["termination_reasons"] = dict(Counter(r.get("termination_reason", "unknown") for r in completed))
        metrics["routes"] = dict(Counter(r.get("route", "single_hop") for r in completed))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        groups[record["question_type"]].append(record)
    metrics["by_question_type"] = {
        kind: {
            "evidence_coverage_at_10": round(mean(item["evidence_coverage_at_10"] for item in items), 4),
            "complete_evidence_at_10": round(sum(item["complete_evidence_at_10"] for item in items) / len(items), 4),
            "official_mrr_at_10": round(mean(item["official_mrr_at_10"] for item in items), 4),
            "count": len(items),
        }
        for kind, items in groups.items()
    }
    return metrics


def _calc_paired_deltas(baseline_records: list[dict[str, Any]], agentic_records: list[dict[str, Any]]) -> dict[str, Any]:
    b_map = {r["id"]: r for r in baseline_records if r.get("status") == "ok"}
    a_map = {r["id"]: r for r in agentic_records if r.get("status") == "ok"}
    common_ids = [qid for qid in b_map if qid in a_map]
    if not common_ids:
        return {"error": "无成对有效记录进行差异对比"}

    fields = [
        ("evidence_coverage_at_10", "float"),
        ("complete_evidence_at_10", "bool"),
        ("official_mrr_at_10", "float"),
        ("evidence_coverage_at_4", "float"),
        ("complete_evidence_at_4", "bool"),
    ]
    deltas: dict[str, Any] = {"common_questions": len(common_ids)}

    for field_name, ftype in fields:
        diffs = []
        for qid in common_ids:
            b_val = float(b_map[qid][field_name])
            a_val = float(a_map[qid][field_name])
            diffs.append(a_val - b_val)
        m_diff = mean(diffs)
        sd_diff = stdev(diffs) if len(diffs) > 1 else 0.0
        se = sd_diff / math.sqrt(len(diffs)) if len(diffs) > 0 else 0.0
        ci_lower = m_diff - 1.96 * se
        ci_upper = m_diff + 1.96 * se
        t_stat = m_diff / se if se > 0 else 0.0
        p_val = 2 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))
        deltas[field_name] = {
            "mean_delta": round(m_diff, 4),
            "std_error": round(se, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "p_value": round(p_val, 4),
            "improved_count": sum(d > 0 for d in diffs),
            "degraded_count": sum(d < 0 for d in diffs),
            "tied_count": sum(d == 0 for d in diffs),
        }
    return deltas


def _generate_markdown_report(
    summary: dict[str, Any],
    eval_results: dict[str, Any],
    paired_analysis: dict[str, Any] | None,
) -> str:
    split_name = summary["split"]
    query_count = summary["query_count"]
    system_name = summary["system"]
    base_prof = summary["config"]["agentic"]["base_profile"]
    lines = [
        "# MultiHop-RAG Phase 3 Agentic 检索评测报告",
        "",
        f"- **评测划分**：`{split_name}` ({query_count} 题)",
        f"- **运行模式**：`{system_name}`",
        f"- **基座 Profile**：`{base_prof}`",
        "",
        "## 1. 系统指标概览",
        "",
        "| 系统 | 完成/总题 | Coverage@4 | Coverage@10 | Complete@4 | Complete@10 | Hits@10 | MRR@10 | 平均延迟(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sys_name, res in eval_results.items():
        m = res["metrics"]
        cov4 = m["evidence_coverage_at_4"]
        cov10 = m["evidence_coverage_at_10"]
        comp4 = m["complete_evidence_at_4"]
        comp10 = m["complete_evidence_at_10"]
        hits10 = m["official_hits_at_10"]
        mrr10 = m["official_mrr_at_10"]
        lat = m["mean_latency_seconds"]
        done = m["completed_questions"]
        tot = m["questions"]
        lines.append(
            f"| {sys_name} | {done}/{tot} | "
            f"{cov4:.4f} | {cov10:.4f} | "
            f"{comp4:.4f} | {comp10:.4f} | "
            f"{hits10:.4f} | {mrr10:.4f} | "
            f"{lat:.3f} |"
        )

    if paired_analysis and "common_questions" in paired_analysis:
        cq = paired_analysis["common_questions"]
        lines.extend([
            "",
            "## 2. 配对差异分析 (Agentic vs Baseline)",
            "",
            f"成对比较题目数：{cq}",
            "",
            "| 指标 | 平均增量 (Delta) | 95% 置信区间 | p-value | 提升/持平/下降 |",
            "|---|---:|:---:|---:|:---:|",
        ])
        for name in ("complete_evidence_at_10", "evidence_coverage_at_10", "official_mrr_at_10", "complete_evidence_at_4", "evidence_coverage_at_4"):
            if name in paired_analysis:
                item = paired_analysis[name]
                ci = f"[{item['ci_95'][0]:+.4f}, {item['ci_95'][1]:+.4f}]"
                counts = f"{item['improved_count']} / {item['tied_count']} / {item['degraded_count']}"
                lines.append(f"| {name} | {item['mean_delta']:+.4f} | {ci} | {item['p_value']:.4f} | {counts} |")

    if "agentic" in eval_results:
        am = eval_results["agentic"]["metrics"]
        hops = am.get("mean_hops", 1.0)
        corr = am.get("correction_rate", 0.0) * 100
        routes_str = json.dumps(am.get("routes", {}), ensure_ascii=False)
        term_str = json.dumps(am.get("termination_reasons", {}), ensure_ascii=False)
        lines.extend([
            "",
            "## 3. Agentic 过程指标",
            "",
            f"- **平均跳数**：{hops}",
            f"- **纠错率**：{corr:.1f}%",
            f"- **路由分布**：{routes_str}",
            f"- **终止原因分布**：{term_str}",
        ])

    lines.append("")
    return "\n".join(lines)
