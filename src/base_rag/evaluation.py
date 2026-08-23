from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import yaml

from base_rag.config import AppConfig
from base_rag.pipeline import Embedder, Generator, PipelineStageError, ask
from base_rag.store import FaissStore


EXPERIMENT_PROFILES = ("dense", "bm25", "hybrid", "hybrid-rerank", "advanced")


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        raise ValueError("评测 YAML 必须包含 questions 数组。")
    for item in questions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str):
            raise ValueError("每题必须包含字符串 id 和 question。")
    return questions


ProgressCallback = Callable[[str, int, int, int, int, str], None]


def evaluate(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    questions_path: Path,
    profile: str,
    generate: bool = True,
    checkpoint_path: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    _validate_evidence_annotations(config, questions)
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(questions, start=1):
        try:
            result = ask(config, embedder, generator, entry["question"], profile=profile, generate=generate)
            records.append(_score_entry(entry, result))
        except Exception as exc:
            records.append(_failed_entry(entry, exc))
        snapshot = _evaluation_snapshot(profile, questions_path, records, complete=index == len(questions))
        if checkpoint_path:
            _write_json(checkpoint_path, snapshot)
        if on_progress:
            on_progress(profile, index, len(questions), _success_count(records), _failure_count(records), entry["id"])
    return _evaluation_snapshot(profile, questions_path, records, complete=True)


def _validate_evidence_annotations(config: AppConfig, questions: list[dict[str, Any]]) -> None:
    """Fail before a paid evaluation if frozen evidence no longer maps to this index."""
    if not any(entry.get("relevant") for entry in questions):
        return
    store = FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    for entry in questions:
        for evidence in entry.get("relevant", []):
            if not isinstance(evidence, dict):
                raise ValueError(f"{entry['id']} 的 relevant 必须是对象。")
            source, text = evidence.get("source"), evidence.get("evidence_contains")
            if not isinstance(source, str) or not isinstance(text, str):
                raise ValueError(f"{entry['id']} 的 relevant 必须包含 source 和 evidence_contains。")
            if not any(source in Path(chunk.source_path).name and text in chunk.text for chunk in store.chunks):
                raise ValueError(f"{entry['id']} 的证据未映射到当前 Chunk：{source} / {text}")


def run_experiment(config: AppConfig, embedder: Embedder, generator: Generator, questions_path: Path) -> Path:
    questions = load_questions(questions_path)
    _validate_evidence_annotations(config, questions)
    root = config.paths.runs_dir / "experiments" / datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "experiment.json", {"status": "in_progress", "questions_path": str(questions_path), "profiles": list(EXPERIMENT_PROFILES), "config": config.safe_dict()})
    results: dict[str, Any] = {}
    for profile in EXPERIMENT_PROFILES:
        print(f"\n[{profile}] 开始", flush=True)
        results[profile] = evaluate(
            config,
            embedder,
            generator,
            questions_path,
            profile,
            generate=profile in {"dense", "advanced"},
            checkpoint_path=root / f"{profile}.json",
            on_progress=_print_progress,
        )
    if config.evaluation.judge_enabled:
        for profile in ("dense", "advanced"):
            for record in results[profile]["records"]:
                record["llm_judge"] = _judge_answer(generator, record, config.evaluation.judge_max_tokens)
    for profile in EXPERIMENT_PROFILES:
        _write_json(root / f"{profile}.json", results[profile])
    _write_json(root / "summary.json", {"profiles": results, "comparisons": _comparisons(results)})
    _write_review_csv(root / "human_review.csv", results)
    (root / "REPORT.md").write_text(_report(results, reviewed=False), encoding="utf-8")
    _write_json(root / "experiment.json", {"status": "complete", "questions_path": str(questions_path), "profiles": list(EXPERIMENT_PROFILES), "config": config.safe_dict()})
    return root


def _evaluation_snapshot(profile: str, questions_path: Path, records: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    return {"profile": profile, "questions_path": str(questions_path), "status": "complete" if complete else "in_progress", "records": records, "metrics": _metrics(records)}


def _failed_entry(entry: dict[str, Any], exc: Exception) -> dict[str, Any]:
    stage = exc.stage if isinstance(exc, PipelineStageError) else "unknown"
    answerable = entry.get("answerable", True)
    return {
        "id": entry["id"], "question": entry["question"], "category": entry.get("category", "legacy"), "split": entry.get("split", "legacy"), "answerable": answerable,
        "expected_sources": _expected_sources(entry), "expected_facts": entry.get("expected_facts", []), "status": "failed", "failure": {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)},
        "result": {"elapsed_seconds": 0.0, "stage_calls": {"embedding": 0, "rewrite": 0, "rerank": 0, "generation": 0}}, "final_sources": [], "candidate_sources": [],
        "source_recall_at_6": 0.0 if answerable else None, "chunk_recall_at_6": 0.0 if answerable else None, "chunk_recall_at_20": 0.0 if answerable else None,
        "mrr_at_6": 0.0 if answerable else None, "ndcg_at_6": 0.0 if answerable else None, "unanswerable_retrieved": False if not answerable else None,
    }


def _success_count(records: list[dict[str, Any]]) -> int:
    return sum(record.get("status") != "failed" for record in records)


def _failure_count(records: list[dict[str, Any]]) -> int:
    return sum(record.get("status") == "failed" for record in records)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _print_progress(profile: str, completed: int, total: int, successes: int, failures: int, question_id: str) -> None:
    width = 24
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{profile:<14}] [{bar}] {completed:>3}/{total} 成功:{successes} 失败:{failures} 当前:{question_id}", end="", flush=True)
    if completed == total:
        print(flush=True)


def rebuild_report(experiment_dir: Path, reviews_path: Path) -> Path:
    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    reviews = list(csv.DictReader(reviews_path.read_text(encoding="utf-8-sig").splitlines()))
    required = [row for row in reviews if row.get("review_required") == "yes"]
    incomplete = [row["review_id"] for row in required if not row.get("human_correctness")]
    if incomplete:
        raise ValueError(f"人工复核尚未完成：{', '.join(incomplete[:8])}")
    report = experiment_dir / "REPORT.reviewed.md"
    report.write_text(_report(summary["profiles"], reviewed=True), encoding="utf-8")
    return report


def _score_entry(entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_sources(entry)
    annotations = entry.get("relevant", [])
    answerable = entry.get("answerable", True)
    final = result["retrieval"]["final"]
    candidates = result["retrieval"]["candidates"]
    final_sources = [_source_name(hit) for hit in final]
    candidate_sources = [_source_name(hit) for hit in candidates]
    source_relevant_final = [_is_relevant_source(hit, expected) for hit in final]
    chunk_relevant_final = [_is_relevant_chunk(hit, annotations, expected) for hit in final]
    source_rank = next((index for index, value in enumerate(source_relevant_final, start=1) if value), None)
    chunk_rank = next((index for index, value in enumerate(chunk_relevant_final, start=1) if value), None)
    return {
        "id": entry["id"],
        "status": "ok",
        "question": entry["question"],
        "category": entry.get("category", "legacy"),
        "split": entry.get("split", "legacy"),
        "answerable": answerable,
        "expected_sources": expected,
        "expected_facts": entry.get("expected_facts", []),
        "result": result,
        "final_sources": final_sources,
        "candidate_sources": candidate_sources,
        "source_recall_at_6": bool(source_rank) if answerable else None,
        "chunk_recall_at_6": bool(chunk_rank) if answerable else None,
        "chunk_recall_at_20": any(_is_relevant_chunk(hit, annotations, expected) for hit in candidates[:20]) if answerable else None,
        "mrr_at_6": (1 / chunk_rank if chunk_rank else 0.0) if answerable else None,
        "ndcg_at_6": _ndcg(chunk_relevant_final) if answerable else None,
        "unanswerable_retrieved": bool(final) if not answerable else None,
    }


def _expected_sources(entry: dict[str, Any]) -> list[str]:
    relevant = entry.get("relevant")
    if isinstance(relevant, list):
        return [item["source"] for item in relevant if isinstance(item, dict) and isinstance(item.get("source"), str)]
    expected = entry.get("expected_source")
    return [expected] if isinstance(expected, str) else []


def _source_name(hit: dict[str, Any]) -> str:
    return Path(hit["chunk"]["source_path"]).name


def _is_relevant_source(hit: dict[str, Any], expected_sources: list[str]) -> bool:
    source = _source_name(hit)
    return any(expected in source for expected in expected_sources)


def _is_relevant_chunk(hit: dict[str, Any], annotations: list[dict[str, Any]], expected_sources: list[str]) -> bool:
    if not annotations:  # legacy question sets only have a source-level label
        return _is_relevant_source(hit, expected_sources)
    source, text = _source_name(hit), hit["chunk"]["text"]
    return any(item["source"] in source and item["evidence_contains"] in text for item in annotations)


def _ndcg(relevance: list[bool]) -> float:
    if not relevance:
        return 0.0
    dcg = sum(1 / __import__("math").log2(index + 1) for index, value in enumerate(relevance, start=1) if value)
    ideal_count = sum(relevance)
    ideal = sum(1 / __import__("math").log2(index + 1) for index in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [record for record in records if record["answerable"]]
    metrics = {
        "questions": len(records),
        "answerable_questions": len(answerable),
        "source_recall_at_6": _average(answerable, "source_recall_at_6"),
        "chunk_recall_at_6": _average(answerable, "chunk_recall_at_6"),
        "chunk_recall_at_20": _average(answerable, "chunk_recall_at_20"),
        "mrr_at_6": _average(answerable, "mrr_at_6"),
        "ndcg_at_6": _average(answerable, "ndcg_at_6"),
        "mean_latency_seconds": round(mean(record["result"]["elapsed_seconds"] for record in records), 3) if records else 0.0,
        "stage_calls": {stage: sum(record["result"]["stage_calls"].get(stage, 0) for record in records) for stage in ("embedding", "rewrite", "rerank", "generation")},
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in answerable:
        by_category[record["category"]].append(record)
    metrics["by_category"] = {category: {key: _average(items, key) for key in ("source_recall_at_6", "mrr_at_6", "ndcg_at_6")} for category, items in by_category.items()}
    return metrics


def _average(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record[key] is not None]
    return round(mean(values), 4) if values else 0.0


def _comparisons(results: dict[str, Any]) -> dict[str, Any]:
    pairs = {"hybrid_vs_dense": ("hybrid", "dense"), "hybrid_vs_bm25": ("hybrid", "bm25"), "rerank_vs_hybrid": ("hybrid-rerank", "hybrid"), "rewrite_vs_rerank": ("advanced", "hybrid-rerank")}
    keys = ("source_recall_at_6", "mrr_at_6", "ndcg_at_6")
    return {name: {key: round(results[left]["metrics"][key] - results[right]["metrics"][key], 4) for key in keys} for name, (left, right) in pairs.items()}


def _write_review_csv(path: Path, results: dict[str, Any]) -> None:
    rng = random.Random(20260823)
    dense = {record["id"]: record for record in results["dense"]["records"]}
    advanced = {record["id"]: record for record in results["advanced"]["records"]}
    rows, key = [], {}
    for question_id in dense:
        left, right = ("dense", "advanced") if rng.random() < 0.5 else ("advanced", "dense")
        left_record = {"dense": dense, "advanced": advanced}[left][question_id]
        right_record = {"dense": dense, "advanced": advanced}[right][question_id]
        left_judge = left_record.get("llm_judge", {})
        right_judge = right_record.get("llm_judge", {})
        needs_review = (not left_record["answerable"] or left_judge.get("correctness", 0) < 2 or left_judge.get("groundedness", 0) < 2 or rng.random() < 0.2)
        key[question_id] = {"answer_a": left, "answer_b": right}
        rows.append({
            "review_id": question_id,
            "question": left_record["question"],
            "expected_facts": " | ".join(left_record["expected_facts"]),
            "answer_a": left_record["result"].get("answer") or "",
            "answer_b": right_record["result"].get("answer") or "",
            "review_required": "yes" if needs_review else "no",
            "human_correctness": "",
            "human_groundedness": "",
            "notes": "",
        })
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["review_id"])
        writer.writeheader()
        writer.writerows(rows)
    (path.parent / "review_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_answer(generator: Generator, record: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    if record.get("status") == "failed":
        return {"status": "skipped", "reason": "question_failed_before_answer"}
    try:
        prompt = f"""你是严格的 RAG 答案评审。只输出 JSON，不要解释。\n问题：{record['question']}\n可验证事实点：{record['expected_facts']}\n允许引用来源：{record['expected_sources']}\n回答：{record['result'].get('answer')}\n请给 correctness、completeness、groundedness、citation_correctness 四个 0-2 整数；不可回答题再给 refusal 0-2；并给 rationale 字符串。不得根据外部常识补全。"""
        value = json.loads(generator.generate(prompt, 0, max_tokens).strip())
        for key in ("correctness", "completeness", "groundedness", "citation_correctness"):
            if not isinstance(value.get(key), int) or value[key] not in (0, 1, 2):
                raise ValueError(f"{key} 非法")
        return value
    except Exception as exc:
        return {"status": "judge_error", "error": str(exc)}


def _report(results: dict[str, Any], reviewed: bool) -> str:
    lines = ["# Advanced RAG 消融实验报告", "", f"答案人工复核：{'已完成' if reviewed else '初评待人工复核'}", "", "| Profile | Source Recall@6 | MRR@6 | nDCG@6 | 平均延迟(s) |", "|---|---:|---:|---:|---:|"]
    for name in EXPERIMENT_PROFILES:
        metric = results[name]["metrics"]
        lines.append(f"| {name} | {metric['source_recall_at_6']:.4f} | {metric['mrr_at_6']:.4f} | {metric['ndcg_at_6']:.4f} | {metric['mean_latency_seconds']:.3f} |")
    lines += ["", "结论仅根据 summary.json 的预注册比较 Delta 判定；正值不等于统计显著，负值也必须保留。"]
    return "\n".join(lines) + "\n"
