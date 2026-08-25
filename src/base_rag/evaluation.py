from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import yaml

from base_rag.bm25 import BM25Store
from base_rag.config import AppConfig
from base_rag.pipeline import Embedder, Generator, PROFILES, PipelineStageError, ask
from base_rag.store import FaissStore


EVALUATION_CATEGORIES = {"lexical", "semantic", "multi_evidence", "ambiguous", "unanswerable"}
_REFUSAL_PREFIX = "证据不足，无法基于已检索文档回答"


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        raise ValueError("评测 YAML 必须包含 questions 数组。")
    seen_ids: set[str] = set()
    for item in questions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str):
            raise ValueError("每题必须包含字符串 id 和 question。")
        question_id = item["id"]
        if question_id in seen_ids:
            raise ValueError(f"评测题 id 重复：{question_id}")
        seen_ids.add(question_id)
        if item.get("category") not in EVALUATION_CATEGORIES:
            raise ValueError(f"{question_id} 的 category 非法。")
        answerable = item.get("answerable")
        relevant, expected_facts = item.get("relevant"), item.get("expected_facts")
        if not isinstance(answerable, bool) or not isinstance(relevant, list) or not isinstance(expected_facts, list) or not all(isinstance(fact, str) for fact in expected_facts):
            raise ValueError(f"{question_id} 必须包含布尔 answerable、relevant 数组和字符串 expected_facts 数组。")
        if answerable and (not relevant or not expected_facts):
            raise ValueError(f"{question_id} 是可回答题，必须标注 relevant 和 expected_facts。")
        for evidence in relevant:
            if not isinstance(evidence, dict) or not all(isinstance(evidence.get(key), str) and evidence[key] for key in ("source", "section", "evidence_contains")):
                raise ValueError(f"{question_id} 的 relevant 必须包含非空 source、section 和 evidence_contains。")
        if not answerable and (relevant or expected_facts or item["category"] != "unanswerable"):
            raise ValueError(f"{question_id} 是无答案题，必须使用 unanswerable 类别且不得标注证据或事实点。")
    return questions


ProgressCallback = Callable[[str, int, int, int, int, str], None]
_STAGE_LABELS = {
    "rewrite": "问题改写",
    "index_load": "加载向量索引",
    "bm25_load": "加载 BM25 索引",
    "dense_retrieval": "Dense 检索",
    "bm25_retrieval": "BM25 检索",
    "rrf_fusion": "RRF 融合",
    "rerank": "Rerank 精排",
    "generation": "答案生成",
}


def evaluate(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    questions_path: Path,
    profile: str,
    generate: bool = True,
    checkpoint_path: Path | None = None,
    on_progress: ProgressCallback | None = None,
    run_log_dir: Path | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    dense_store = FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    _validate_evidence_annotations(config, questions, store=dense_store)
    selected = PROFILES[profile]
    bm25_store = BM25Store.load(config.paths.index_dir, dense_store.chunks, dense_store.metadata.corpus_hash) if selected.bm25 else None
    concurrency = max(1, min(config.evaluation.concurrency, len(questions)))
    completed_records: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"eval-{profile}") as executor:
        futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {
            executor.submit(
                _evaluate_entry,
                config,
                embedder,
                generator,
                entry,
                profile,
                generate,
                index,
                len(questions),
                concurrency == 1,
                dense_store,
                bm25_store,
                run_log_dir,
            ): (index, entry)
            for index, entry in enumerate(questions)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, entry = futures[future]
            completed_records[index] = future.result()
            records = [completed_records[item] for item in sorted(completed_records)]
            snapshot = _evaluation_snapshot(profile, questions_path, records, complete=completed == len(questions), generate=generate)
            if checkpoint_path:
                _write_json(checkpoint_path, snapshot)
            if on_progress:
                on_progress(profile, completed, len(questions), _success_count(records), _failure_count(records), entry["id"])

    records = [completed_records[index] for index in range(len(questions))]
    return _evaluation_snapshot(profile, questions_path, records, complete=True, generate=generate)


def create_evaluation_artifact_dir(config: AppConfig, profile: str) -> Path:
    """Create the directory for one complete evaluation and all question logs."""
    if profile not in PROFILES:
        raise ValueError("评测产物必须包含合法的 profile。")
    root = config.paths.runs_dir / profile / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    root.mkdir(parents=True, exist_ok=False)
    return root


def create_retrieval_evaluation_batch_dir(config: AppConfig) -> Path:
    """Create one shared artifact root for all retrieval-ablation profiles."""
    root = config.paths.runs_dir / "retrieval-evaluations" / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    root.mkdir(parents=True, exist_ok=False)
    return root


def write_evaluation_artifact(config: AppConfig, result: dict[str, Any], artifact_dir: Path | None = None) -> Path:
    """Persist one complete evaluation beside its per-question run records."""
    profile = result.get("profile")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("评测产物必须包含合法的 profile。")
    root = artifact_dir or create_evaluation_artifact_dir(config, profile)
    root.mkdir(parents=True, exist_ok=True)
    kind = "retrieval_evaluation" if result.get("mode") == "retrieval" else "evaluation"
    _write_json(root / "summary.json", {"kind": kind, "config": config.safe_dict(), "evaluation": result})
    (root / "REPORT.md").write_text(_evaluation_report(result), encoding="utf-8")
    return root


def _evaluate_entry(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    entry: dict[str, Any],
    profile: str,
    generate: bool,
    question_index: int,
    question_total: int,
    show_stage_progress: bool,
    dense_store: FaissStore,
    bm25_store: BM25Store | None,
    run_log_dir: Path | None,
) -> dict[str, Any]:
    stage_progress = _stage_progress(profile, generate, question_index + 1, question_total, entry["id"], enabled=show_stage_progress)
    try:
        result = ask(
            config,
            embedder,
            generator,
            entry["question"],
            profile=profile,
            generate=generate,
            on_stage=stage_progress,
            dense_store=dense_store,
            bm25_store=bm25_store,
            run_log_dir=run_log_dir,
        )
        return _score_entry(entry, result)
    except Exception as exc:
        return _failed_entry(entry, exc)
    finally:
        stage_progress.finish()


def _source_section_text(source_path: Path, section: str) -> str:
    current_section: str | None = None
    current_lines: list[str] = []
    sections: list[tuple[str | None, str]] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_section, text))
            current_section, current_lines = match.group(1), []
        else:
            current_lines.append(line)
    text = "\n".join(current_lines).strip()
    if text:
        sections.append((current_section, text))
    matches = [text for heading, text in sections if heading == section]
    if len(matches) != 1:
        raise ValueError(f"{source_path.name} 的 section 必须唯一且存在：{section}")
    return matches[0]


def _validate_evidence_annotations(config: AppConfig, questions: list[dict[str, Any]], store: FaissStore | None = None) -> None:
    """Fail before a paid evaluation if frozen evidence no longer maps to this index."""
    if not any(entry.get("relevant") for entry in questions):
        return
    store = store or FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions)
    for entry in questions:
        for evidence in entry.get("relevant", []):
            if not isinstance(evidence, dict):
                raise ValueError(f"{entry['id']} 的 relevant 必须是对象。")
            source, section, text = evidence.get("source"), evidence.get("section"), evidence.get("evidence_contains")
            if not all(isinstance(value, str) and value for value in (source, section, text)):
                raise ValueError(f"{entry['id']} 的 relevant 必须包含非空 source、section 和 evidence_contains。")
            if Path(source).name != source:
                raise ValueError(f"{entry['id']} 的 source 必须是完整文件名：{source}")
            source_path = config.paths.corpus_dir / source
            if not source_path.is_file():
                raise ValueError(f"{entry['id']} 的 source 不在当前语料中：{source}")
            section_text = _source_section_text(source_path, section)
            if section_text.count(text) != 1:
                raise ValueError(f"{entry['id']} 的证据摘录必须在来源区段中唯一：{source} / {section} / {text}")
            matching_chunks = [chunk for chunk in store.chunks if source == Path(chunk.source_path).name and section == chunk.section and text in chunk.text]
            if not matching_chunks:
                raise ValueError(f"{entry['id']} 的证据未映射到当前 Chunk：{source} / {section} / {text}")
            evidence["gold_chunk_ids"] = sorted({chunk.chunk_id for chunk in matching_chunks})


def write_retrieval_evaluation_batch(config: AppConfig, questions_path: Path, results: dict[str, dict[str, Any]], batch_dir: Path) -> Path:
    """Persist a comparable all-profile retrieval-only evaluation batch."""
    profiles = [{"profile": profile, "metrics": result["metrics"], "artifact_dir": f"profiles/{profile}"} for profile, result in results.items()]
    _write_json(
        batch_dir / "summary.json",
        {
            "kind": "retrieval_evaluation_batch",
            "mode": "retrieval",
            "config": config.safe_dict(),
            "questions_path": str(questions_path),
            "profiles": profiles,
        },
    )
    lines = [
        "# Advanced RAG 全量检索消融报告",
        "",
        f"- 题集：`{questions_path}`",
        "- 模式：仅检索；未调用 LLM 生成最终答案。",
        "- 评分分母：每个 Profile 中检索阶段成功完成的可回答题；失败数单列，不按 0 分计入。",
        "",
        "| Profile | 已完成/总题 | 失败 | Source Recall@6 | Chunk Recall@6 | Chunk Recall@20 | Evidence Coverage@6 | MRR@6 | nDCG@6 | 平均延迟(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {profile} | {metrics['completed_questions']}/{metrics['questions']} | {metrics['failed_questions']} | "
            f"{metrics['source_recall_at_6']:.4f} | {metrics['chunk_recall_at_6']:.4f} | {metrics['chunk_recall_at_20']:.4f} | "
            f"{metrics['evidence_coverage_at_6']:.4f} | {metrics['mrr_at_6']:.4f} | "
            f"{metrics['ndcg_at_6']:.4f} | {metrics['mean_latency_seconds']:.3f} |"
        )
    (batch_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return batch_dir


def _evaluation_snapshot(profile: str, questions_path: Path, records: list[dict[str, Any]], complete: bool, generate: bool = True) -> dict[str, Any]:
    return {
        "profile": profile,
        "mode": "end_to_end" if generate else "retrieval",
        "questions_path": str(questions_path),
        "status": "complete" if complete else "in_progress",
        "records": records,
        "metrics": _metrics(records, generate=generate),
    }


def _evaluation_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    retrieval_only = result.get("mode") == "retrieval"
    lines = [
        "# Advanced RAG 全量检索评测报告" if retrieval_only else "# Advanced RAG 全量评测报告",
        "",
        f"- Profile：`{result['profile']}`",
        f"- 题集：`{result['questions_path']}`",
        f"- 模式：{'仅检索；未调用 LLM 生成最终答案。' if retrieval_only else '端到端；包含最终答案生成。'}",
        f"- 状态：`{result['status']}`",
        f"- 题目：{metrics['questions']}（完成 {metrics['completed_questions']}，失败 {metrics['failed_questions']}；可回答 {metrics['answerable_questions']}，无答案 {metrics['unanswerable_questions']}）",
        "- 检索指标分母：检索阶段成功完成的可回答题；失败题不按 0 分计入。",
        "",
        "## 检索指标",
        "",
        "| Source Recall@6 | Chunk Recall@6 | Chunk Recall@20 | Evidence Coverage@6 | MRR@6 | nDCG@6 | 平均延迟(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['source_recall_at_6']:.4f} | {metrics['chunk_recall_at_6']:.4f} | {metrics['chunk_recall_at_20']:.4f} | {metrics['evidence_coverage_at_6']:.4f} | {metrics['mrr_at_6']:.4f} | {metrics['ndcg_at_6']:.4f} | {metrics['mean_latency_seconds']:.3f} |",
        "",
    ]
    if retrieval_only:
        lines += [
            "## 无答案检索指标",
            "",
            "| 无答案误检率 |",
            "|---:|",
            f"| {metrics['unanswerable_retrieval_rate']:.4f} |",
            "",
        ]
    else:
        lines += [
            "## 拒答指标",
            "",
            "| 无答案误检率 | 拒答成功率 | 可回答题误拒率 |",
            "|---:|---:|---:|",
            f"| {metrics['unanswerable_retrieval_rate']:.4f} | {metrics['refusal_success_rate']:.4f} | {metrics['false_refusal_rate']:.4f} |",
            "",
        ]
    lines += [
        "## 分类指标",
        "",
        "| 类别 | Source Recall@6 | Evidence Coverage@6 | MRR@6 | nDCG@6 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in sorted(metrics["by_category"].items()):
        lines.append(f"| {category} | {values['source_recall_at_6']:.4f} | {values['evidence_coverage_at_6']:.4f} | {values['mrr_at_6']:.4f} | {values['ndcg_at_6']:.4f} |")
    lines += ["", "本报告描述本次单 Profile、全量题集的结果。"]
    return "\n".join(lines) + "\n"


def _failed_entry(entry: dict[str, Any], exc: Exception) -> dict[str, Any]:
    stage = exc.stage if isinstance(exc, PipelineStageError) else "unknown"
    answerable = entry.get("answerable", True)
    return {
        "id": entry["id"], "question": entry["question"], "category": entry.get("category", "legacy"), "answerable": answerable,
        "expected_sources": _expected_sources(entry), "expected_facts": entry.get("expected_facts", []), "status": "failed", "failure": {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)},
        "result": {"elapsed_seconds": 0.0, "stage_calls": {"embedding": 0, "rewrite": 0, "rerank": 0, "generation": 0}}, "final_sources": [], "candidate_sources": [],
        "source_recall_at_6": 0.0 if answerable else None, "chunk_recall_at_6": 0.0 if answerable else None, "chunk_recall_at_20": 0.0 if answerable else None,
        "evidence_coverage_at_6": 0.0 if answerable else None, "evidence_coverage_at_20": 0.0 if answerable else None,
        "mrr_at_6": 0.0 if answerable else None, "ndcg_at_6": 0.0 if answerable else None, "unanswerable_retrieved": False if not answerable else None, "refused": False,
    }


def _success_count(records: list[dict[str, Any]]) -> int:
    return sum(record.get("status") != "failed" for record in records)


def _failure_count(records: list[dict[str, Any]]) -> int:
    return sum(record.get("status") == "failed" for record in records)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def print_progress(profile: str, completed: int, total: int, successes: int, failures: int, question_id: str) -> None:
    width = 24
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{profile:<14}] [{bar}] {completed:>3}/{total} 成功:{successes} 失败:{failures} 当前:{question_id}", end="", flush=True)
    if completed == total:
        print(flush=True)


def _stage_plan(profile: str, generate: bool) -> tuple[str, ...]:
    selected = PROFILES[profile]
    stages: list[str] = []
    if selected.rewrite:
        stages.append("rewrite")
    stages.append("index_load")
    if selected.bm25:
        stages.append("bm25_load")
    if selected.dense:
        stages.append("dense_retrieval")
    if selected.bm25:
        stages.append("bm25_retrieval")
    if selected.dense and selected.bm25:
        stages.append("rrf_fusion")
    if selected.rerank:
        stages.append("rerank")
    if generate:
        stages.append("generation")
    return tuple(stages)


class _StageProgress:
    def __init__(self, profile: str, question_index: int, question_total: int, question_id: str, stages: tuple[str, ...], enabled: bool = True) -> None:
        self.profile = profile
        self.question_index = question_index
        self.question_total = question_total
        self.question_id = question_id
        self.stages = stages
        self.finished: set[str] = set()
        self.started = False
        self.enabled = enabled

    def __call__(self, stage: str, status: str) -> None:
        self.started = True
        if status in {"completed", "skipped", "failed"}:
            self.finished.add(stage)
        if not self.enabled:
            return
        width = 18
        completed = len(self.finished)
        total = len(self.stages)
        filled = int(width * completed / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        label = _STAGE_LABELS.get(stage, stage)
        state = {"running": "进行中", "completed": "完成", "skipped": "跳过", "failed": "失败"}[status]
        print(f"\r[{self.profile:<14}] 题目 {self.question_index:>3}/{self.question_total} {self.question_id:<10} 阶段 [{bar}] {completed}/{total} {label}：{state}", end="", flush=True)

    def finish(self) -> None:
        if self.started and self.enabled:
            print(flush=True)


def _stage_progress(profile: str, generate: bool, question_index: int, question_total: int, question_id: str, enabled: bool = True) -> _StageProgress:
    return _StageProgress(profile, question_index, question_total, question_id, _stage_plan(profile, generate), enabled=enabled)


def _score_entry(entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_sources(entry)
    annotations = entry.get("relevant", [])
    answerable = entry.get("answerable", True)
    final = result["retrieval"]["final"]
    candidates = result["retrieval"]["candidates"]
    final_sources = [_source_name(hit) for hit in final]
    candidate_sources = [_source_name(hit) for hit in candidates]
    refused = _is_refusal(result.get("answer"))
    return {
        "id": entry["id"],
        "status": "ok",
        "question": entry["question"],
        "category": entry.get("category", "legacy"),
        "answerable": answerable,
        "expected_sources": expected,
        "expected_facts": entry.get("expected_facts", []),
        "result": result,
        "final_sources": final_sources,
        "candidate_sources": candidate_sources,
        "source_recall_at_6": _source_recall(final, expected) if answerable else None,
        "chunk_recall_at_6": _chunk_recall(final, annotations, expected) if answerable else None,
        "chunk_recall_at_20": _chunk_recall(candidates[:20], annotations, expected) if answerable else None,
        "evidence_coverage_at_6": _evidence_coverage(final, annotations, expected) if answerable else None,
        "evidence_coverage_at_20": _evidence_coverage(candidates[:20], annotations, expected) if answerable else None,
        "mrr_at_6": _mrr(final, annotations, expected) if answerable else None,
        "ndcg_at_6": _ndcg(final, annotations, expected) if answerable else None,
        "unanswerable_retrieved": bool(final) if not answerable else None,
        "refused": refused,
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
    return source in expected_sources


def _is_relevant_chunk(hit: dict[str, Any], annotations: list[dict[str, Any]], expected_sources: list[str]) -> bool:
    if not annotations:  # legacy question sets only have a source-level label
        return _is_relevant_source(hit, expected_sources)
    source, section, text = _source_name(hit), hit["chunk"].get("section"), hit["chunk"]["text"]
    return any(item.get("source") == source and item.get("section") == section and item.get("evidence_contains") in text for item in annotations)


def _source_recall(hits: list[dict[str, Any]], expected_sources: list[str]) -> float:
    expected = set(expected_sources)
    if not expected:
        return 0.0
    retrieved = {_source_name(hit) for hit in hits}
    return round(len(expected & retrieved) / len(expected), 4)


def _gold_chunk_groups(annotations: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for item in annotations:
        chunk_ids = item.get("gold_chunk_ids")
        if isinstance(chunk_ids, list) and chunk_ids and all(isinstance(value, str) and value for value in chunk_ids):
            group = tuple(sorted(set(chunk_ids)))
        else:
            group = (f"annotation:{item.get('source')}:{item.get('section')}:{item.get('evidence_contains')}",)
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def _hit_matches_gold_group(hit: dict[str, Any], group: tuple[str, ...], annotations: list[dict[str, Any]], expected_sources: list[str]) -> bool:
    if group and not group[0].startswith("annotation:"):
        return hit["chunk"].get("chunk_id") in group
    return any(
        group == (f"annotation:{item.get('source')}:{item.get('section')}:{item.get('evidence_contains')}",)
        and item.get("source") == _source_name(hit)
        and item.get("section") == hit["chunk"].get("section")
        and item.get("evidence_contains") in hit["chunk"].get("text", "")
        for item in annotations
    ) if annotations else _is_relevant_source(hit, expected_sources)


def _chunk_recall(hits: list[dict[str, Any]], annotations: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not annotations:
        return _source_recall(hits, expected_sources)
    groups = _gold_chunk_groups(annotations)
    covered = sum(any(_hit_matches_gold_group(hit, group, annotations, expected_sources) for hit in hits) for group in groups)
    return round(covered / len(groups), 4) if groups else 0.0


def _evidence_coverage(hits: list[dict[str, Any]], annotations: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not annotations:
        return 1.0 if any(_is_relevant_source(hit, expected_sources) for hit in hits) else 0.0
    covered = sum(
        any(item.get("source") == _source_name(hit) and item.get("section") == hit["chunk"].get("section") and item.get("evidence_contains") in hit["chunk"]["text"] for hit in hits)
        for item in annotations
    )
    return round(covered / len(annotations), 4)


def _mrr(hits: list[dict[str, Any]], annotations: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not annotations:
        rank = next((index for index, hit in enumerate(hits, start=1) if _is_relevant_source(hit, expected_sources)), None)
        return 1 / rank if rank else 0.0
    groups = _gold_chunk_groups(annotations)
    reciprocal_ranks = []
    for group in groups:
        rank = next((index for index, hit in enumerate(hits, start=1) if _hit_matches_gold_group(hit, group, annotations, expected_sources)), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return round(mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0


def _ndcg(hits: list[dict[str, Any]], annotations: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not hits:
        return 0.0
    if not annotations:
        relevance = [_is_relevant_source(hit, expected_sources) for hit in hits]
        ideal_count = min(len(set(expected_sources)), 6)
    else:
        groups = _gold_chunk_groups(annotations)
        credited: set[tuple[str, ...]] = set()
        relevance = []
        for hit in hits:
            group = next((item for item in groups if item not in credited and _hit_matches_gold_group(hit, item, annotations, expected_sources)), None)
            relevance.append(group is not None)
            if group is not None:
                credited.add(group)
        ideal_count = min(len(groups), 6)
    dcg = sum(1 / __import__("math").log2(index + 1) for index, value in enumerate(relevance, start=1) if value)
    ideal = sum(1 / __import__("math").log2(index + 1) for index in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _metrics(records: list[dict[str, Any]], generate: bool = True) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") != "failed"]
    answerable = [record for record in completed if record["answerable"]]
    unanswerable = [record for record in completed if not record["answerable"]]
    metrics = {
        "questions": len(records),
        "completed_questions": len(completed),
        "failed_questions": len(records) - len(completed),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "source_recall_at_6": _average(answerable, "source_recall_at_6"),
        "chunk_recall_at_6": _average(answerable, "chunk_recall_at_6"),
        "chunk_recall_at_20": _average(answerable, "chunk_recall_at_20"),
        "evidence_coverage_at_6": _average(answerable, "evidence_coverage_at_6"),
        "evidence_coverage_at_20": _average(answerable, "evidence_coverage_at_20"),
        "mrr_at_6": _average(answerable, "mrr_at_6"),
        "ndcg_at_6": _average(answerable, "ndcg_at_6"),
        "unanswerable_retrieval_rate": _rate(unanswerable, "unanswerable_retrieved"),
        "refusal_success_rate": _rate(unanswerable, "refused") if generate else None,
        "false_refusal_rate": _rate(answerable, "refused") if generate else None,
        "mean_latency_seconds": round(mean(record["result"]["elapsed_seconds"] for record in completed), 3) if completed else 0.0,
        "stage_calls": {stage: sum(record["result"]["stage_calls"].get(stage, 0) for record in completed) for stage in ("embedding", "rewrite", "rerank", "generation")},
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in answerable:
        by_category[record["category"]].append(record)
    metrics["by_category"] = {category: {key: _average(items, key) for key in ("source_recall_at_6", "evidence_coverage_at_6", "mrr_at_6", "ndcg_at_6")} for category, items in by_category.items()}
    return metrics


def _average(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record[key] is not None]
    return round(mean(values), 4) if values else 0.0


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(record.get(key, False)) for record in records) / len(records), 4) if records else 0.0


def _is_refusal(answer: object) -> bool:
    return isinstance(answer, str) and answer.strip().startswith(_REFUSAL_PREFIX)
