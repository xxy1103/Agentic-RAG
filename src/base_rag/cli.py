from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from base_rag.config import load_config
from base_rag.embedding import DashScopeEmbedder, DashScopeGenerator
from base_rag.evaluation import create_retrieval_evaluation_batch_dir, evaluate, print_progress, write_evaluation_artifact, write_retrieval_evaluation_batch
from base_rag.multihoprag import evaluate_multihoprag, prepare_multihoprag
from base_rag.pipeline import PROFILES, ask, ingest
from base_rag.progress import print_task_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="可验证的本地 Advanced RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "ask", "eval"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    subparsers.choices["ask"].add_argument("--question", required=True, help="需要回答的问题")
    subparsers.choices["ask"].add_argument("--profile", choices=tuple(PROFILES), help="检索 Profile；省略时使用 YAML 默认值")
    for name in ("eval",):
        subparsers.choices[name].add_argument("--questions", default="evaluations/phase2_questions.yaml", help="评测问题集")
    prepare_multihop = subparsers.add_parser("prepare-multihop", help="下载并转换独立的 MultiHop-RAG 公开基准")
    prepare_multihop.add_argument("--config", default="config/multihoprag.yaml", help="MultiHop-RAG YAML 配置文件")
    prepare_multihop.add_argument("--force", action="store_true", help="重新下载并覆盖已准备的公开基准")
    eval_multihop = subparsers.add_parser("eval-multihop", help="评测独立的 MultiHop-RAG 跨文档检索基准")
    eval_multihop.add_argument("--config", default="config/multihoprag.yaml", help="MultiHop-RAG YAML 配置文件")
    eval_multihop.add_argument("--profile", choices=tuple(PROFILES), help="仅评测一个 Profile；默认评测全部五种")
    eval_multihop.add_argument("--limit", type=int, help="只评测前 N 道题，用于可控的冒烟测试")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        embedder = DashScopeEmbedder(config.models)
        generator = DashScopeGenerator(config.models)
        if args.command == "ingest":
            print(json.dumps(ingest(config, embedder, on_progress=print_task_progress), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            result = ask(config, embedder, generator, args.question, profile=args.profile)
            print(result["answer"])
        elif args.command == "eval":
            questions_path = Path(args.questions)
            batch_dir = create_retrieval_evaluation_batch_dir(config)
            results = {}
            for profile in PROFILES:
                profile_dir = batch_dir / "profiles" / profile
                profile_dir.mkdir(parents=True, exist_ok=False)
                result = evaluate(
                    config,
                    embedder,
                    generator,
                    questions_path,
                    profile,
                    generate=False,
                    on_progress=print_progress,
                    run_log_dir=profile_dir / "questions",
                )
                write_evaluation_artifact(config, result, profile_dir)
                results[profile] = result
            write_retrieval_evaluation_batch(config, questions_path, results, batch_dir)
            print(json.dumps({"artifact_dir": str(batch_dir), "mode": "retrieval", "profiles": {profile: result["metrics"] for profile, result in results.items()}}, ensure_ascii=False, indent=2))
        elif args.command == "prepare-multihop":
            print(json.dumps(prepare_multihoprag(config, force=args.force, on_progress=print_task_progress), ensure_ascii=False, indent=2))
        elif args.command == "eval-multihop":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit 必须是正整数。")
            profiles = (args.profile,) if args.profile else tuple(PROFILES)
            batch_dir, results = evaluate_multihoprag(config, embedder, generator, profiles=profiles, limit=args.limit, on_progress=print_progress)
            print(json.dumps({"artifact_dir": str(batch_dir), "benchmark": "MultiHop-RAG", "mode": "retrieval", "profiles": {profile: result["metrics"] for profile, result in results.items()}}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
