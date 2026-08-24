from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from base_rag.config import load_config
from base_rag.embedding import DashScopeEmbedder, DashScopeGenerator
from base_rag.evaluation import create_evaluation_artifact_dir, evaluate, print_progress, rebuild_report, run_experiment, write_evaluation_artifact
from base_rag.pipeline import PROFILES, ask, ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="可验证的本地 Advanced RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "ask", "eval-dev", "eval-test", "experiment"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    subparsers.choices["ask"].add_argument("--question", required=True, help="需要回答的问题")
    for name in ("ask", "eval-dev", "eval-test"):
        subparsers.choices[name].add_argument("--profile", choices=tuple(PROFILES), help="检索 Profile；省略时使用 YAML 默认值")
    for name in ("eval-dev", "eval-test", "experiment"):
        subparsers.choices[name].add_argument("--questions", default="evaluations/phase2_questions.yaml", help="评测问题集")
    report = subparsers.add_parser("experiment-report")
    report.add_argument("--experiment", required=True, help="experiment 产物目录")
    report.add_argument("--reviews", required=True, help="已填写的人工复核 CSV")
    args = parser.parse_args()
    try:
        if args.command == "experiment-report":
            print(rebuild_report(Path(args.experiment), Path(args.reviews)))
            return
        config = load_config(args.config)
        embedder = DashScopeEmbedder(config.models)
        generator = DashScopeGenerator(config.models)
        if args.command == "ingest":
            print(json.dumps(ingest(config, embedder), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            result = ask(config, embedder, generator, args.question, profile=args.profile)
            print(result["answer"])
        elif args.command in {"eval-dev", "eval-test"}:
            split = args.command.removeprefix("eval-")
            profile = args.profile or config.runtime.default_profile
            artifact_dir = create_evaluation_artifact_dir(config, profile, split)
            result = evaluate(config, embedder, generator, Path(args.questions), profile, split=split, on_progress=print_progress, run_log_dir=artifact_dir / "questions")
            artifact_dir = write_evaluation_artifact(config, result, artifact_dir)
            print(json.dumps({"artifact_dir": str(artifact_dir), "profile": profile, "split": split, **result["metrics"]}, ensure_ascii=False, indent=2))
        else:
            print(run_experiment(config, embedder, generator, Path(args.questions)))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
