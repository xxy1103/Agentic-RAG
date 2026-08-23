from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from base_rag.config import load_config
from base_rag.embedding import DashScopeEmbedder, DashScopeGenerator
from base_rag.pipeline import ask, ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="本地文档 FAISS RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "ask", "eval"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    subparsers.choices["ask"].add_argument("--question", required=True, help="需要回答的问题")
    subparsers.choices["eval"].add_argument("--questions", default="evaluations/questions.yaml", help="固定问题集")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        if args.command == "ingest":
            print(json.dumps(ingest(config, DashScopeEmbedder(config.models)), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            result = ask(config, DashScopeEmbedder(config.models), DashScopeGenerator(config.models), args.question)
            print(result["answer"])
        else:
            print(json.dumps(_evaluate(config, Path(args.questions)), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _evaluate(config, questions_path: Path) -> dict[str, object]:
    entries = yaml.safe_load(questions_path.read_text(encoding="utf-8"))["questions"]
    embedder = DashScopeEmbedder(config.models)
    generator = DashScopeGenerator(config.models)
    records = []
    for entry in entries:
        result = ask(config, embedder, generator, entry["question"])
        expected = entry["expected_source"]
        hit = any(expected in citation for citation in result["citations"])
        records.append({"id": entry["id"], "expected_source": expected, "hit_at_k": hit, "citations": result["citations"]})
    hit_count = sum(record["hit_at_k"] for record in records)
    return {"questions": len(records), "hit_at_k": round(hit_count / len(records), 3) if records else 0, "records": records}
