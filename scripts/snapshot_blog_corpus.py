"""按已锁定的主题优先级复制 70 篇博客文章，并生成审查清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


KEYWORDS = {
    "kdd_agent": ("kdd", "agent", "数据 agent"),
    "rag": ("rag", "embedding", "向量检索", "检索增强", "langchain", "langgraph"),
    "systems": ("操作系统", "计算机系统", "计算机网络", "进程", "内存管理", "同步", "互斥", "链接", "io设备", "linux"),
    "python_data": ("python", "数据处理", "数据结构", "sql", "mysql", "scrapy", "flask", "pandas", "numpy", "爬虫"),
    "engineering": ("git", "工程", "架构", "docker", "cmake", "web开发", "spring", "部署", "vscode", "blog"),
}
ORDER = ("kdd_agent", "rag", "systems", "python_data", "engineering")
TARGETS = {"systems": 20, "python_data": 15, "engineering": 15}
EXCLUDED_STEMS = {"ans", "answer", "主题样式完全展示"}


def parse_article(path: Path) -> dict[str, object] | None:
    raw = path.read_text(encoding="utf-8")
    body = re.sub(r"^---\s*\n.*?\n---\s*\n?", "", raw, count=1, flags=re.S)
    visible = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()
    if path.stem in EXCLUDED_STEMS or len(visible) < 1500 or re.search(r"^draft:\s*true\s*$", raw, flags=re.M | re.I):
        return None
    title_match = re.search(r"^title:\s*[\"']?(.*?)[\"']?\s*$", raw, flags=re.M)
    tags = re.findall(r"^\s*-\s*[\"']?([^\"'\n]+)", raw.split("---", 2)[1] if raw.startswith("---") and raw.count("---") >= 2 else "")
    title = title_match.group(1).strip() if title_match else path.stem
    head = f"{path.name} {title} {' '.join(tags)}".lower()
    full = visible.lower()
    head_matched = [category for category in ORDER if any(word in head for word in KEYWORDS[category])]
    body_matched = [category for category in ORDER if any(word in full for word in KEYWORDS[category])]
    matched = head_matched or body_matched
    if not matched:
        return None
    primary = matched[0]
    title_hits = sum(word in head for word in KEYWORDS[primary])
    body_hits = sum(word in full for word in KEYWORDS[primary])
    return {"path": path, "title": title, "tags": tags, "characters": len(visible), "category": primary, "score": (title_hits, body_hits, len(visible), path.name.lower())}


def select(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        groups[str(item["category"])].append(item)
    for group in groups.values():
        group.sort(key=lambda x: (-x["score"][0], -x["score"][1], -x["score"][2], x["score"][3]))
    selected: list[dict[str, object]] = []
    for category in ("kdd_agent", "rag"):
        selected.extend(groups[category])
    for category, target in TARGETS.items():
        selected.extend(groups[category][:target])
    used = {item["path"] for item in selected}
    cursors = {category: 0 for category in ("systems", "python_data", "engineering")}
    while len(selected) < 70:
        progressed = False
        for category in ("systems", "python_data", "engineering", "kdd_agent", "rag"):
            group = groups[category]
            while cursors.get(category, 0) < len(group) and group[cursors.get(category, 0)]["path"] in used:
                cursors[category] = cursors.get(category, 0) + 1
            if cursors.get(category, 0) < len(group) and len(selected) < 70:
                item = group[cursors[category]]
                selected.append(item)
                used.add(item["path"])
                cursors[category] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != 70:
        raise RuntimeError(f"符合规则的文章只有 {len(selected)} 篇，无法建立 70 篇快照。")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", default=Path("data/raw"), type=Path)
    parser.add_argument("--manifest", default=Path("data/corpus_manifest.json"), type=Path)
    args = parser.parse_args()
    candidates = [item for path in args.source.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".mdx"} if (item := parse_article(path))]
    selected = select(candidates)
    args.target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in selected:
        source: Path = item["path"]
        destination = args.target / source.name
        shutil.copy2(source, destination)
        manifest.append({"source_path": str(source.resolve()), "snapshot_path": str(destination.resolve()), "theme": item["category"], "title": item["title"], "tags": item["tags"], "characters": item["characters"], "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(manifest), "themes": {theme: sum(row["theme"] == theme for row in manifest) for theme in ORDER}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
