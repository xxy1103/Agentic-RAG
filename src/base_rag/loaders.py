from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup

from base_rag.models import SourceDocument


class DocumentLoadError(ValueError):
    pass


def load_path(path: Path) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".md", ".mdx"}:
        return [_load_markdown(path)]
    if suffix in {".html", ".htm"}:
        return [_load_html(path)]
    raise DocumentLoadError(f"不支持的文件类型：{path.suffix}")


def _source_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _load_markdown(path: Path) -> SourceDocument:
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"^\s*(import|export)\s+.*$", "", body, flags=re.M)
    title = frontmatter.get("title") or _first_heading(body) or path.stem
    text = _normalise(body)
    if not text:
        raise DocumentLoadError(f"Markdown 没有可检索文本：{path}")
    return SourceDocument(_source_id(path), text, str(path.resolve()), "markdown", str(title), metadata={"frontmatter": frontmatter})


def _load_html(path: Path) -> SourceDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    text = _normalise(soup.get_text("\n", strip=True))
    if not text:
        raise DocumentLoadError(f"HTML 没有可检索正文：{path}")
    return SourceDocument(_source_id(path), text, str(path.resolve()), "html", title or path.stem)


def _load_pdf(path: Path) -> list[SourceDocument]:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentLoadError("缺少 PyMuPDF，请安装项目依赖。") from exc
    docs: list[SourceDocument] = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            text = _normalise(page.get_text("text"))
            if text:
                docs.append(SourceDocument(_source_id(path), text, str(path.resolve()), "pdf", path.stem, page=index))
    if not docs:
        raise DocumentLoadError(f"PDF 无可提取文本（扫描件不在首版支持范围内）：{path}")
    return docs


def _split_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    if not raw.startswith("---"):
        return {}, raw
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, flags=re.S)
    if not match:
        return {}, raw
    try:
        import yaml
        return yaml.safe_load(match.group(1)) or {}, match.group(2)
    except Exception:
        return {}, match.group(2)


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return match.group(1).strip() if match else None


def _normalise(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
