from pathlib import Path

from base_rag.chunking import chunk_documents
from base_rag.loaders import load_path
from base_rag.models import SourceDocument


def test_markdown_loader_strips_noise_and_keeps_alt_text(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("---\ntitle: 测试文章\ntags:\n  - rag\n---\n<!-- hidden -->\n# 第一节\n![图示](https://example.com/a.png)\n保留的正文。", encoding="utf-8")
    document = load_path(path)[0]
    assert document.title == "测试文章"
    assert "hidden" not in document.text
    assert "图示" in document.text


def test_html_loader_removes_navigation(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<html><head><title>网页</title><style>x</style></head><body><nav>菜单</nav><article>正文内容</article><script>bad()</script></body></html>", encoding="utf-8")
    document = load_path(path)[0]
    assert document.title == "网页"
    assert "正文内容" in document.text
    assert "菜单" not in document.text


def test_heading_sections_and_stable_chunks() -> None:
    document = SourceDocument("source", "# 第一节\n" + "甲" * 80 + "\n# 第二节\n" + "乙" * 80, "sample.md", "markdown", "样例")
    first = chunk_documents([document], 60, 10)
    second = chunk_documents([document], 60, 10)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert {item.section for item in first} == {"第一节", "第二节"}
    assert any(left.text[-10:] in right.text for left, right in zip(first, first[1:]) if left.section == right.section)
