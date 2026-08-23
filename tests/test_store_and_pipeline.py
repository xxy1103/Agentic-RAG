from dataclasses import dataclass
from pathlib import Path

import numpy as np

from base_rag.config import load_config
from base_rag.models import Chunk
from base_rag.pipeline import ask
from base_rag.store import FaissStore, IndexMetadata


def test_faiss_persistence_and_rank(tmp_path: Path) -> None:
    chunks = [
        Chunk("a", "苹果", "s", "a.md", "markdown", 0),
        Chunk("b", "香蕉", "s", "b.md", "markdown", 1),
    ]
    metadata = IndexMetadata("fake", 2, "hash", 2, 1000, 150)
    store = FaissStore.build(np.array([[1, 0], [0, 1]], dtype=np.float32), chunks, metadata)
    store.save(tmp_path / "index")
    restored = FaissStore.load(tmp_path / "index", "fake", 2)
    assert restored.search(np.array([0.9, 0.1], dtype=np.float32), 2, None)[0].chunk.chunk_id == "a"


class FakeEmbedder:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeGenerator:
    def generate(self, prompt, temperature, max_tokens):
        assert "检索证据" in prompt
        return "这是基于资料的回答。"


def test_answer_saves_valid_retrieval_citations(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""paths:\n  corpus_dir: corpus\n  index_dir: index\n  runs_dir: runs\nmodels:\n  api_base: http://example.test\n  embedding_model: fake\n  embedding_dimensions: 2\n  llm_model: fake\n  timeout_seconds: 1\n  max_retries: 1\n  retry_backoff_seconds: 0\ningestion:\n  allowed_extensions: ['.md']\n  chunk_size: 100\n  chunk_overlap: 10\n  embedding_batch_size: 1\n  fail_on_error: true\nretrieval:\n  top_k: 1\n  min_score: 0.2\n  max_context_characters: 1000\ngeneration:\n  temperature: 0\n  max_tokens: 10\nruntime:\n  save_runs: true\n""", encoding="utf-8")
    config = load_config(config_path)
    chunk = Chunk("x", "证据正文", "s", "source.md", "markdown", 0, section="章节")
    FaissStore.build(np.array([[1.0, 0.0]], dtype=np.float32), [chunk], IndexMetadata("fake", 2, "h", 1, 100, 10)).save(config.paths.index_dir)
    result = ask(config, FakeEmbedder(), FakeGenerator(), "问题")
    assert result["citations"] == ["source.md｜章节"]
    assert "参考来源" in result["answer"]
    assert list(config.paths.runs_dir.glob("*.json"))
