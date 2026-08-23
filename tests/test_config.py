from pathlib import Path

import pytest

from base_rag.config import load_config


def test_config_rejects_key_in_yaml(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("api_key: forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="API Key"):
        load_config(config)


def test_default_config_has_no_secret() -> None:
    config = load_config("config/default.yaml")
    assert "DASHSCOPE_API_KEY" not in str(config.safe_dict())
    assert config.ingestion.chunk_overlap < config.ingestion.chunk_size
