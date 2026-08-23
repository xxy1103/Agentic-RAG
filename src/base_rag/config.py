from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    corpus_dir: Path
    index_dir: Path
    runs_dir: Path


@dataclass(frozen=True)
class ModelsConfig:
    api_base: str
    embedding_model: str
    embedding_dimensions: int
    llm_model: str
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class IngestionConfig:
    allowed_extensions: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    fail_on_error: bool


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    min_score: float | None
    max_context_characters: int


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class RuntimeConfig:
    save_runs: bool


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    models: ModelsConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    config_path: Path

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_path"] = str(self.config_path)
        for key in ("corpus_dir", "index_dir", "runs_dir"):
            data["paths"][key] = str(data["paths"][key])
        return data


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML 配置必须是对象。")
    if _contains_key(raw, "api_key") or _contains_key(raw, "dashscope_api_key"):
        raise ValueError("API Key 只能写在 .env 或环境变量，不能写入 YAML。")
    base = path.parent.parent
    paths = raw["paths"]
    config = AppConfig(
        paths=PathsConfig(**{name: _resolve(base, paths[name]) for name in ("corpus_dir", "index_dir", "runs_dir")} ),
        models=ModelsConfig(**raw["models"]),
        ingestion=IngestionConfig(
            allowed_extensions=tuple(item.lower() for item in raw["ingestion"]["allowed_extensions"]),
            **{key: value for key, value in raw["ingestion"].items() if key != "allowed_extensions"},
        ),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generation=GenerationConfig(**raw["generation"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        config_path=path,
    )
    _validate(config)
    return config


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(key.lower() == target or _contains_key(child, target) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


def _validate(config: AppConfig) -> None:
    if config.ingestion.chunk_overlap >= config.ingestion.chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")
    if config.models.embedding_dimensions <= 0 or config.retrieval.top_k <= 0:
        raise ValueError("embedding_dimensions 和 top_k 必须为正数。")
