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
    retry_delay_seconds: float


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
    dense_candidate_k: int = 20
    sparse_candidate_k: int = 20
    fusion_candidate_k: int = 30
    rrf_k: int = 60


@dataclass(frozen=True)
class QueryRewriteConfig:
    enabled: bool = False
    max_tokens: int = 400
    language: str = "zh"


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = "qwen3-rerank"
    top_n: int = 6
    instruct: str = "Given a web search query, retrieve relevant passages that answer the query."


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class RuntimeConfig:
    save_runs: bool
    default_profile: str = "dense"


@dataclass(frozen=True)
class EvaluationConfig:
    concurrency: int = 4


@dataclass(frozen=True)
class MultiHopRAGConfig:
    dataset_dir: Path


@dataclass(frozen=True)
class AgenticConfig:
    base_profile: str = "hybrid-rerank"
    max_hops: int = 3
    max_corrections_per_hop: int = 1
    structured_output_retries: int = 1
    recursion_limit: int = 25


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    models: ModelsConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    query_rewrite: QueryRewriteConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    multihoprag: MultiHopRAGConfig | None
    agentic: AgenticConfig
    config_path: Path

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_path"] = str(self.config_path)
        for key in ("corpus_dir", "index_dir", "runs_dir"):
            data["paths"][key] = str(data["paths"][key])
        if data["multihoprag"] is not None:
            data["multihoprag"]["dataset_dir"] = str(data["multihoprag"]["dataset_dir"])
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
    models = dict(raw["models"])
    if "retry_delay_seconds" not in models and "retry_backoff_seconds" in models:
        models["retry_delay_seconds"] = models.pop("retry_backoff_seconds")
    config = AppConfig(
        paths=PathsConfig(**{name: _resolve(base, paths[name]) for name in ("corpus_dir", "index_dir", "runs_dir")} ),
        models=ModelsConfig(**models),
        ingestion=IngestionConfig(
            allowed_extensions=tuple(item.lower() for item in raw["ingestion"]["allowed_extensions"]),
            **{key: value for key, value in raw["ingestion"].items() if key != "allowed_extensions"},
        ),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        query_rewrite=QueryRewriteConfig(**raw.get("query_rewrite", {})),
        reranker=RerankerConfig(**raw.get("reranker", {})),
        generation=GenerationConfig(**raw["generation"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        multihoprag=MultiHopRAGConfig(dataset_dir=_resolve(base, raw["multihoprag"]["dataset_dir"])) if "multihoprag" in raw else None,
        agentic=AgenticConfig(**raw.get("agentic", {})),
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
    if min(config.retrieval.dense_candidate_k, config.retrieval.sparse_candidate_k, config.retrieval.fusion_candidate_k, config.retrieval.rrf_k) <= 0:
        raise ValueError("候选数和 rrf_k 必须为正数。")
    if config.reranker.top_n <= 0:
        raise ValueError("reranker.top_n 必须为正数。")
    if config.query_rewrite.language not in {"auto", "zh", "en"}:
        raise ValueError("query_rewrite.language 只能是 auto、zh 或 en。")
    if config.models.max_retries <= 0 or config.models.retry_delay_seconds < 0:
        raise ValueError("max_retries 必须为正数，retry_delay_seconds 不能为负数。")
    if config.evaluation.concurrency <= 0:
        raise ValueError("evaluation.concurrency 必须为正数。")
    if config.agentic.base_profile not in {"dense", "bm25", "hybrid", "hybrid-rerank", "advanced"}:
        raise ValueError("agentic.base_profile 必须是已知的 Profile。")
    if config.agentic.max_hops <= 0:
        raise ValueError("agentic.max_hops 必须为正数。")
    if config.agentic.max_corrections_per_hop < 0:
        raise ValueError("agentic.max_corrections_per_hop 不能为负数。")
    if config.agentic.structured_output_retries < 0:
        raise ValueError("agentic.structured_output_retries 不能为负数。")
    if config.agentic.recursion_limit <= 0:
        raise ValueError("agentic.recursion_limit 必须为正数。")
