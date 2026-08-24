from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from base_rag.config import ModelsConfig, QueryRewriteConfig


class TextGenerator(Protocol):
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str
    rewritten_query: str
    keywords: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"intent": self.intent, "rewritten_query": self.rewritten_query, "keywords": self.keywords}


class QueryRewriter:
    def __init__(self, generator: TextGenerator, models: ModelsConfig, config: QueryRewriteConfig) -> None:
        self.generator = generator
        self.models = models
        self.config = config

    def rewrite(self, question: str) -> QueryAnalysis:
        last_error: Exception | None = None
        for attempt in range(self.models.max_retries):
            content = self.generator.generate(_prompt(question), 0, self.config.max_tokens)
            try:
                return _parse(content)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.models.max_retries:
                    time.sleep(self.models.retry_delay_seconds)
        assert last_error is not None
        raise ValueError(f"Query Rewrite 未返回合法 JSON：{last_error}") from last_error


def _prompt(question: str) -> str:
    return f"""你是 RAG 查询理解器。分析用户问题并只输出一个 JSON 对象，不要 Markdown、解释或代码块。
JSON 必须严格含有 intent（字符串）、rewritten_query（字符串）和 keywords（字符串数组）。
rewritten_query 应补全指代与上下文，保持原意；keywords 只保留实体、术语、缩写、版本号和限定词。

用户问题：{question}"""


def _parse(content: str) -> QueryAnalysis:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("根节点不是对象")
    intent = value.get("intent")
    rewritten_query = value.get("rewritten_query")
    keywords = value.get("keywords")
    if not isinstance(intent, str) or not intent.strip() or not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise ValueError("intent 或 rewritten_query 缺失")
    if not isinstance(keywords, list) or not keywords or not all(isinstance(item, str) and item.strip() for item in keywords):
        raise ValueError("keywords 必须是非空字符串数组")
    return QueryAnalysis(intent.strip(), rewritten_query.strip(), [item.strip() for item in keywords])
