from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from base_rag.config import ModelsConfig, RerankerConfig
from base_rag.models import SearchHit
from base_rag.network import NetworkRequestError, RETRYABLE_HTTP_STATUS_CODES, request_with_retry


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


class DashScopeReranker:
    def __init__(self, models: ModelsConfig, config: RerankerConfig) -> None:
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY。请在 .env 中设置它。")
        self.models = models
        self.config = config
        self.api_key = api_key
        self.endpoint = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": [hit.chunk.text for hit in hits],
            "top_n": min(top_n, len(hits)),
            "instruct": self.config.instruct,
        }
        response = self._retry(lambda: self._request(payload))
        results = response.get("results")
        if not isinstance(results, list):
            raise ValueError("qwen3-rerank 响应缺少 results。")
        returned: list[SearchHit] = []
        seen: set[int] = set()
        for rank, item in enumerate(results[:top_n], start=1):
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise ValueError("qwen3-rerank 返回了无效候选索引。")
            index = item["index"]
            if index < 0 or index >= len(hits) or index in seen:
                raise ValueError("qwen3-rerank 返回了越界或重复候选索引。")
            score = item.get("relevance_score")
            if not isinstance(score, (int, float)):
                raise ValueError("qwen3-rerank 返回了无效相关性分数。")
            seen.add(index)
            returned.append(replace(hits[index], score=float(score), rank=rank, rerank_score=float(score), rerank_rank=rank))
        if not returned:
            raise ValueError("qwen3-rerank 没有返回任何候选。")
        return returned

    def _retry(self, operation):
        return request_with_retry(
            operation,
            max_attempts=self.models.max_retries,
            retry_delay_seconds=self.models.retry_delay_seconds,
        )

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.models.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise NetworkRequestError(
                f"qwen3-rerank HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}",
                status_code=exc.code,
                retryable=exc.code in RETRYABLE_HTTP_STATUS_CODES,
            ) from exc
        except URLError as exc:
            raise NetworkRequestError(f"qwen3-rerank 网络错误：{exc.reason}", retryable=True) from exc
        if not isinstance(value, dict):
            raise ValueError("qwen3-rerank 响应不是对象。")
        return value
