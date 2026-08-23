from __future__ import annotations

import os
import time
from collections.abc import Iterable

import numpy as np
from dotenv import load_dotenv

from base_rag.config import ModelsConfig


class DashScopeEmbedder:
    def __init__(self, config: ModelsConfig) -> None:
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY。请在 .env 中设置它。")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=config.api_base, timeout=config.timeout_seconds)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.embedding_dimensions), dtype=np.float32)
        response = self._retry(lambda: self.client.embeddings.create(model=self.config.embedding_model, input=texts, dimensions=self.config.embedding_dimensions))
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        if vectors.shape[1] != self.config.embedding_dimensions:
            raise ValueError(f"Embedding 维度不匹配：期望 {self.config.embedding_dimensions}，实际 {vectors.shape[1]}。")
        return vectors

    def _retry(self, operation):
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error


class DashScopeGenerator:
    def __init__(self, config: ModelsConfig) -> None:
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY。请在 .env 中设置它。")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=config.api_base, timeout=config.timeout_seconds)

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self._retry(lambda: self.client.chat.completions.create(model=self.config.llm_model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens))
        return (response.choices[0].message.content or "证据不足，无法基于已检索文档回答。").strip()

    def _retry(self, operation):
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error


def batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
