from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar
from urllib.error import HTTPError, URLError


T = TypeVar("T")
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
}


class NetworkRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def request_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    retry_delay_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry only transient network failures, using a fixed delay between attempts."""
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须为正数。")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == max_attempts or not is_retryable_network_error(exc):
                raise
            sleep(retry_delay_seconds)
    raise AssertionError("unreachable")


def is_retryable_network_error(exc: BaseException) -> bool:
    """Recognise connection/timeout errors, throttling, and transient HTTP failures."""
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, NetworkRequestError):
            return current.retryable
        if isinstance(current, HTTPError):
            return current.code in RETRYABLE_HTTP_STATUS_CODES
        if isinstance(current, (TimeoutError, ConnectionError, URLError)):
            return True
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int):
            return status_code in RETRYABLE_HTTP_STATUS_CODES
        if type(current).__name__ in _RETRYABLE_EXCEPTION_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False
