from __future__ import annotations

from urllib.error import HTTPError

import pytest

from base_rag.network import request_with_retry


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_transient_network_error_retries_after_fixed_delay() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _StatusError(429)
        return "ok"

    result = request_with_retry(operation, max_attempts=3, retry_delay_seconds=5, sleep=sleeps.append)

    assert result == "ok"
    assert calls == 3
    assert sleeps == [5, 5]


def test_non_retryable_error_fails_immediately() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise _StatusError(401)

    with pytest.raises(_StatusError):
        request_with_retry(operation, max_attempts=3, retry_delay_seconds=5, sleep=sleeps.append)

    assert calls == 1
    assert sleeps == []


def test_non_retryable_http_error_is_not_mistaken_for_connection_failure() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise HTTPError("https://example.test", 401, "unauthorised", {}, None)

    with pytest.raises(HTTPError):
        request_with_retry(operation, max_attempts=3, retry_delay_seconds=5, sleep=lambda _: None)

    assert calls == 1
