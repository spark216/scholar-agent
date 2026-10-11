"""Shared resilient HTTP transport for academic metadata providers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any

import httpx

from backend.app.config import get_settings


_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_CIRCUIT_LOCK = RLock()


@dataclass
class _CircuitState:
    failures: int = 0
    opened_at: float | None = None


_CIRCUITS: dict[str, _CircuitState] = {}


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        provider: str,
        error_type: str,
        *,
        attempts: int,
        circuit_open: bool = False,
    ) -> None:
        super().__init__(f"{provider} request failed: {error_type}")
        self.provider = provider
        self.error_type = error_type
        self.attempts = attempts
        self.circuit_open = circuit_open


@lru_cache(maxsize=1)
def get_academic_http_client() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        timeout=httpx.Timeout(
            timeout=settings.academic_request_timeout,
            connect=settings.academic_connect_timeout,
            pool=settings.academic_pool_timeout,
        ),
        limits=httpx.Limits(
            max_connections=settings.academic_http_max_connections,
            max_keepalive_connections=settings.academic_http_max_keepalive_connections,
            keepalive_expiry=60.0,
        ),
        follow_redirects=True,
        headers={"Accept-Encoding": "gzip, deflate"},
    )


def _circuit_available(provider: str) -> bool:
    settings = get_settings()
    now = time.monotonic()
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.setdefault(provider, _CircuitState())
        if state.opened_at is None:
            return True
        if now - state.opened_at >= settings.academic_circuit_cooldown_seconds:
            # Allow one half-open probe after the cooldown.
            state.opened_at = None
            state.failures = max(0, settings.academic_circuit_failure_threshold - 1)
            return True
        return False


def _record_success(provider: str) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS[provider] = _CircuitState()


def _record_failure(provider: str) -> None:
    settings = get_settings()
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.setdefault(provider, _CircuitState())
        state.failures += 1
        if state.failures >= settings.academic_circuit_failure_threshold:
            state.opened_at = time.monotonic()


def reset_academic_http_state() -> None:
    """Reset connection/circuit state for tests and controlled restarts."""
    with _CIRCUIT_LOCK:
        _CIRCUITS.clear()
    if get_academic_http_client.cache_info().currsize:
        get_academic_http_client().close()
    get_academic_http_client.cache_clear()


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ),
    )


def _retry_after_seconds(exc: Exception) -> float:
    """Return a bounded numeric Retry-After delay when a provider supplies one."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return 0.0
    value = exc.response.headers.get("Retry-After")
    try:
        return min(5.0, max(0.0, float(value or 0.0)))
    except ValueError:
        return 0.0


def _request_error_type(exc: Exception | None) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP{exc.response.status_code}"
    return type(exc).__name__ if exc is not None else "UnknownError"


def academic_http_request(
    provider: str,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Request with keep-alive, bounded retry and a per-provider circuit."""
    if not _circuit_available(provider):
        raise ProviderRequestError(
            provider,
            "CircuitOpenError",
            attempts=0,
            circuit_open=True,
        )

    settings = get_settings()
    max_attempts = max(1, settings.academic_request_max_attempts)
    last_error: Exception | None = None
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        try:
            response = get_academic_http_client().request(method, url, **kwargs)
            response.raise_for_status()
            response.extensions["scholaragent_attempts"] = attempts
            _record_success(provider)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not _is_transient(exc) or attempts >= max_attempts:
                break
            delay = settings.academic_retry_backoff_seconds * (2 ** (attempts - 1))
            delay += random.uniform(0.0, max(0.0, delay * 0.2))
            delay = max(delay, _retry_after_seconds(exc))
            time.sleep(delay)

    _record_failure(provider)
    error_type = _request_error_type(last_error)
    raise ProviderRequestError(provider, error_type, attempts=attempts) from last_error


def request_attempts(response: httpx.Response) -> int:
    return max(1, int(response.extensions.get("scholaragent_attempts") or 1))
