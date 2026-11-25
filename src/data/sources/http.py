"""Shared polite HTTP: retries, exponential backoff, a fair-use delay, raw-response caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

_LAST_CALL: Dict[str, float] = {}


class FetchError(RuntimeError):
    pass


def _sleep_polite(key: str, delay_s: float) -> None:
    last = _LAST_CALL.get(key)
    if last is not None:
        wait = delay_s - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
    _LAST_CALL[key] = time.time()


def get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    source: str = "generic",
    timeout_s: float = 90.0,
    retries: int = 4,
    backoff_s: float = 2.0,
    polite_delay_s: float = 0.7,
    user_agent: str = "AquaSentinel/0.1",
    accept: str = "application/json",
) -> requests.Response:
    """GET with retries. Raises FetchError once every attempt is exhausted."""
    headers = {"User-Agent": user_agent, "Accept": accept}
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        _sleep_polite(source, polite_delay_s)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:  # network-level failure
            last_exc = exc
        else:
            if resp.status_code == 200:
                return resp
            # 400/403/404 on a per-site query means "nothing here" - not worth retrying.
            if resp.status_code in (400, 403, 404):
                raise FetchError(f"{resp.status_code} for {resp.url[:200]}")
            last_exc = FetchError(f"{resp.status_code} for {resp.url[:200]}")
        time.sleep(backoff_s * (2**attempt))
    raise FetchError(f"failed after {retries} attempts: {url} ({last_exc})")


def cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    blob = url + "|" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def save_raw(raw_dir: Path, name: str, payload: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    path.write_text(payload, encoding="utf-8")
    return path


def http_kwargs(http_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the `http:` block of data.yaml into get() keyword arguments."""
    return {
        "timeout_s": http_cfg["timeout_s"],
        "retries": http_cfg["retries"],
        "backoff_s": http_cfg["backoff_s"],
        "polite_delay_s": http_cfg["polite_delay_s"],
        "user_agent": http_cfg["user_agent"],
    }
