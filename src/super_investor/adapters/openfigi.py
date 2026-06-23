"""OpenFIGI client for CUSIP -> ticker resolution.

13F filings identify positions by CUSIP only; downstream feature joins want a
ticker. OpenFIGI's free /v3/mapping endpoint maps batches of CUSIPs to
(figi, ticker, name, exchange).

Quotas:
  - anonymous: 10 jobs/req, 25 req/min  (250 jobs/min)
  - free key:  100 jobs/req, 250 req/min  (25000 jobs/min)
  - paid:      1000 jobs/req, higher

Disk-caches each CUSIP -> mapping JSON so re-runs hit the local store.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


_FIGI_URL = "https://api.openfigi.com/v3/mapping"
# OpenFIGI per-request job cap: 10 anonymous, 100 with free key, 1000 paid.
_MAX_BATCH_NOKEY = 10
_MAX_BATCH_KEY = 100
# Per-request gap in seconds to stay below the rate cap.
_GAP_NOKEY = 60.0 / 25  # 25 req/min => 2.4s
_GAP_KEY = 60.0 / 250   # 250 req/min => 0.24s


class OpenFigiClient:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        self.api_key = api_key or os.environ.get("OPENFIGI_API_KEY") or None
        self.cache_dir = cache_dir
        self._max_batch = _MAX_BATCH_KEY if self.api_key else _MAX_BATCH_NOKEY
        self._min_gap = _GAP_KEY if self.api_key else _GAP_NOKEY
        self._last_t: float = 0.0
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last_t = time.monotonic()

    def _cache_path(self, cusip: str) -> Path | None:
        if not self.cache_dir:
            return None
        # First 2 chars of CUSIP shard the cache so we don't blow a single dir.
        return self.cache_dir / cusip[:2] / f"{cusip}.json"

    def _cache_get(self, cusip: str) -> dict | None:
        p = self._cache_path(cusip)
        if p and p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _cache_put(self, cusip: str, payload: dict) -> None:
        p = self._cache_path(cusip)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload))

    def map_cusips(self, cusips: Iterable[str]) -> dict[str, dict]:
        """Return {cusip: {ticker, name, exchange, figi, ...}} for every input.

        Unresolvable CUSIPs map to an empty dict. Per OpenFIGI, the first match
        whose `marketSector` is "Equity" is preferred.
        """
        cusips = [c.upper().strip() for c in cusips if c and len(c) >= 8]
        cusips = sorted(set(cusips))
        out: dict[str, dict] = {}
        uncached: list[str] = []
        for c in cusips:
            hit = self._cache_get(c)
            if hit is not None:
                out[c] = hit
            else:
                uncached.append(c)

        if uncached:
            self._fetch_uncached(uncached, out)

        return out

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=3, max=60),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)),
    )
    def _fetch_batch(self, batch: list[str]) -> list[dict]:
        self._throttle()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        with httpx.Client(timeout=30.0, headers=headers) as client:
            r = client.post(_FIGI_URL, json=payload)
            if r.status_code == 429:
                # Server says back off — sleep for the Retry-After if provided, else 30s
                retry_after = float(r.headers.get("Retry-After", "30"))
                time.sleep(min(retry_after, 60.0))
                r.raise_for_status()  # raises → tenacity retries
            r.raise_for_status()
            return r.json()

    def _fetch_uncached(self, cusips: list[str], out: dict[str, dict]) -> None:
        for start in range(0, len(cusips), self._max_batch):
            batch = cusips[start : start + self._max_batch]
            response = self._fetch_batch(batch)
            assert len(response) == len(batch), \
                f"OpenFIGI returned {len(response)} responses for batch of {len(batch)}"
            for cusip, entry in zip(batch, response):
                data = entry.get("data", []) or []
                # Prefer Equity / US Composite if there are multiple matches.
                equity = next(
                    (d for d in data if d.get("marketSector") == "Equity"),
                    None,
                )
                resolved = equity or (data[0] if data else {})
                payload = {
                    "ticker": resolved.get("ticker", ""),
                    "name": resolved.get("name", ""),
                    "exchange_code": resolved.get("exchCode", ""),
                    "figi": resolved.get("figi", ""),
                    "security_type": resolved.get("securityType", ""),
                    "market_sector": resolved.get("marketSector", ""),
                    "_raw": data,
                }
                out[cusip] = payload
                self._cache_put(cusip, payload)
