"""Polygon.io daily-bar adapter — a hardened price source vs flaky yfinance.

Single-ticker fetcher returning the canonical (date, close, volume) frame, with
split/dividend-ADJUSTED closes (`adjusted=true`). Used both as a `get_history`
provider (prices.py) and by the channel builder (scripts/fetch_polygon.py).

Key from Keychain slot ``polygon-api-key`` (REST). The flat-files S3 secret lives
in ``polygon-api-secret`` and is NOT needed for the REST aggregates path.

PAID-API NOTE: Polygon is a flat-rate subscription (no per-call billing), but the
FREE tier is 5 req/min and ~2 years of history. The caller throttles; verify the
active plan's rate limit + history depth with one test call before any universe
walk. This module never loops a universe by itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..secrets import get_secret

log = logging.getLogger("super_investor.adapters.polygon")
BASE = "https://api.polygon.io"
EMPTY = pd.DataFrame(columns=["date", "close", "volume"])


def fetch_daily_bars(ticker: str, start: str = "2014-01-01",
                     end: str | None = None, cache_dir: Path | None = None,
                     max_pages: int = 30) -> pd.DataFrame:
    """Adjusted daily bars for `ticker` as (date, close, volume). Empty on miss.

    Follows Polygon's `next_url` pagination up to `max_pages` (50k bars/page is
    far more than any single name needs, but delisted/long histories can page).
    """
    import requests

    key = get_secret("polygon-api-key", env_var="POLYGON_API_KEY")
    if not key:
        log.warning("fetch_daily_bars(%s): no polygon-api-key", ticker)
        return EMPTY.copy()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (f"{BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000")
    rows: list[dict] = []
    pages = 0
    sess = requests.Session()
    while url and pages < max_pages:
        try:
            r = sess.get(url, params={"apiKey": key}, timeout=30)
        except Exception as e:  # noqa: BLE001
            log.warning("fetch_daily_bars(%s): %s: %s", ticker, type(e).__name__, e)
            break
        if r.status_code != 200:
            log.warning("fetch_daily_bars(%s): HTTP %s %s", ticker, r.status_code, r.text[:120])
            break
        try:
            j = r.json()
        except ValueError:
            log.warning("fetch_daily_bars(%s): non-JSON body %s", ticker, r.text[:120])
            break
        if not isinstance(j, dict):
            break
        rows.extend(j.get("results") or [])
        url = j.get("next_url")
        pages += 1
    if not rows:
        return EMPTY.copy()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None).dt.normalize()
    out = df.rename(columns={"c": "close", "v": "volume"})[["date", "close", "volume"]]
    out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return out
