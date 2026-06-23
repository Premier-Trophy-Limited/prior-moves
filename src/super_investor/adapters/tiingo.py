"""Tiingo daily-price adapter — a hardened price source vs flaky yfinance.

Single-ticker fetcher returning the canonical (date, close, volume) frame using
Tiingo's split/dividend-ADJUSTED fields (``adjClose`` / ``adjVolume``). Used both
as a `get_history` provider (prices.py) and by the channel builder
(scripts/fetch_tiingo.py).

Key from Keychain slot ``tiingo-api-key``.

PAID-API NOTE: Tiingo is a flat-rate subscription (no per-call billing). The free
tier allows ~50 symbols/hr, 500/day, limited history; paid lifts those. The
caller throttles; verify the active plan's limits with one test call before any
universe walk. This module never loops a universe by itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..secrets import get_secret

log = logging.getLogger("super_investor.adapters.tiingo")
BASE = "https://api.tiingo.com/tiingo/daily"
EMPTY = pd.DataFrame(columns=["date", "close", "volume"])


def fetch_daily_bars(ticker: str, start: str = "2014-01-01",
                     end: str | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    """Adjusted daily prices for `ticker` as (date, close, volume). Empty on miss."""
    import requests

    key = get_secret("tiingo-api-key", env_var="TIINGO_API_KEY")
    if not key:
        log.warning("fetch_daily_bars(%s): no tiingo-api-key", ticker)
        return EMPTY.copy()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    url = f"{BASE}/{ticker.upper()}/prices"
    try:
        r = requests.get(
            url,
            params={"startDate": start, "endDate": end, "format": "json",
                    "resampleFreq": "daily", "token": key},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("fetch_daily_bars(%s): %s: %s", ticker, type(e).__name__, e)
        return EMPTY.copy()
    if r.status_code != 200:
        log.warning("fetch_daily_bars(%s): HTTP %s %s", ticker, r.status_code, r.text[:120])
        return EMPTY.copy()
    try:
        rows = r.json()
    except ValueError:
        log.warning("fetch_daily_bars(%s): non-JSON body %s", ticker, r.text[:120])
        return EMPTY.copy()
    # Tiingo returns a JSON list of bar dicts on success; an error is a dict/str.
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        log.warning("fetch_daily_bars(%s): unexpected payload %s", ticker, str(rows)[:120])
        return EMPTY.copy()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    # adjClose/adjVolume are split+dividend adjusted; fall back to raw if absent.
    df["close"] = df.get("adjClose", df.get("close"))
    df["volume"] = df.get("adjVolume", df.get("volume"))
    out = df[["date", "close", "volume"]].dropna(subset=["close"])
    return out.sort_values("date").reset_index(drop=True)
