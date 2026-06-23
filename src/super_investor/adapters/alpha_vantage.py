"""Alpha Vantage — fundamentals + news sentiment + economic indicators.

Requires free key (https://www.alphavantage.co/support/#api-key). Limit:
free tier 5 calls/min, 500/day. Store in macOS Keychain (preferred) or
env-var fallback:

  security add-generic-password -s "alphavantage-api-key" -a "$USER" -w "<key>"
  # or:
  export ALPHAVANTAGE_API_KEY=...

Endpoints used:
  function=OVERVIEW       — company snapshot (PE, EPS, dividend, etc.)
  function=NEWS_SENTIMENT — ticker-tagged news + sentiment 1.0..-1.0
  function=EARNINGS       — quarterly EPS surprise

Channel prefix ``avg_*``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from super_investor.secrets import get_secret


UA = "super-investor-mirror research@example.com"
BASE = "https://www.alphavantage.co/query"


def _key() -> str | None:
    return get_secret("alphavantage-api-key", env_var="ALPHAVANTAGE_API_KEY")


def _get(params: dict, cache_path: Path | None = None) -> dict:
    api_key = _key()
    if not api_key:
        return {}
    if cache_path and cache_path.exists() and cache_path.stat().st_size > 100:
        import json
        return json.loads(cache_path.read_text())
    params = {**params, "apikey": api_key}
    try:
        r = requests.get(BASE, params=params, headers={"User-Agent": UA}, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if cache_path is not None:
                cache_path.write_text(r.text)
            return data
    except Exception:
        pass
    return {}


def fetch_news_sentiment(
    ticker: str,
    time_from: str = "20210601T0000",
    time_to: str = "20260601T0000",
    cache_dir: Path | None = None,
) -> list[dict]:
    """Pull NEWS_SENTIMENT endpoint for one ticker."""
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    cp = cache_dir / f"{safe}_news.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    data = _get(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": safe,
            "time_from": time_from,
            "time_to": time_to,
            "limit": 1000,
            "sort": "EARLIEST",
        },
        cache_path=cp,
    )
    return data.get("feed", []) or []


def fetch_earnings(ticker: str, cache_dir: Path | None = None) -> list[dict]:
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    cp = cache_dir / f"{safe}_earnings.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    data = _get({"function": "EARNINGS", "symbol": safe}, cache_path=cp)
    return data.get("quarterlyEarnings", []) or []


def aggregate_to_ticker_quarter(
    news_rows: Iterable[tuple[str, list[dict]]],
    earnings_rows: Iterable[tuple[str, list[dict]]] | None = None,
) -> pd.DataFrame:
    """Combine news + earnings into per-(ticker, quarter) features."""
    frames = []
    for ticker, feed in news_rows:
        if not feed:
            continue
        rows = []
        for item in feed:
            tp = item.get("time_published", "")
            if len(tp) < 8:
                continue
            try:
                ts = pd.Timestamp(f"{tp[:4]}-{tp[4:6]}-{tp[6:8]}", tz="UTC")
            except Exception:
                continue
            sentiment = float(item.get("overall_sentiment_score", 0) or 0)
            # ticker-specific sentiment if present
            for ts_obj in (item.get("ticker_sentiment") or []):
                if ts_obj.get("ticker", "").upper() == ticker.upper():
                    try:
                        sentiment = float(ts_obj.get("ticker_sentiment_score", sentiment))
                    except Exception:
                        pass
                    break
            rows.append({
                "ticker": ticker.upper(),
                "ts": ts,
                "sentiment": sentiment,
                "relevance": float((item.get("ticker_sentiment") or [{}])[0].get("relevance_score", 0) or 0),
            })
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["quarter_end"] = df["ts"].dt.to_period("Q").dt.end_time
        g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
            avg_n_news=("sentiment", "size"),
            avg_sent_mean=("sentiment", "mean"),
            avg_sent_min=("sentiment", "min"),
            avg_sent_max=("sentiment", "max"),
            avg_relevance_mean=("relevance", "mean"),
        )
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["quarter_end"] = pd.to_datetime(out["quarter_end"], utc=True)
    # Earnings join (optional)
    if earnings_rows:
        e_rows = []
        for ticker, q_list in earnings_rows:
            for q in q_list:
                try:
                    q_end_raw = q.get("fiscalDateEnding") or ""
                    q_end = pd.Timestamp(q_end_raw, tz="UTC")
                except Exception:
                    continue
                surprise = q.get("surprisePercentage")
                e_rows.append({
                    "ticker": ticker.upper(),
                    "quarter_end": q_end,
                    "avg_eps_surprise_pct": float(surprise) if surprise not in (None, "", "None") else None,
                })
        if e_rows:
            edf = pd.DataFrame(e_rows)
            out = out.merge(edf, on=["ticker", "quarter_end"], how="left")
    return out
