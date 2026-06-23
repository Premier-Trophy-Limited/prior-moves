"""StockTwits trending symbols + per-symbol stream sentiment.

The public trending endpoint is open + paginated, separate from the per-
symbol page that 403'd. Hits:

  https://api.stocktwits.com/api/2/trending/symbols.json
  https://api.stocktwits.com/api/2/streams/symbol/<SYM>.json

Channel prefix ``stt_*`` (stocktwits-trending, distinct from sth_* historical).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class StStream:
    ticker: str
    n_messages: int
    n_bullish: int
    n_bearish: int
    quarter_end: pd.Timestamp


def fetch_trending(cache_dir: Path | None = None) -> list[str]:
    url = "https://api.stocktwits.com/api/2/trending/symbols.json"
    cp = cache_dir / "trending.json" if cache_dir else None
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            return []
        if cp and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cp.write_text(r.text)
        data = r.json()
    except Exception:
        if cp and cp.exists():
            import json
            data = json.loads(cp.read_text())
        else:
            return []
    return [s.get("symbol", "").upper() for s in data.get("symbols", [])]


def fetch_symbol_stream(symbol: str, cache_dir: Path | None = None) -> StStream | None:
    safe = symbol.upper().replace("/", "-").replace(".", "-")
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{safe}.json"
    cp = cache_dir / f"{safe}.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            return None
        if cp:
            cp.write_text(r.text)
        data = r.json()
    except Exception:
        if cp and cp.exists():
            import json
            data = json.loads(cp.read_text())
        else:
            return None
    msgs = data.get("messages", [])

    def _sentiment(msg: dict) -> str | None:
        ent = msg.get("entities") or {}
        sent = ent.get("sentiment")
        if not isinstance(sent, dict):
            return None
        return sent.get("basic")

    bull = sum(1 for m in msgs if _sentiment(m) == "Bullish")
    bear = sum(1 for m in msgs if _sentiment(m) == "Bearish")
    q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    return StStream(ticker=safe, n_messages=len(msgs), n_bullish=bull, n_bearish=bear, quarter_end=q)


def aggregate_to_ticker_quarter(streams: Iterable[StStream]) -> pd.DataFrame:
    rows = []
    for s in streams:
        if s is None:
            continue
        rows.append({
            "ticker": s.ticker,
            "quarter_end": s.quarter_end,
            "stt_n_msgs": s.n_messages,
            "stt_n_bull": s.n_bullish,
            "stt_n_bear": s.n_bearish,
            "stt_bull_bear_ratio": s.n_bullish / max(s.n_bearish, 1),
            "stt_is_trending": 1,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
