"""StockTwits historical per-ticker page scrape.

The API only returns ~30 most-recent messages per symbol. The web page
``stocktwits.com/symbol/<sym>`` loads more on infinite scroll; with
scrapling.StealthyFetcher + network_idle we get a deeper snapshot.

Channel prefix ``sth_*``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _fetcher():
    from scrapling.fetchers import StealthyFetcher
    return StealthyFetcher


@dataclass
class StocktwitsCount:
    ticker: str
    n_messages: int
    n_bullish: int
    n_bearish: int
    quarter_end: pd.Timestamp


_TAG_RE = re.compile(r"<[^>]+>")
_BULL_RE = re.compile(r"bullish", re.IGNORECASE)
_BEAR_RE = re.compile(r"bearish", re.IGNORECASE)
_MSG_BLOCK = re.compile(r'data-testid="StreamMessage"', re.IGNORECASE)


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "")


def fetch_symbol(ticker: str, cache_dir: Path | None = None) -> StocktwitsCount | None:
    SF = _fetcher()
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"https://stocktwits.com/symbol/{safe}"
    cp = cache_dir / f"{safe}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    html = None
    if cp and cp.exists():
        html = cp.read_text()
    else:
        try:
            resp = SF.fetch(url, headless=True, network_idle=True, timeout=60_000)
            html = resp.html_content if hasattr(resp, "html_content") else str(resp)
            if cp and html:
                cp.write_text(html)
        except Exception:
            return None
    if not html:
        return None
    n_msgs = len(_MSG_BLOCK.findall(html))
    n_bull = len(_BULL_RE.findall(html))
    n_bear = len(_BEAR_RE.findall(html))
    q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    return StocktwitsCount(
        ticker=safe,
        n_messages=n_msgs,
        n_bullish=n_bull,
        n_bearish=n_bear,
        quarter_end=q,
    )


def aggregate_to_ticker_quarter(counts: Iterable[StocktwitsCount]) -> pd.DataFrame:
    rows = []
    for c in counts:
        if c is None:
            continue
        rows.append({
            "ticker": c.ticker,
            "quarter_end": c.quarter_end,
            "sth_n_msgs": c.n_messages,
            "sth_n_bullish": c.n_bullish,
            "sth_n_bearish": c.n_bearish,
            "sth_bull_bear_ratio": (
                c.n_bullish / max(c.n_bearish, 1)
            ),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
