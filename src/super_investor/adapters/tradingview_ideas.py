"""TradingView per-ticker community ideas scrape via scrapling stealth.

Per-ticker page:
  https://www.tradingview.com/symbols/NASDAQ-<sym>/ideas/
  https://www.tradingview.com/symbols/NYSE-<sym>/ideas/

Pages list ideas with author, title, like count, published date. Channel
prefix ``tv_*``. JS-heavy → scrapling.StealthyFetcher.
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


_TAG_RE = re.compile(r"<[^>]+>")
_LIKE_RE = re.compile(r'"likes_count":\s*(\d+)', re.IGNORECASE)
_DATE_RE = re.compile(r'datetime="([^"]+)"', re.IGNORECASE)


@dataclass
class TVIdea:
    ticker: str
    n_ideas: int
    likes_sum: int
    quarter_end: pd.Timestamp


def fetch_symbol(ticker: str, exchange: str = "NASDAQ", cache_dir: Path | None = None) -> TVIdea | None:
    SF = _fetcher()
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"https://www.tradingview.com/symbols/{exchange}-{safe}/ideas/"
    cp = cache_dir / f"{exchange}-{safe}.html" if cache_dir else None
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
    likes = sum(int(m.group(1)) for m in _LIKE_RE.finditer(html))
    n_ideas = len(_LIKE_RE.findall(html))
    q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    return TVIdea(ticker=safe, n_ideas=n_ideas, likes_sum=likes, quarter_end=q)


def aggregate_to_ticker_quarter(items: Iterable[TVIdea]) -> pd.DataFrame:
    rows = []
    for it in items:
        if it is None:
            continue
        rows.append({
            "ticker": it.ticker,
            "quarter_end": it.quarter_end,
            "tv_n_ideas": it.n_ideas,
            "tv_likes_sum": it.likes_sum,
            "tv_likes_per_idea": (it.likes_sum / max(it.n_ideas, 1)),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
