"""TipRanks analyst consensus + price targets.

Per-ticker:
  https://www.tipranks.com/stocks/<sym>/forecast

JS-rendered, paywall hints. scrapling.StealthyFetcher.
Channel prefix ``tr_*``.
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
class TipranksSnapshot:
    ticker: str
    n_analysts: int | None
    buy_count: int | None
    hold_count: int | None
    sell_count: int | None
    avg_target: float | None
    quarter_end: pd.Timestamp


_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_BUY_RE = re.compile(r"(\d+)\s*Buys?", re.IGNORECASE)
_HOLD_RE = re.compile(r"(\d+)\s*Holds?", re.IGNORECASE)
_SELL_RE = re.compile(r"(\d+)\s*Sells?", re.IGNORECASE)
_TARGET_RE = re.compile(r"price target[^$]*\$([0-9\.,]+)", re.IGNORECASE)


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "").strip()


def fetch_snapshot(ticker: str, cache_dir: Path | None = None) -> TipranksSnapshot | None:
    SF = _fetcher()
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"https://www.tipranks.com/stocks/{safe.lower()}/forecast"
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
    text = _clean(html)
    buys = int(_BUY_RE.search(text).group(1)) if _BUY_RE.search(text) else None
    holds = int(_HOLD_RE.search(text).group(1)) if _HOLD_RE.search(text) else None
    sells = int(_SELL_RE.search(text).group(1)) if _SELL_RE.search(text) else None
    target = None
    tm = _TARGET_RE.search(text)
    if tm:
        try:
            target = float(tm.group(1).replace(",", ""))
        except Exception:
            target = None
    n_analysts = sum(v for v in (buys, holds, sells) if v is not None) or None
    q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    return TipranksSnapshot(
        ticker=safe,
        n_analysts=n_analysts,
        buy_count=buys,
        hold_count=holds,
        sell_count=sells,
        avg_target=target,
        quarter_end=q,
    )


def aggregate_to_ticker_quarter(
    snaps: Iterable[TipranksSnapshot],
) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s is None:
            continue
        rows.append({
            "ticker": s.ticker,
            "quarter_end": s.quarter_end,
            "tr_n_analysts": s.n_analysts,
            "tr_buys": s.buy_count,
            "tr_holds": s.hold_count,
            "tr_sells": s.sell_count,
            "tr_avg_target": s.avg_target,
            "tr_buy_ratio": (
                (s.buy_count or 0) / s.n_analysts if s.n_analysts else None
            ),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
