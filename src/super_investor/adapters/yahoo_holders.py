"""Yahoo Finance institutional + major holders scrape.

Per-ticker pages:
  https://finance.yahoo.com/quote/<sym>/holders
  https://finance.yahoo.com/quote/<sym>/institutional-holders

Shows top-10 institutional holders + shares + stake %.

Channel prefix ``yh2_*`` (yh already used by yfinance_historical).
"""
from __future__ import annotations

import re
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
class HolderRow:
    ticker: str
    holder: str
    shares: int
    pct_held: float
    value_usd: float
    quarter_end: pd.Timestamp


_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return _TAG_RE.sub("", s or "").replace("&nbsp;", " ").strip()


def _to_int(s: str) -> int | None:
    s = _clean(s).replace(",", "")
    try:
        return int(s)
    except Exception:
        return None


def _to_float(s: str) -> float | None:
    s = _clean(s).replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(s)
    except Exception:
        return None


def fetch_ticker(ticker: str, cache_dir: Path | None = None) -> list[HolderRow]:
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"https://finance.yahoo.com/quote/{safe}/holders"
    cp = cache_dir / f"{safe}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    html = None
    if cp and cp.exists() and cp.stat().st_size > 500:
        html = cp.read_text()
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 200:
                html = r.text
                if cp:
                    cp.write_text(html)
        except Exception:
            return []
    if not html:
        return []
    q_end = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    out: list[HolderRow] = []
    # there are multiple tables; pick ones with "Holder" column
    for table in _TABLE_RE.findall(html):
        rows = _ROW_RE.findall(table)
        if not rows:
            continue
        header = [_clean(c) for c in _CELL_RE.findall(rows[0])]
        if not any("Holder" in h or "holder" in h for h in header):
            continue
        for row in rows[1:]:
            cells = [_clean(c) for c in _CELL_RE.findall(row)]
            if len(cells) < 4:
                continue
            shares = _to_int(cells[1])
            pct = _to_float(cells[-2]) if len(cells) >= 5 else _to_float(cells[2])
            val = _to_float(cells[-1])
            if shares is None:
                continue
            out.append(HolderRow(
                ticker=safe,
                holder=cells[0][:80],
                shares=shares,
                pct_held=pct or 0.0,
                value_usd=val or 0.0,
                quarter_end=q_end,
            ))
    return out


def aggregate_to_ticker_quarter(rows: Iterable[HolderRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ticker": r.ticker,
        "quarter_end": r.quarter_end,
        "shares": r.shares,
        "pct_held": r.pct_held,
        "value_usd": r.value_usd,
    } for r in rows])
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        yh2_n_holders=("shares", "size"),
        yh2_total_shares=("shares", "sum"),
        yh2_max_pct=("pct_held", "max"),
        yh2_sum_pct=("pct_held", "sum"),
        yh2_total_value=("value_usd", "sum"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
