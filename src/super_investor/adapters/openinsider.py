"""OpenInsider scrape — per-ticker insider trades (alt source to Form 4).

URLs (open, no auth, static HTML):
  https://openinsider.com/screener?s=<sym>&...
  https://openinsider.com/insider-purchases  (cluster purchases)

For per-ticker history we hit /screener?s=<sym>. Channel prefix ``oi_*``.
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
class OpenInsiderTrade:
    ticker: str
    filing_date: pd.Timestamp
    trade_date: pd.Timestamp
    insider_name: str
    title: str
    trade_type: str  # P-Purchase, S-Sale, etc
    price: float | None
    qty: int | None
    value: float | None


_TABLE_RE = re.compile(r'<table[^>]+class="[^"]*tinytable[^"]*"[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return _TAG_RE.sub("", s or "").replace("&nbsp;", " ").strip()


def _num(s: str) -> float | None:
    s = _clean(s).replace(",", "").replace("$", "").replace("+", "")
    try:
        return float(s)
    except Exception:
        return None


def fetch_ticker(ticker: str, cache_dir: Path | None = None) -> list[OpenInsiderTrade]:
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"http://openinsider.com/screener?s={safe}&o=&pl=&ph=&ll=&lh=&fd=730&fdr=&td=730&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1"
    cp = cache_dir / f"{safe}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    html = None
    if cp and cp.exists() and cp.stat().st_size > 200:
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
    m = _TABLE_RE.search(html)
    if not m:
        return []
    body = m.group(1)
    rows = _ROW_RE.findall(body)
    out: list[OpenInsiderTrade] = []
    for row in rows[1:]:
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if len(cells) < 12:
            continue
        # column layout: [X, FilingDate, TradeDate, Ticker, InsiderName, Title, TradeType, Price, Qty, Owned, OwnedDelta, Value]
        try:
            fd = pd.Timestamp(cells[1])
            td = pd.Timestamp(cells[2])
        except Exception:
            continue
        out.append(OpenInsiderTrade(
            ticker=safe,
            filing_date=fd,
            trade_date=td,
            insider_name=cells[4],
            title=cells[5],
            trade_type=cells[6],
            price=_num(cells[7]),
            qty=int(_num(cells[8]) or 0),
            value=_num(cells[11]) if len(cells) > 11 else None,
        ))
    return out


def aggregate_to_ticker_quarter(
    trades: Iterable[OpenInsiderTrade],
) -> pd.DataFrame:
    rows = []
    for t in trades:
        q_end = t.trade_date.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        is_buy = "P" in (t.trade_type or "").upper()
        is_sell = "S" in (t.trade_type or "").upper()
        rows.append({
            "ticker": t.ticker,
            "quarter_end": q_end,
            "is_buy": int(is_buy),
            "is_sell": int(is_sell),
            "value": t.value or 0.0,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        oi_n_buys=("is_buy", "sum"),
        oi_n_sells=("is_sell", "sum"),
        oi_total_value=("value", "sum"),
        oi_max_value=("value", "max"),
    )
    g["oi_buy_sell_ratio"] = g["oi_n_buys"] / (g["oi_n_sells"].replace(0, 1))
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
