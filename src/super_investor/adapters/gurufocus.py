"""GuruFocus super-investor portfolios via scrapling StealthyFetcher.

Per-guru pages mirror DataRoma but with deeper history. Channel prefix ``gf_*``.

URLs:
  https://www.gurufocus.com/guru/warren-buffett/current-portfolio/portfolio
  https://www.gurufocus.com/guru/<slug>/quarterly-buys
  https://www.gurufocus.com/guru/<slug>/historical-portfolio

This is a best-effort scrape: GuruFocus uses Cloudflare + paywall on deep data.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


GURUS = [
    "warren-buffett",
    "charlie-munger",
    "seth-klarman",
    "bill-ackman",
    "michael-burry",
    "david-tepper",
    "ray-dalio",
    "carl-icahn",
    "daniel-loeb",
    "joel-greenblatt",
    "li-lu",
    "mohnish-pabrai",
    "prem-watsa",
    "howard-marks",
    "stanley-druckenmiller",
]


@dataclass
class GuruHolding:
    guru: str
    ticker: str
    portfolio_pct: float
    activity: str
    quarter_end: pd.Timestamp


def _fetcher():
    from scrapling.fetchers import StealthyFetcher
    return StealthyFetcher


_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"\(([A-Z\.\-]{1,7})\)")


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "").strip()


def fetch_guru_portfolio(slug: str, cache_dir: Path | None = None) -> list[GuruHolding]:
    SF = _fetcher()
    url = f"https://www.gurufocus.com/guru/{slug}/current-portfolio/portfolio"
    cp = cache_dir / f"{slug}_current.html" if cache_dir else None
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
            return []
    if not html:
        return []
    text = _clean(html)
    quarter = _guess_quarter(text)
    out: list[GuruHolding] = []
    for m in _TICKER_RE.finditer(text):
        sym = m.group(1).upper().replace(".", "-")
        if len(sym) > 5 or sym in {"USD", "EUR", "GBP", "ETF", "NYSE", "NASDAQ"}:
            continue
        out.append(GuruHolding(
            guru=slug,
            ticker=sym,
            portfolio_pct=0.0,
            activity="",
            quarter_end=quarter,
        ))
    # dedup
    seen = set()
    dedup: list[GuruHolding] = []
    for h in out:
        if h.ticker in seen:
            continue
        seen.add(h.ticker)
        dedup.append(h)
    return dedup


_QUARTER_RE = re.compile(r"Q([1-4])\s*(\d{4})")


def _guess_quarter(text: str) -> pd.Timestamp:
    m = _QUARTER_RE.search(text)
    if not m:
        # default current quarter
        ts = pd.Timestamp.utcnow().tz_convert("UTC")
        return ts.to_period("Q").end_time.tz_localize("UTC")
    q, y = m.group(1), m.group(2)
    md = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[q]
    return pd.Timestamp(f"{y}-{md}", tz="UTC")


def aggregate_to_ticker_quarter(
    holdings: Iterable[GuruHolding],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for h in holdings:
        if h.ticker not in universe or pd.isna(h.quarter_end):
            continue
        rows.append({
            "ticker": h.ticker,
            "quarter_end": h.quarter_end,
            "guru": h.guru,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "gf_n_gurus", "gf_has_holding",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        gf_n_gurus=("guru", "nunique"),
    )
    g["gf_has_holding"] = 1
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
