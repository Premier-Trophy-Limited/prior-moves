"""Insider Monkey hedge fund holdings scrape — alt source to DataRoma.

Per-fund pages:
  https://www.insidermonkey.com/hedge-fund/<slug>/

Static HTML, no anti-bot. Channel prefix ``im_*``.
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

# Major value-investor + activist hedge funds tracked on IM
IM_FUNDS = [
    "berkshire-hathaway-warren-buffett-262",
    "baupost-group-seth-klarman-87",
    "pershing-square-capital-management-bill-ackman-99",
    "third-point-daniel-loeb-95",
    "appaloosa-management-david-tepper-1",
    "icahn-capital-carl-icahn-1009",
    "scion-asset-management-michael-burry-1099",
    "soros-fund-management-george-soros-150",
    "tudor-investment-paul-tudor-jones-39",
    "viking-global-investors-andreas-halvorsen-95",
    "two-sigma-advisors-john-overdeck-david-siegel-2042",
    "renaissance-technologies-james-simons-187",
]


@dataclass
class IMHolding:
    fund: str
    ticker: str
    quarter_end: pd.Timestamp


_TICKER_RE = re.compile(r'<a[^>]+href="[^"]*/(?:stock|insider)-trading/[A-Z0-9.\-]+/?[^"]*"[^>]*>([A-Z]{1,5})</a>', re.IGNORECASE)
_TICKER_FALLBACK_RE = re.compile(r'<td[^>]*>\s*<a[^>]*>([A-Z]{1,5})</a>\s*</td>')


def fetch_fund(slug: str, cache_dir: Path | None = None) -> list[IMHolding]:
    url = f"https://www.insidermonkey.com/hedge-fund/{slug}/"
    cp = cache_dir / f"{slug}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    html = None
    if cp and cp.exists() and cp.stat().st_size > 1000:
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
    # take current quarter
    q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    matches = _TICKER_RE.findall(html) or _TICKER_FALLBACK_RE.findall(html)
    seen: set[str] = set()
    out: list[IMHolding] = []
    for sym in matches:
        sym = sym.upper().replace(".", "-")
        if sym in seen or len(sym) > 5:
            continue
        seen.add(sym)
        out.append(IMHolding(fund=slug, ticker=sym, quarter_end=q))
    return out


def aggregate_to_ticker_quarter(
    holdings: Iterable[IMHolding], universe: set[str]
) -> pd.DataFrame:
    rows = []
    for h in holdings:
        if h.ticker not in universe:
            continue
        rows.append({
            "ticker": h.ticker,
            "quarter_end": h.quarter_end,
            "fund": h.fund,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        im_n_funds=("fund", "nunique"),
    )
    g["im_has_holding"] = 1
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
