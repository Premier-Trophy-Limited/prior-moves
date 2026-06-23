"""WhaleWisdom — 13F filer detail + recent buys.

Per-filer pages (e.g. /filer/scion-asset-management) list current holdings
and quarter-over-quarter changes. JS-rendered; scrapling.StealthyFetcher with
network_idle handles it.

Channel prefix ``ww_*``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


WHALE_FILERS = [
    "berkshire-hathaway",
    "scion-asset-management",
    "baupost-group",
    "pershing-square-capital-management",
    "appaloosa-management",
    "bridgewater-associates",
    "icahn-capital",
    "third-point",
    "gotham-asset-management",
    "tweedy-browne",
    "fairholme-capital-management",
    "fpa-capital-management",
    "first-eagle-investment-management",
    "wallace-r-weitz-co",
    "yacktman-asset-management",
]


@dataclass
class WhaleHolding:
    filer: str
    ticker: str
    quarter_end: pd.Timestamp


def _fetcher():
    from scrapling.fetchers import StealthyFetcher
    return StealthyFetcher


_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_LINK_RE = re.compile(r'/stock/([A-Z\.\-]{1,7})/', re.IGNORECASE)


def fetch_filer_holdings(slug: str, cache_dir: Path | None = None) -> list[WhaleHolding]:
    SF = _fetcher()
    url = f"https://whalewisdom.com/filer/{slug}"
    cp = cache_dir / f"{slug}.html" if cache_dir else None
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
    # default quarter — current
    ts = pd.Timestamp.utcnow().tz_convert("UTC")
    q = ts.to_period("Q").end_time.tz_localize("UTC")
    out: list[WhaleHolding] = []
    seen: set[str] = set()
    for m in _TICKER_LINK_RE.finditer(html):
        sym = m.group(1).upper().replace(".", "-")
        if sym in seen or len(sym) > 5:
            continue
        seen.add(sym)
        out.append(WhaleHolding(filer=slug, ticker=sym, quarter_end=q))
    return out


def aggregate_to_ticker_quarter(
    holdings: Iterable[WhaleHolding],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for h in holdings:
        if h.ticker not in universe or pd.isna(h.quarter_end):
            continue
        rows.append({
            "ticker": h.ticker,
            "quarter_end": h.quarter_end,
            "filer": h.filer,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end", "ww_n_filers", "ww_has_holding",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        ww_n_filers=("filer", "nunique"),
    )
    g["ww_has_holding"] = 1
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
