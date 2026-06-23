"""Wikipedia per-article monthly pageviews via Wikimedia REST API.

Endpoint:
  https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
    en.wikipedia/all-access/all-agents/<article>/monthly/<YYYYMMDDhh>/<YYYYMMDDhh>

Pageviews ≈ retail attention signal. Strong leading indicator for new
positions in mid/small-cap names that suddenly get press coverage.

Channel prefix ``wp_*``.
"""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UA = "super-investor-mirror research@example.com"
BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents"
)


@dataclass
class WikiViewRow:
    ticker: str
    article: str
    quarter_end: pd.Timestamp
    views_sum: int
    views_max: int


def fetch_pageviews_for_article(
    article: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Pull monthly pageviews for a single article between YYYYMMDD bounds."""
    safe = urllib.parse.quote(article, safe="")
    url = (
        f"{BASE}/{safe}/monthly/{start}00/{end}00"
    )
    cp = (
        cache_dir / f"{article.replace('/', '_')}.json" if cache_dir else None
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists() and cp.stat().st_size > 50:
        import json
        return json.loads(cp.read_text()).get("items", [])
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            return []
        if cp:
            cp.write_text(r.text)
        return r.json().get("items", [])
    except Exception:
        return []


def aggregate_to_ticker_quarter(
    rows: Iterable[tuple[str, str, list[dict]]],
) -> pd.DataFrame:
    """Each input row: (ticker, article, items_list)."""
    out_rows = []
    for ticker, article, items in rows:
        for item in items:
            ts_raw = item.get("timestamp", "")
            if len(ts_raw) < 10:
                continue
            try:
                ts = pd.Timestamp(
                    f"{ts_raw[:4]}-{ts_raw[4:6]}-{ts_raw[6:8]}", tz="UTC"
                )
            except Exception:
                continue
            views = int(item.get("views", 0) or 0)
            q_end = ts.to_period("Q").end_time
            if q_end.tzinfo is None:
                q_end = q_end.tz_localize("UTC")
            out_rows.append({
                "ticker": ticker.upper(),
                "quarter_end": q_end,
                "views": views,
            })
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        wp_views_sum=("views", "sum"),
        wp_views_max=("views", "max"),
        wp_views_mean=("views", "mean"),
        wp_n_months=("views", "size"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
