"""Macrotrends per-ticker historical metrics scrape — fundamentals deep history.

Per-ticker URLs (slug = ticker/<sanitized-name>):
  https://www.macrotrends.net/stocks/charts/<TICKER>/<slug>/revenue
  https://www.macrotrends.net/stocks/charts/<TICKER>/<slug>/gross-profit
  ...

Static HTML tables with up to 20y data. Channel prefix ``mt_*``.
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


_TABLE_RE = re.compile(r'<table[^>]*id="style-1"[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()


def _to_num(s: str) -> float | None:
    s = _clean(s).replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def fetch_metric(
    ticker: str,
    metric: str,
    slug_guess: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull one metric table for one ticker.

    macrotrends URLs need a sanitized company slug — we try multiple guesses.
    """
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    if slug_guess:
        candidates.append(slug_guess)
    # Fall back to lowercased ticker repeated as slug — macrotrends will 404 if wrong
    candidates += [safe.lower()]
    for slug in candidates:
        url = f"https://www.macrotrends.net/stocks/charts/{safe}/{slug}/{metric}"
        cp = cache_dir / f"{safe}_{metric}.html" if cache_dir else None
        html = None
        if cp and cp.exists() and cp.stat().st_size > 500:
            html = cp.read_text()
        else:
            try:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=20, allow_redirects=True)
                if r.status_code == 200 and "404" not in r.text[:1000].lower():
                    html = r.text
                    if cp:
                        cp.write_text(html)
            except Exception:
                continue
        if not html:
            continue
        df = _parse_table(html, ticker, metric)
        if not df.empty:
            return df
    return pd.DataFrame()


def _parse_table(html: str, ticker: str, metric: str) -> pd.DataFrame:
    m = _TABLE_RE.search(html)
    if not m:
        return pd.DataFrame()
    body = m.group(1)
    rows = _ROW_RE.findall(body)
    out_rows = []
    for row in rows[1:]:
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if len(cells) < 2:
            continue
        date_raw = cells[0]
        val = _to_num(cells[1])
        try:
            ts = pd.Timestamp(date_raw)
        except Exception:
            continue
        out_rows.append({
            "ticker": ticker.upper(),
            "date": ts,
            "metric": metric,
            "value": val,
        })
    return pd.DataFrame(out_rows)


def aggregate_to_ticker_quarter(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    df = rows_df.copy()
    df["quarter_end"] = df["date"].dt.to_period("Q").dt.end_time
    # pivot metric→column
    pivoted = df.pivot_table(
        index=["ticker", "quarter_end"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivoted.columns = [
        c if c in {"ticker", "quarter_end"} else f"mt_{c.replace('-', '_')}"
        for c in pivoted.columns
    ]
    pivoted["quarter_end"] = pd.to_datetime(pivoted["quarter_end"], utc=True)
    return pivoted
