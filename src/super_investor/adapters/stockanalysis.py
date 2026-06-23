"""stockanalysis.com — clean fundamentals tables.

Per-ticker URLs:
  https://stockanalysis.com/stocks/<symbol>/financials/                — quarterly income stmt
  https://stockanalysis.com/stocks/<symbol>/financials/balance-sheet/  — balance sheet
  https://stockanalysis.com/stocks/<symbol>/financials/cash-flow/      — cash flow
  https://stockanalysis.com/stocks/<symbol>/statistics/                — key ratios

Static HTML tables, no anti-bot. Channel prefix ``sa_*``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE = "https://stockanalysis.com/stocks"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


def _get(url: str, retries: int = 3, sleep: float = 1.0) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"GET {url} failed: {last}")


_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"[^\d\.\-]")


def _clean(s: str) -> str:
    return _TAG_RE.sub("", s or "").replace("&nbsp;", " ").strip()


def _to_num(s: str) -> float | None:
    s = _clean(s).replace(",", "").replace("$", "").replace("%", "")
    if s in {"", "-", "—", "n/a", "N/A"}:
        return None
    try:
        return float(_NUM_RE.sub("", s))
    except Exception:
        return None


def _parse_period(label: str) -> pd.Timestamp:
    """e.g. 'Mar 31, 2026' or 'Q1 2026' or '12/31/2025'."""
    label = _clean(label)
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(pd.to_datetime(label, format=fmt)).tz_localize("UTC")
        except Exception:
            pass
    m = re.match(r"Q(\d)\s*(\d{4})", label)
    if m:
        q, y = m.group(1), m.group(2)
        md = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[q]
        return pd.Timestamp(f"{y}-{md}", tz="UTC")
    return pd.NaT


def fetch_quarterly_financials(symbol: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch quarterly income statement table for one ticker."""
    safe = symbol.upper().replace("/", "-").replace(".", "-")
    url = f"{BASE}/{safe.lower()}/financials/?p=quarterly"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"{safe}_financials_q.html"
        if cp.exists():
            html = cp.read_text()
        else:
            html = _get(url)
            if html:
                cp.write_text(html)
    else:
        html = _get(url)
    if not html:
        return pd.DataFrame()
    return _parse_financial_table(html, symbol)


def _parse_financial_table(html: str, symbol: str) -> pd.DataFrame:
    tables = _TABLE_RE.findall(html)
    for tbl in tables:
        rows = _ROW_RE.findall(tbl)
        if len(rows) < 4:
            continue
        header_cells = [_clean(c) for c in _CELL_RE.findall(rows[0])]
        # need first cell="Fiscal Period" or similar; subsequent are period labels
        if not header_cells or "fiscal" not in header_cells[0].lower():
            continue
        periods = [_parse_period(c) for c in header_cells[1:]]
        recs: dict[pd.Timestamp, dict[str, float | None]] = {p: {} for p in periods if not pd.isna(p)}
        for row in rows[1:]:
            cells = [_clean(c) for c in _CELL_RE.findall(row)]
            if not cells:
                continue
            metric = cells[0].lower()
            values = cells[1:]
            for i, p in enumerate(periods):
                if pd.isna(p) or i >= len(values):
                    continue
                recs[p][metric] = _to_num(values[i])
        rows_out = []
        for p, m in recs.items():
            rows_out.append({
                "ticker": symbol.upper(),
                "quarter_end": p,
                "revenue": m.get("revenue"),
                "rev_growth_yoy": m.get("revenue growth (yoy)"),
                "eps_diluted": m.get("eps (diluted)"),
                "operating_income": m.get("operating income"),
                "net_income": m.get("net income"),
                "fcf": m.get("free cash flow"),
                "gross_margin": m.get("gross margin"),
            })
        return pd.DataFrame(rows_out)
    return pd.DataFrame()


def aggregate_to_ticker_quarter(rows: Iterable[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.concat([r for r in rows if not r.empty], ignore_index=True)
    if df.empty:
        return df
    df = df.rename(columns={
        "revenue": "sa_revenue",
        "rev_growth_yoy": "sa_rev_growth_yoy",
        "eps_diluted": "sa_eps_diluted",
        "operating_income": "sa_op_income",
        "net_income": "sa_net_income",
        "fcf": "sa_fcf",
        "gross_margin": "sa_gross_margin",
    })
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
