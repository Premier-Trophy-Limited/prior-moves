"""DataRoma scraper — super-investor portfolios + recent buys.

DataRoma.com mirrors 13F holdings for ~80 super-investors (Buffett, Klarman,
Ackman, Burry, Loeb, Greenblatt, Pabrai, Watsa, etc.) and surfaces:

  https://www.dataroma.com/m/managers.php           — manager list
  https://www.dataroma.com/m/holdings.php?m=BRK     — Berkshire holdings
  https://www.dataroma.com/m/g/portfolio_b.php?m=BRK&L=  — buys/adds/sells diff page
  https://www.dataroma.com/m/g/portfolio_h.php?m=BRK     — historical timeline

Pure static HTML, no anti-bot. scrapling.Fetcher is fine.

Channel prefix: ``dr_*`` (DataRoma).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE = "https://www.dataroma.com/m"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class DataromaHolding:
    manager: str
    ticker: str
    portfolio_pct: float
    shares: int | None
    activity: str  # buy, add, reduce, sold-out, hold, '' (snapshot)
    activity_pct: float | None
    quarter_end: pd.Timestamp


def _get(url: str, retries: int = 3, sleep: float = 1.0) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"GET {url} failed: {last}")


_NUM = re.compile(r"[^0-9.\-]")


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace("%", "")
    if not s or s in {"-", "—"}:
        return None
    try:
        return float(_NUM.sub("", s))
    except Exception:
        return None


def _to_int(s: str) -> int | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if not s or s in {"-", "—"}:
        return None
    try:
        return int(s)
    except Exception:
        return None


_MANAGER_RE = re.compile(
    r'<a href="/?m?/?holdings\.php\?m=([A-Za-z0-9_]+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)


def list_managers(html: str | None = None) -> list[tuple[str, str]]:
    """Return list of (manager_code, manager_name) — e.g. ('BRK', 'Berkshire Hathaway')."""
    if html is None:
        html = _get(f"{BASE}/managers.php")
    pairs = _MANAGER_RE.findall(html)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for code, name in pairs:
        if code in seen:
            continue
        seen.add(code)
        out.append((code, name.strip()))
    return out


_ROW_TICKER_RE = re.compile(
    r'<a href="/m/stock\.php\?sym=([A-Z\.\-]+)"', re.IGNORECASE,
)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_QUARTER_RE = re.compile(r"Q([1-4])\s*(\d{4})", re.IGNORECASE)
_QUARTER_FMT = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}


def _parse_quarter(html: str) -> pd.Timestamp:
    m = _QUARTER_RE.search(html)
    if not m:
        return pd.NaT
    q, yr = m.group(1), m.group(2)
    md = _QUARTER_FMT.get(q)
    if not md:
        return pd.NaT
    return pd.Timestamp(f"{yr}-{md}", tz="UTC")


def _strip(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()


def fetch_manager_holdings(manager_code: str, cache_dir: Path | None = None) -> list[DataromaHolding]:
    """Snapshot of one super-investor's current holdings.

    Row layout (table#grid columns):
      0=hist, 1=stock, 2=port%, 3=recent activity, 4=shares, 5=$price, 6=$value, ...
    """
    url = f"{BASE}/holdings.php?m={manager_code}"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"holdings_{manager_code}.html"
        if cp.exists():
            html = cp.read_text()
        else:
            html = _get(url)
            cp.write_text(html)
    else:
        html = _get(url)
    quarter = _parse_quarter(html)
    # restrict to <tbody>...</tbody> region of #grid to skip header
    tb_start = html.find("<tbody>")
    tb_end = html.find("</tbody>", tb_start) if tb_start > 0 else -1
    body = html[tb_start:tb_end] if tb_start > 0 and tb_end > 0 else html
    out: list[DataromaHolding] = []
    for row in re.split(r"<tr[^>]*>", body, flags=re.IGNORECASE)[1:]:
        tm = _ROW_TICKER_RE.search(row)
        if not tm:
            continue
        ticker = tm.group(1).upper().replace(".", "-")
        cells = _TD_RE.findall(row)
        if len(cells) < 5:
            continue
        port_pct = _to_float(_strip(cells[2]))
        activity = _strip(cells[3])
        shares = _to_int(_strip(cells[4]))
        act_label, act_pct = _parse_activity(activity)
        out.append(DataromaHolding(
            manager=manager_code,
            ticker=ticker,
            portfolio_pct=port_pct or 0.0,
            shares=shares,
            activity=act_label,
            activity_pct=act_pct,
            quarter_end=quarter,
        ))
    return out


_ACT_RE = re.compile(r"(Buy|Add|Reduce|Sold Out|New|Hold)\s*([+\-]?[\d\.]+%)?", re.IGNORECASE)


def _parse_activity(text: str) -> tuple[str, float | None]:
    if not text:
        return "", None
    m = _ACT_RE.search(text)
    if not m:
        return text.strip().lower(), None
    label = m.group(1).lower().replace(" ", "_")
    pct = _to_float(m.group(2)) if m.group(2) else None
    return label, pct


def aggregate_holdings_to_ticker_quarter(
    holdings: Iterable[DataromaHolding],
    universe: set[str],
) -> pd.DataFrame:
    """Per-(ticker, quarter_end) features prefix ``dr_``."""
    rows = []
    for h in holdings:
        if not h.ticker or h.ticker not in universe:
            continue
        if pd.isna(h.quarter_end):
            continue
        rows.append({
            "ticker": h.ticker,
            "quarter_end": h.quarter_end,
            "manager": h.manager,
            "portfolio_pct": h.portfolio_pct,
            "activity": h.activity,
            "is_new_entry": int(h.activity in {"new", "buy"} and (h.activity_pct or 0) > 0),
            "is_add": int(h.activity == "add"),
            "is_reduce": int(h.activity == "reduce"),
            "is_sold_out": int(h.activity in {"sold_out", "sold out"}),
            "activity_pct": h.activity_pct or 0.0,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "dr_n_managers", "dr_new_entries", "dr_adds",
            "dr_reduces", "dr_sold_outs", "dr_avg_port_pct",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        dr_n_managers=("manager", "nunique"),
        dr_new_entries=("is_new_entry", "sum"),
        dr_adds=("is_add", "sum"),
        dr_reduces=("is_reduce", "sum"),
        dr_sold_outs=("is_sold_out", "sum"),
        dr_avg_port_pct=("portfolio_pct", "mean"),
        dr_max_port_pct=("portfolio_pct", "max"),
        dr_total_port_pct=("portfolio_pct", "sum"),
        dr_avg_activity_pct=("activity_pct", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
