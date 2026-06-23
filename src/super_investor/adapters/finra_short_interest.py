"""FINRA biweekly short-interest CSV download.

FINRA publishes biweekly aggregated short-interest data for every Reg-SHO
covered security. The current location:

  https://cdn.finra.org/equity/regsho/{daily,monthly}/...

Daily file pattern (recent): https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt
Monthly aggregate:           https://cdn.finra.org/equity/regsho/monthly/...

We use the daily files (per-day per-ticker short volume + total volume).
Channel prefix ``sh_*``.

For 5y coverage we'd walk every business day → ~1250 files × 3-5MB each =
multi-GB download. We cap to the last 2 years; the rest is regenerable.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("super_investor.adapters.finra_short_interest")


UA = "super-investor-mirror research@example.com"
BASE = "https://cdn.finra.org/equity/regsho/daily"


def fetch_daily(date: pd.Timestamp, cache_dir: Path | None = None) -> pd.DataFrame:
    """Pull one day's CNMS short-volume file."""
    ymd = date.strftime("%Y%m%d")
    url = f"{BASE}/CNMSshvol{ymd}.txt"
    cp = cache_dir / f"CNMSshvol{ymd}.txt" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    text = None
    if cp and cp.exists() and cp.stat().st_size > 100:
        text = cp.read_text()
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200:
                text = r.text
                if cp:
                    cp.write_text(text)
        except Exception as e:
            log.warning("fetch_daily(%s): %s: %s", url, type(e).__name__, e)
            pass
    if not text:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(text), sep="|")
    except Exception as e:
        log.warning("fetch_daily(%s): %s: %s", url, type(e).__name__, e)
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={
        "Symbol": "ticker",
        "Date": "date",
        "ShortVolume": "short_vol",
        "ShortExemptVolume": "short_exempt_vol",
        "TotalVolume": "total_vol",
        "Market": "market",
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-")
    return df


def fetch_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None = None,
    sleep: float = 0.05,
) -> pd.DataFrame:
    frames = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # weekdays only
            try:
                f = fetch_daily(cur, cache_dir=cache_dir)
                if not f.empty:
                    frames.append(f)
            except Exception:
                pass
            time.sleep(sleep)
        cur = cur + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_to_ticker_quarter(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    df = df[df["ticker"].isin(universe)].copy()
    if df.empty:
        return df
    df["quarter_end"] = df["date"].dt.to_period("Q").dt.end_time
    df["short_pct"] = df["short_vol"] / df["total_vol"].replace(0, 1)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        sh_short_vol_sum=("short_vol", "sum"),
        sh_total_vol_sum=("total_vol", "sum"),
        sh_short_pct_mean=("short_pct", "mean"),
        sh_short_pct_max=("short_pct", "max"),
        sh_n_days=("date", "nunique"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
