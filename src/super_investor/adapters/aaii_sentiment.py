"""AAII Investor Sentiment Survey — weekly bull/neutral/bear pct since 1987.

Public CSV at:
  https://www.aaii.com/files/surveys/sentiment.xls
  Also accessible as fallback via the AAII archive page.

Macro-level signal — same value across all tickers per quarter, but adds
regime context for the model. Channel prefix ``aa_*``.
"""
from __future__ import annotations

import io
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

# Direct CSV from AAII archive (weekly data 1987 - present)
PRIMARY = "https://www.aaii.com/files/surveys/sentiment.xls"
# Alt mirror used by some quant repos
FALLBACK = (
    "https://www.aaii.com/sentimentsurvey/sent_results"
)


def fetch_raw(cache_dir: Path | None = None) -> bytes:
    cp = cache_dir / "sentiment.xls" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists() and cp.stat().st_size > 1000:
        return cp.read_bytes()
    r = requests.get(PRIMARY, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    if cp:
        cp.write_bytes(r.content)
    return r.content


def parse_sentiment(raw: bytes) -> pd.DataFrame:
    """Parse AAII xls bytes into a long table.

    AAII file is multi-sheet, top rows are notes, columns vary by year. We:
      1. read first sheet header=None
      2. find row containing 'Bullish' header
      3. take that as column header, drop notes above + below
      4. coerce date + 3 percent columns
    """
    engines = ("xlrd", "openpyxl", None)
    df = None
    last_err: Exception | None = None
    for eng in engines:
        try:
            df = pd.read_excel(io.BytesIO(raw), header=None, engine=eng)
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"all engines failed: {last_err}")
    header_row = None
    for i in range(min(20, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        if row.str.contains("bullish").any():
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    headers = df.iloc[header_row].fillna("").astype(str).str.strip().str.lower().tolist()
    data = df.iloc[header_row + 1 :].copy()
    data.columns = headers
    bull_col = next((c for c in headers if "bull" in c), None)
    bear_col = next((c for c in headers if "bear" in c), None)
    neut_col = next((c for c in headers if "neut" in c), None)
    date_col = next(
        (c for c in headers if "date" in c or c == "" or "reported" in c),
        headers[0],
    )
    if not bull_col:
        return pd.DataFrame()
    out_cols = [c for c in (date_col, bull_col, bear_col, neut_col) if c]
    out = data[out_cols].copy()
    out.columns = ["date", "bullish", "bearish", "neutral"][: len(out_cols)]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ("bullish", "bearish", "neutral"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "bullish"]).reset_index(drop=True)
    return out


def aggregate_to_quarter(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """Returns macro-level row per (ticker, quarter) — every ticker gets same row.

    To avoid blowing up parquet size we emit *only* quarter-end rows; the
    join code in per_investor.py will cross-fill via the macro path.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["quarter_end"] = df["date"].dt.to_period("Q").dt.end_time
    g = df.groupby("quarter_end", as_index=False).agg(
        aa_bull_mean=("bullish", "mean"),
        aa_bear_mean=("bearish", "mean"),
        aa_neutral_mean=("neutral", "mean"),
        aa_bull_max=("bullish", "max"),
        aa_bear_max=("bearish", "max"),
        aa_n_weeks=("bullish", "size"),
    )
    g["aa_bull_bear_ratio"] = g["aa_bull_mean"] / g["aa_bear_mean"].replace(0, 1e-9)
    g["aa_extreme_bull"] = (g["aa_bull_max"] > 50).astype(int)
    g["aa_extreme_bear"] = (g["aa_bear_max"] > 50).astype(int)
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
