"""ARK Invest daily holdings + trade disclosure CSVs.

Cathie Wood is a real super-investor with full daily disclosure.

Daily holdings CSVs (free, no auth):
  ARKK: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv
  ARKQ: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv
  ARKW: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv
  ARKG: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv
  ARKF: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv
  ARKX: https://ark-funds.com/wp-content/uploads/funds-etf-csv/ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv

Per-day per-ETF: ticker, shares, market_value, weight_pct. Channel prefix ``ak_*``.
"""
from __future__ import annotations

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

ARK_ETFS = {
    "ARKK": "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKQ": "ARK_AUTONOMOUS_TECHNOLOGY_%26_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
    "ARKW": "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "ARKG": "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "ARKF": "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
    "ARKX": "ARK_SPACE_EXPLORATION_%26_INNOVATION_ETF_ARKX_HOLDINGS.csv",
}
BASE = "https://ark-funds.com/wp-content/uploads/funds-etf-csv"


def fetch_etf(etf: str, cache_dir: Path | None = None) -> pd.DataFrame:
    fname = ARK_ETFS.get(etf)
    if not fname:
        return pd.DataFrame()
    url = f"{BASE}/{fname}"
    cp = cache_dir / f"{etf}.csv" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    text = None
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code == 200:
            text = r.text
            if cp:
                cp.write_text(text)
    except Exception:
        pass
    if text is None and cp and cp.exists():
        text = cp.read_text()
    if not text:
        return pd.DataFrame()
    # ARK CSV has trailing footer rows; pandas reads them fine if we
    # filter to rows where 'ticker' is non-null
    import io
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "ticker" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-")
    df["etf"] = etf
    # date column varies: 'date' or 'as_of_date'
    date_col = "date" if "date" in df.columns else next(
        (c for c in df.columns if "date" in c), None
    )
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.Timestamp.utcnow().normalize()
    return df


def fetch_all(cache_dir: Path | None = None, sleep: float = 0.5) -> pd.DataFrame:
    frames = []
    for etf in ARK_ETFS:
        try:
            f = fetch_etf(etf, cache_dir=cache_dir)
            if not f.empty:
                frames.append(f)
                print(f"  {etf}: +{len(f)} rows", flush=True)
        except Exception as e:
            print(f"  {etf}: FAIL {e}", flush=True)
        time.sleep(sleep)
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
    weight_col = next((c for c in df.columns if "weight" in c), None)
    shares_col = next((c for c in df.columns if "shares" in c), None)
    mv_col = next((c for c in df.columns if c in {"market value ($)", "market_value", "mv"}), None)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        ak_n_etfs=("etf", "nunique"),
        ak_n_days=("date", "nunique"),
    )
    if weight_col:
        wm = df.groupby(["ticker", "quarter_end"], as_index=False)[weight_col].agg(["mean", "max"])
        wm.columns = ["ak_weight_mean", "ak_weight_max"]
        g = g.merge(wm.reset_index(), on=["ticker", "quarter_end"], how="left")
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
