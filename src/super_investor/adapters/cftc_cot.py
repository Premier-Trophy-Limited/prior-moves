"""CFTC Commitments of Traders (COT) report — futures positioning macro signal.

Free weekly CSVs at:
  https://www.cftc.gov/dea/futures/deacmesf.htm  (CME futures)
  https://www.cftc.gov/dea/options/deacmeof.htm  (options)

Bulk text archives:
  https://www.cftc.gov/files/dea/history/dea_fut_xls_<YYYY>.zip

We pull the annual zip files for last 5 years and aggregate per-quarter
(non-)commercial net positioning for SP500 e-mini, gold, oil, USD index,
etc. Macro signal — same row per ticker per quarter. Channel prefix ``ct_*``.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UA = "super-investor-mirror research@example.com"


def fetch_annual_zip(year: int, cache_dir: Path | None = None) -> bytes:
    url = f"https://www.cftc.gov/files/dea/history/dea_fut_xls_{year}.zip"
    cp = cache_dir / f"dea_fut_xls_{year}.zip" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists() and cp.stat().st_size > 1000:
        return cp.read_bytes()
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
        if r.status_code == 200:
            if cp:
                cp.write_bytes(r.content)
            return r.content
    except Exception:
        pass
    return b""


def parse_annual_zip(raw: bytes) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
            if not names:
                return pd.DataFrame()
            with zf.open(names[0]) as fh:
                df = pd.read_excel(fh)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def aggregate_to_quarter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Find date + commodity + position columns
    date_col = next((c for c in df.columns if "Report_Date" in c or c == "As_of_Date_In_Form_YYMMDD"), None)
    name_col = next((c for c in df.columns if "Market_and_Exchange_Names" in c or "CFTC_Market_Code" in c), None)
    if not date_col or not name_col:
        return pd.DataFrame()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, name_col])
    df["quarter_end"] = df[date_col].dt.to_period("Q").dt.end_time
    long_col = next((c for c in df.columns if "NonComm_Positions_Long_All" in c), None)
    short_col = next((c for c in df.columns if "NonComm_Positions_Short_All" in c), None)
    if not long_col or not short_col:
        return pd.DataFrame()
    df["ct_net_pos"] = df[long_col] - df[short_col]
    # focus on benchmarks — S&P 500 e-mini, gold, dxy
    bench = df[df[name_col].str.contains(
        "E-MINI S&P 500|GOLD - COMMODITY EXCHANGE|U.S. DOLLAR INDEX",
        regex=True, na=False, case=False,
    )]
    if bench.empty:
        bench = df  # fall back to all
    g = bench.groupby("quarter_end", as_index=False).agg(
        ct_net_pos_mean=("ct_net_pos", "mean"),
        ct_n_contracts=(long_col, "mean"),
        ct_n_markets=(name_col, "nunique"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
