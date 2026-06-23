"""Ken French Data Library adapter — Fama-French 5 factors + momentum, daily.

Free public research data (Dartmouth / Tuck). No API key, no billing. $0. Used to
build a factor-adjusted event label so we isolate event alpha from market / size /
value / profitability / investment / momentum beta.

Source CSVs (zipped):
  5 factors daily : F-F_Research_Data_5_Factors_2x3_daily
  momentum daily  : F-F_Momentum_Factor_daily

Output frame columns (daily, decimal returns — the library publishes percent, we
divide by 100): date, mkt_rf, smb, hml, rmw, cma, mom, rf.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger(__name__)

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_FIVE = f"{_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM = f"{_BASE}/F-F_Momentum_Factor_daily_CSV.zip"

FACTOR_COLS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")


def _fetch_zip_csv(url: str) -> bytes:
    with httpx.Client(timeout=60.0, follow_redirects=True,
                      headers={"User-Agent": "super-investor-mirror researcher"}) as c:
        r = c.get(url)
        r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = zf.namelist()[0]
    return zf.read(name)


def _parse_french_daily(raw: bytes) -> pd.DataFrame:
    """Parse a Ken French daily CSV: skip the text preamble, read the dated
    block, stop at the first non-date row (annual section / footer)."""
    text = raw.decode("latin-1")
    rows: list[dict] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        first = parts[0]
        if header is None:
            # header row = first line containing a known factor token (the
            # momentum file has just one data column, so don't gate on width)
            if any(k in line for k in ("Mkt-RF", "Mom", "SMB", "HML", "RMW")):
                header = parts
            continue
        if not (first.isdigit() and len(first) == 8):
            # left the daily date block (blank line or annual YYYY rows)
            if rows:
                break
            continue
        try:
            vals = [float(x) for x in parts[1:] if x != ""]
        except ValueError:
            continue
        rows.append({"date": pd.Timestamp(first), "_vals": vals, "_cols": header[1:]})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame([{"date": r["date"],
                         **{c.strip(): v for c, v in zip(r["_cols"], r["_vals"])}}
                        for r in rows])
    return out


def fetch_factors(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Daily 5-factor + momentum frame, cached. Percent → decimal."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "factors_daily.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    five = _parse_french_daily(_fetch_zip_csv(_FIVE))
    mom = _parse_french_daily(_fetch_zip_csv(_MOM))
    if five.empty:
        log.warning("ken_french: 5-factor parse empty")
        return pd.DataFrame()

    five = five.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
                                "RMW": "rmw", "CMA": "cma", "RF": "rf"})
    df = five
    if not mom.empty:
        mom = mom.rename(columns={mom.columns[1]: "mom"})[["date", "mom"]]
        df = df.merge(mom, on="date", how="left")
    keep = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in df.columns:
        if c != "date":
            df[c] = df[c] / 100.0  # library publishes percent
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    log.info("ken_french: %d daily rows %s..%s", len(df),
             df["date"].min().date(), df["date"].max().date())
    return df
