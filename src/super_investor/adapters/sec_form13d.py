"""SEC Schedule 13D/13G activist disclosure aggregates per ticker × quarter.

Reuses the existing edgar_forms.py infrastructure for 13D-direct filings.
Channel prefix ``ad_*`` (activist disclosure).

13D = active intent (>5% stake + intent to influence), 13G = passive >5%.
Both are leading indicators of major position changes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UA = "super-investor-mirror research@example.com"


def fetch_recent_13d_for_cik(cik: str, after: pd.Timestamp, cache_dir: Path | None = None) -> list[dict]:
    """Pull recent 13D/G filings for one issuer CIK."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{int(cik):010d}.json" if cache_dir else None
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    if cp and cp.exists() and cp.stat().st_size > 500:
        import json
        data = json.loads(cp.read_text())
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code != 200:
                return []
            data = r.json()
            if cp:
                cp.write_text(r.text)
        except Exception:
            return []
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", [])
    fdates = recent.get("filingDate", [])
    out = []
    for i, frm in enumerate(forms):
        if frm not in {"SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}:
            continue
        try:
            fd = pd.Timestamp(fdates[i])
        except Exception:
            continue
        if fd < after:
            continue
        out.append({"form": frm, "filing_date": fd, "cik": str(cik)})
    return out


def aggregate_to_ticker_quarter(
    rows: Iterable[dict],
    ticker_by_cik: dict[str, str],
) -> pd.DataFrame:
    out_rows = []
    for r in rows:
        cik = str(int(r["cik"]))
        tk = ticker_by_cik.get(cik)
        if not tk:
            continue
        fd = r["filing_date"]
        q_end = fd.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        out_rows.append({
            "ticker": tk.upper(),
            "quarter_end": q_end,
            "is_13d": int(r["form"].startswith("SC 13D")),
            "is_13g": int(r["form"].startswith("SC 13G")),
        })
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        ad_n_13d=("is_13d", "sum"),
        ad_n_13g=("is_13g", "sum"),
    )
    g["ad_n_total"] = g["ad_n_13d"] + g["ad_n_13g"]
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
