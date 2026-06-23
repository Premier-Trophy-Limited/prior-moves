"""GDELT news-tone per company via the free DOC 2.0 API (no BigQuery, no key).

https://api.gdeltproject.org/api/v2/doc/doc
    ?query="<company>"&mode=timelinetone&format=json
    &startdatetime=YYYYMMDDHHMMSS&enddatetime=...

Returns a daily average-tone timeline (GDELT tone: roughly -100..+100, where
negative = adverse coverage). We aggregate to (ticker, quarter): mean tone,
tone volatility, and article volume. Channel prefix ``gl_``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

UA = "super-investor-mirror research@example.com"
API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_tone_timeline(
    company: str,
    start: str = "20210601000000",
    end: str = "20260601000000",
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> pd.DataFrame:
    key = (cache_key or company).replace("/", "_").replace(" ", "_")[:60]
    cp = cache_dir / f"{key}.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    data = None
    if cp and cp.exists() and cp.stat().st_size > 20:
        import json
        try:
            data = json.loads(cp.read_text())
        except Exception:
            data = None
    if data is None:
        params = {
            "query": f'"{company}"',
            "mode": "timelinetone",
            "format": "json",
            "startdatetime": start,
            "enddatetime": end,
        }
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
            if r.status_code != 200 or not r.text.strip().startswith("{"):
                return pd.DataFrame()
            data = r.json()
            if cp:
                cp.write_text(r.text)
        except Exception:
            return pd.DataFrame()
    series = (data.get("timeline") or [{}])
    points = series[0].get("data", []) if series else []
    rows = []
    for pt in points:
        try:
            ts = pd.Timestamp(pt["date"])
            rows.append({"date": ts, "tone": float(pt["value"])})
        except Exception:
            continue
    return pd.DataFrame(rows)


def aggregate(rows: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    out = []
    for ticker, tl in rows:
        if tl is None or tl.empty:
            continue
        tl = tl.copy()
        tl["quarter_end"] = tl["date"].dt.to_period("Q").dt.end_time
        g = tl.groupby("quarter_end", as_index=False).agg(
            gl_tone_mean=("tone", "mean"),
            gl_tone_std=("tone", "std"),
            gl_tone_min=("tone", "min"),
            gl_n_days=("tone", "size"),
        )
        g["ticker"] = ticker.upper()
        out.append(g)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
