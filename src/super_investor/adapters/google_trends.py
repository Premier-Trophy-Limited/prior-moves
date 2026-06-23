"""Google Trends per-ticker monthly search interest via pytrends.

Channel prefix ``gt_*``. Rate-limited; cap to top-N tickers + 5y window.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class TrendRow:
    ticker: str
    keyword: str
    quarter_end: pd.Timestamp
    interest_mean: float
    interest_max: float


def fetch_ticker(
    ticker: str,
    company_name: str | None = None,
    timeframe: str = "today 5-y",
    cache_dir: Path | None = None,
) -> list[TrendRow]:
    from pytrends.request import TrendReq
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    cp = cache_dir / f"{safe}.parquet" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists():
        try:
            cached = pd.read_parquet(cp)
            return _df_to_rows(cached, safe, company_name or safe)
        except Exception:
            pass
    pytrends = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=1)
    keyword = company_name or safe
    try:
        pytrends.build_payload([keyword], timeframe=timeframe, geo="US")
        df = pytrends.interest_over_time()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    if "isPartial" in df.columns:
        df = df[~df["isPartial"]]
    df = df.reset_index().rename(columns={"date": "date", keyword: "interest"})
    if cp is not None:
        df.to_parquet(cp, index=False)
    return _df_to_rows(df, safe, keyword)


def _df_to_rows(df: pd.DataFrame, ticker: str, keyword: str) -> list[TrendRow]:
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    df["q_end"] = df["date"].dt.to_period("Q").dt.end_time
    out = []
    for q, g in df.groupby("q_end"):
        out.append(TrendRow(
            ticker=ticker,
            keyword=keyword,
            quarter_end=pd.Timestamp(q),
            interest_mean=float(g["interest"].mean()),
            interest_max=float(g["interest"].max()),
        ))
    return out


def aggregate_to_ticker_quarter(rows: Iterable[TrendRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "quarter_end": r.quarter_end,
            "gt_interest_mean": r.interest_mean,
            "gt_interest_max": r.interest_max,
        }
        for r in rows
    ])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
