"""Mimic backtest — simulate buying model picks vs investor's actual picks vs SPY.

For each held-out quarter Q and investor I:
  - my_picks = top-K p_new_entry from runs/per_investor/<I>/holdout_predictions.parquet
  - actual_picks = rows where label == 'new_entry' in same Q
  - spy_baseline = single SPY position
Buy equal-weight at Q close, hold N quarters, sell at Q+N close.
Use yfinance close prices via super_investor.adapters.prices.

Per-quarter return = mean across pick set's quarterly_return values.

Aggregate across quarters with simple arithmetic mean (geometric also reported).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]


@dataclass
class QuarterReturns:
    investor: str
    quarter: pd.Timestamp
    my_picks_n: int
    actual_picks_n: int
    my_picks_return: float
    actual_picks_return: float
    spy_return: float
    hit_tickers_return: float


def _load_actual_new_entries(repo: Path, investor: str) -> pd.DataFrame:
    # Labels encode new_entry directly; the raw per-investor holdings parquet
    # is not needed here (a previous version read it and threw it away).
    labels = pd.read_parquet(repo / "data" / "13f" / "labels.parquet")
    labels["period_of_report"] = pd.to_datetime(labels["period_of_report"])
    return labels[(labels["investor_slug"] == investor) & (labels["label"] == "new_entry")]


def backtest_investor(repo: Path, investor: str, top_k: int = 10,
                      hold_quarters: int = 1,
                      source_subdir: str = "per_investor",
                      preds_name: str = "holdout_predictions.parquet") -> list[QuarterReturns]:
    """``preds_name`` selects the prediction set to rank. The default is the
    filed-holdings holdout (candidate set = the future filing's constituents —
    structurally leaky, see docs/HONESTY_2026-06-10.md). Pass
    ``holdout_predictions_broad.parquet`` for the leak-free broad-universe
    re-score that matches what the live product actually ranks."""
    from super_investor.adapters.prices import quarterly_return

    slug_dir = repo / "runs" / source_subdir / investor
    preds_path = slug_dir / preds_name
    if not preds_path.exists():
        return []
    df = pd.read_parquet(preds_path)
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])

    # Need ticker — join via cusip_map. Drop any pre-existing ticker col so
    # the merge doesn't suffix-rename it to ticker_x / ticker_y.
    cusip_map_path = repo / "data" / "tickers" / "cusip_to_ticker.parquet"
    cm = None
    if cusip_map_path.exists():
        cm = pd.read_parquet(cusip_map_path)[["cusip", "ticker"]].dropna()
        df = df.drop(columns=["ticker"], errors="ignore").merge(cm, on="cusip", how="left")
    actual = _load_actual_new_entries(repo, investor)
    if cm is not None:
        actual = actual.drop(columns=["ticker"], errors="ignore").merge(cm, on="cusip", how="left")

    cache_dir = repo / "data" / "prices_cache"

    out: list[QuarterReturns] = []
    for q in sorted(df["period_of_report"].unique()):
        q_ts = pd.Timestamp(q)
        sub = df[df["period_of_report"] == q].sort_values("p", ascending=False)
        my_picks = sub.head(top_k)
        actual_q = actual[actual["period_of_report"] == q_ts]

        def _portfolio_return(tickers: list[str]) -> float:
            rs = []
            for t in tickers:
                if not isinstance(t, str) or not t:
                    continue
                r = quarterly_return(t, q_ts, cache_dir, hold_quarters=hold_quarters)
                if r == r:
                    rs.append(r)
            return float(np.mean(rs)) if rs else float("nan")

        my_tickers = my_picks["ticker"].dropna().tolist() if "ticker" in my_picks.columns else []
        actual_tickers = actual_q["ticker"].dropna().tolist() if "ticker" in actual_q.columns else []
        hit_tickers = list(set(my_tickers) & set(actual_tickers))

        my_r = _portfolio_return(my_tickers)
        act_r = _portfolio_return(actual_tickers)
        spy_r = quarterly_return("SPY", q_ts, cache_dir, hold_quarters=hold_quarters)
        hit_r = _portfolio_return(hit_tickers)

        out.append(QuarterReturns(
            investor=investor, quarter=q_ts,
            my_picks_n=len(my_tickers),
            actual_picks_n=len(actual_tickers),
            my_picks_return=my_r,
            actual_picks_return=act_r,
            spy_return=float(spy_r) if spy_r == spy_r else float("nan"),
            hit_tickers_return=hit_r,
        ))
    return out


def backtest_all(
    repo: Path = REPO,
    top_k: int = 10,
    hold_quarters: int = 1,
    source_subdir: str = "per_investor",
    preds_name: str = "holdout_predictions.parquet",
    write: bool = True,
) -> pd.DataFrame:
    out_root = repo / "runs" / source_subdir
    if not out_root.exists():
        raise FileNotFoundError(f"{out_root} missing; train per-investor models first")
    rows: list[dict] = []
    for slug_dir in sorted(out_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        investor = slug_dir.name
        try:
            for q in backtest_investor(
                repo, investor, top_k=top_k, hold_quarters=hold_quarters,
                source_subdir=source_subdir, preds_name=preds_name,
            ):
                rows.append({
                    "investor": q.investor, "quarter": q.quarter,
                    "my_picks_n": q.my_picks_n, "actual_picks_n": q.actual_picks_n,
                    "my_picks_return": q.my_picks_return,
                    "actual_picks_return": q.actual_picks_return,
                    "spy_return": q.spy_return,
                    "hit_tickers_return": q.hit_tickers_return,
                })
        except Exception as e:
            print(f"backtest fail {investor}: {e}")
    df = pd.DataFrame(rows)
    if df.empty or not write:
        return df
    out_path = (
        repo / "runs" / source_subdir
        / f"backtest_topK_{top_k}_hold{hold_quarters}q.parquet"
    )
    df.to_parquet(out_path, index=False)
    return df
