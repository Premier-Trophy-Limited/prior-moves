"""Factor-adjusted event label — isolate event alpha from factor beta.

The default event label is raw ``return_ticker - return_SPY``, which still
carries the name's market / size / value / profitability / investment / momentum
beta. This module computes the Fama-French residual instead:

    alpha = R_i(window) - rf_cum(window) - Σ_k beta_k · factor_k_cum(window)

Leak-safety: betas are estimated ONLY on the 252 trading days ending at/before
the anchor (no forward information enters the coefficients). The forward factor
realizations are part of the LABEL (the thing we predict), computed after the
fact for training — that is by construction, not leakage. Features elsewhere stay
strictly as-of.

Pure, network-free (prices read cache-only; factors passed in). Unit-tested
offline. Returns NaN when history is insufficient so callers fall back to the SPY
label cleanly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from super_investor.adapters.ken_french import FACTOR_COLS

_MIN_BETA_OBS = 120  # need at least ~6 months of overlap to trust betas


def _daily_returns(ticker: str, prices_cache: Path, cache_only: bool = True) -> pd.Series:
    """Daily simple returns indexed by date, from the cached close series."""
    from super_investor.adapters.prices import get_history
    hist = get_history(ticker, prices_cache, cache_only=cache_only)
    if hist.empty:
        return pd.Series(dtype=float)
    s = hist.set_index(pd.to_datetime(hist["date"]))["close"].sort_index()
    return s.pct_change().dropna()


def estimate_betas(tic_ret: pd.Series, factors: pd.DataFrame,
                   anchor: pd.Timestamp, window: int = 252) -> dict | None:
    """OLS betas of (R_i - rf) on the 6 factors over the `window` trading days
    ending at/before `anchor`. Pre-anchor data only — no look-ahead."""
    cols = [c for c in FACTOR_COLS if c in factors.columns]
    f = factors[factors["date"] <= anchor].tail(window)
    if len(f) < _MIN_BETA_OBS or not cols:
        return None
    j = f.set_index("date")
    y = tic_ret.reindex(j.index).to_numpy()
    X = j[cols].to_numpy()
    rf = j["rf"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(rf)
    if mask.sum() < _MIN_BETA_OBS:
        return None
    yv = (y[mask] - rf[mask])
    Xv = np.column_stack([np.ones(mask.sum()), X[mask]])
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    return {c: float(b) for c, b in zip(cols, coef[1:])}


def _cum(series: pd.Series) -> float:
    """Compound a daily-return series to a single window return."""
    arr = series.to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.prod(1.0 + arr) - 1.0)


def factor_excess(ticker: str, anchor: pd.Timestamp, hold_months: float,
                  prices_cache: Path, factors: pd.DataFrame,
                  cache_only: bool = True) -> float:
    """Fama-French residual return of `ticker` over (anchor, anchor+hold].

    Returns NaN if betas can't be estimated or the forward window is empty.
    """
    if factors is None or factors.empty:
        return float("nan")
    tic_ret = _daily_returns(ticker, prices_cache, cache_only=cache_only)
    if tic_ret.empty:
        return float("nan")
    betas = estimate_betas(tic_ret, factors, anchor)
    if betas is None:
        return float("nan")

    end = anchor + pd.Timedelta(days=int(round(hold_months * 30.44)))
    fwd = factors[(factors["date"] > anchor) & (factors["date"] <= end)]
    fwd_ret = tic_ret[(tic_ret.index > anchor) & (tic_ret.index <= end)]
    if fwd.empty or fwd_ret.empty:
        return float("nan")

    r_i = _cum(fwd_ret)
    rf_cum = _cum(fwd["rf"])
    expected_excess = sum(b * _cum(fwd[c]) for c, b in betas.items() if c in fwd.columns)
    if not np.isfinite(r_i):
        return float("nan")
    return float(r_i - rf_cum - expected_excess)
