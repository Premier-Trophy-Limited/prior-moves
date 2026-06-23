"""Factor-adjusted label — unit tests (no network; factors + prices synthetic).

Covers leak-safe beta estimation (pre-anchor data only), the cumulative-return
helper, and the end-to-end factor_excess on a tmp price cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from super_investor import factor_label as FL  # noqa: E402


def _synthetic_factors(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    df = pd.DataFrame({
        "date": dates, "mkt_rf": mkt, "smb": rng.normal(0, 0.004, n),
        "hml": rng.normal(0, 0.004, n), "rmw": rng.normal(0, 0.003, n),
        "cma": rng.normal(0, 0.003, n), "mom": rng.normal(0, 0.005, n),
        "rf": np.full(n, 0.0001),
    })
    return df


def test_estimate_betas_recovers_known_beta():
    f = _synthetic_factors()
    true_beta = 1.3
    # construct a return series = rf + 1.3*mkt_rf + noise
    ret = pd.Series(f["rf"].to_numpy() + true_beta * f["mkt_rf"].to_numpy()
                    + np.random.default_rng(1).normal(0, 1e-4, len(f)),
                    index=f["date"])
    anchor = f["date"].iloc[400]
    betas = FL.estimate_betas(ret, f, anchor, window=252)
    assert betas is not None
    assert abs(betas["mkt_rf"] - true_beta) < 0.1


def test_estimate_betas_uses_only_pre_anchor():
    f = _synthetic_factors()
    ret = pd.Series(f["mkt_rf"].to_numpy(), index=f["date"])
    early = f["date"].iloc[80]    # < _MIN_BETA_OBS (120) history before this → None
    assert FL.estimate_betas(ret, f, early, window=252) is None


def test_cum_compounds():
    s = pd.Series([0.1, 0.1])
    assert abs(FL._cum(s) - (1.1 * 1.1 - 1.0)) < 1e-12
    assert np.isnan(FL._cum(pd.Series(dtype=float)))


def test_factor_excess_on_tmp_cache(tmp_path):
    from super_investor.adapters import prices as P  # noqa: F401
    f = _synthetic_factors()
    # price series whose returns ≈ rf + 1.0*mkt → factor residual ≈ 0
    rets = f["rf"].to_numpy() + f["mkt_rf"].to_numpy()
    close = 100.0 * np.cumprod(1.0 + rets)
    hist = pd.DataFrame({"date": f["date"], "close": close})
    hist.to_parquet(tmp_path / "AAA.parquet", index=False)
    anchor = f["date"].iloc[400]
    a = FL.factor_excess("AAA", anchor, hold_months=3, prices_cache=tmp_path,
                         factors=f, cache_only=True)
    assert np.isfinite(a)
    assert abs(a) < 0.05   # residual small, by construction


def test_factor_excess_nan_when_no_prices(tmp_path):
    f = _synthetic_factors()
    a = FL.factor_excess("ZZZ", f["date"].iloc[400], 3, tmp_path, f, cache_only=True)
    assert np.isnan(a)
