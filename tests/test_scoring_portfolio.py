"""Unit tests for the scoring composite, portfolio constructor, and labels.

These cover the validated-model math surfaces that previously had zero tests:
percentile ranking (NaN neutrality), the impact haircut, weight validation,
water-fill position capping, sector capping, equal-weight construction, and
the quarter-over-quarter label diff (new_entry / add / trim / exit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from super_investor.scoring import (
    ScoreWeights,
    _pct_rank,
    compute_prior_scores,
    is_crypto_etf,
    is_etf,
    score_label,
)
from super_investor.portfolio import (
    RiskProfile,
    _apply_sector_cap,
    _waterfill_cap,
    build_portfolio,
)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_pct_rank_nan_is_neutral():
    s = pd.Series([1.0, 2.0, np.nan, 3.0])
    r = _pct_rank(s)
    assert r.iloc[2] == 0.5            # missing data sits at the median
    assert r.iloc[0] < r.iloc[1] < r.iloc[3]
    assert ((r >= 0) & (r <= 1)).all()


def test_pct_rank_all_nan_returns_half():
    r = _pct_rank(pd.Series([np.nan, np.nan]))
    assert (r == 0.5).all()


def test_score_label_thresholds():
    assert score_label(75) == "★★★★★"
    assert score_label(74.9) == "★★★★☆"
    assert score_label(60) == "★★★★☆"
    assert score_label(45) == "★★★☆☆"
    assert score_label(30) == "★★☆☆☆"
    assert score_label(0) == "★☆☆☆☆"


def test_weights_validate():
    ScoreWeights().validate()  # defaults must sum to 1.0
    with pytest.raises(AssertionError):
        ScoreWeights(conviction=0.9).validate()


def test_etf_lists():
    assert is_etf("SPY") and is_etf("spy ")
    assert not is_etf("NVDA")
    assert is_crypto_etf("IBIT") and is_etf("IBIT")  # crypto ETFs stay in both


def _candidates(n=6) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "max_p": rng.uniform(0.05, 0.6, n),
        "mean_p": rng.uniform(0.02, 0.3, n),
        "consensus_raw": rng.uniform(0, 5, n),
        "momentum_1m": rng.normal(0, 0.1, n),
        "momentum_3m": rng.normal(0, 0.2, n),
        "cross_signal_count": rng.integers(0, 4, n),
        "dollar_vol": [1e9, 2e8, 6e7, 1e7, 2e6, np.nan],
    })


def test_compute_prior_scores_bounds_and_order():
    out = compute_prior_scores(_candidates())
    assert (out["prior_score"] >= 0).all() and (out["prior_score"] <= 100).all()
    # sorted descending
    assert out["prior_score"].is_monotonic_decreasing
    # haircut bounded [0.70, 1.0]; NaN dollar_vol -> 0.92
    assert out["impact_haircut"].between(0.70, 1.0).all()
    nan_row = out[out["ticker"] == "T5"].iloc[0]
    assert nan_row["impact_haircut"] == pytest.approx(0.92)
    assert nan_row["liquidity_tier"] == "unknown"
    # tier mapping
    tiers = dict(zip(out["ticker"], out["liquidity_tier"]))
    assert tiers["T0"] == "mega" and tiers["T1"] == "mega"   # >= $200M ADV
    assert tiers["T2"] == "large" and tiers["T3"] == "mid" and tiers["T4"] == "thin"


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


def test_waterfill_cap_holds_and_preserves_total():
    w = pd.Series([0.6, 0.3, 0.1])
    capped = _waterfill_cap(w, cap=0.4)
    assert capped.max() <= 0.4 + 1e-9
    assert capped.sum() == pytest.approx(1.0)


def test_waterfill_cap_infeasible_cap_breaks_cleanly():
    # 3 names, cap 0.2 -> max feasible total 0.6; must not loop forever
    w = pd.Series([0.5, 0.3, 0.2])
    capped = _waterfill_cap(w, cap=0.2)
    assert capped.max() <= 0.2 + 1e-6


def test_sector_cap_scales_down():
    df = pd.DataFrame({
        "sector": ["Tech", "Tech", "Energy"],
        "weight": [0.4, 0.4, 0.2],
    })
    out = _apply_sector_cap(df, cap=0.5)
    assert out[out["sector"] == "Tech"]["weight"].sum() == pytest.approx(0.5)
    assert out[out["sector"] == "Energy"]["weight"].sum() == pytest.approx(0.2)


def _agg_for_portfolio() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "name": ["A", "B", "C", "D"],
        "sector": ["Tech", "Tech", "Energy", "Health"],
        "prior_score": [90.0, 80.0, 70.0, 60.0],
        "liquidity_tier": ["mega", "large", "large", "mid"],
        "last_close": [100.0, 50.0, 25.0, 10.0],
    })


def test_build_portfolio_equal_weight_unlevered():
    profile = RiskProfile(capital=100_000, n_positions=4, max_position=1.0,
                          max_sector=1.0, cash_buffer=0.0, min_tier="mid",
                          equal_weight=True, label="test")
    res = build_portfolio(_agg_for_portfolio(), profile)
    h = res["holdings"]
    assert len(h) == 4
    deployed = res["summary"]["deployed_$"]
    assert deployed <= 100_000 + 1e-6          # never over-deploys capital
    assert res["summary"]["gross_exposure_pct"] <= 1.0 + 1e-9  # unlevered


def test_build_portfolio_min_tier_gate():
    profile = RiskProfile(capital=50_000, n_positions=4, min_tier="large",
                          label="test")
    res = build_portfolio(_agg_for_portfolio(), profile)
    assert "DDD" not in set(res["holdings"]["ticker"])  # mid name gated out


def test_build_portfolio_min_price_gate():
    profile = RiskProfile(capital=50_000, n_positions=4, min_tier="mid",
                          min_price=20.0, label="test")
    res = build_portfolio(_agg_for_portfolio(), profile)
    assert set(res["holdings"]["ticker"]) <= {"AAA", "BBB", "CCC"}


# ---------------------------------------------------------------------------
# labels — quarter-over-quarter diff
# ---------------------------------------------------------------------------


def test_build_labels_diff(tmp_path):
    from super_investor.labels import build_labels

    rows = []
    # Q1: X=100 sh, Y=100 sh        Q2: X=150 (add), Z=50 (new), Y absent (exit)
    for cusip, shares, period, acc in [
        ("X", 100, "2024-03-31", "a1"), ("Y", 100, "2024-03-31", "a1"),
        ("W", 100, "2024-03-31", "a1"),
        ("X", 150, "2024-06-30", "a2"), ("Z", 50, "2024-06-30", "a2"),
        ("W", 100, "2024-06-30", "a2"),
    ]:
        rows.append({
            "investor_slug": "tst", "cusip": cusip, "shares": shares,
            "value_usd": shares * 10, "name_of_issuer": cusip,
            "period_of_report": pd.Timestamp(period),
            "filed_at": pd.Timestamp(period) + pd.Timedelta(days=40),
            "accession": acc,
        })
    p = tmp_path / "tst.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)

    out = build_labels(p)
    q2 = out[out["period_of_report"] == pd.Timestamp("2024-06-30")]
    lab = dict(zip(q2["cusip"], q2["label"]))
    assert lab["X"] == "add"          # +50%
    assert lab["Z"] == "new_entry"
    assert lab["Y"] == "exit"         # synthesized exit row
    assert lab["W"] == "hold"         # unchanged
    y_row = q2[q2["cusip"] == "Y"].iloc[0]
    assert y_row["shares"] == 0 and y_row["shares_delta_pct"] == -100.0
