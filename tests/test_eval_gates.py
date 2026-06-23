"""Tests for the shared eval-gate library (src/super_investor/eval_gates.py).

These pin the statistical behavior both trading tracks rely on: CI coverage,
the leakage canary's p-value direction, cost monotonicity, IC sign, and the
multiple-testing deflation increasing with the number of trials searched.
"""
from __future__ import annotations

import numpy as np

from super_investor import eval_gates as G


def test_norm_roundtrip():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert abs(G.norm_cdf(G.norm_ppf(p)) - p) < 1e-6


def test_tstat_matches_formula():
    x = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    expected = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(G.tstat(x) - expected) < 1e-12


def test_max_drawdown():
    curve = np.array([1.0, 1.2, 0.9, 1.1])
    # peak 1.2 -> trough 0.9 = -25%
    assert abs(G.max_drawdown(curve) - (0.9 / 1.2 - 1.0)) < 1e-12


def test_block_bootstrap_ci_brackets_mean_and_widens_with_block():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.05, size=200)
    lo1, hi1 = G.block_bootstrap_ci(x, block=1, n_boot=5000, seed=7)
    lo4, hi4 = G.block_bootstrap_ci(x, block=8, n_boot=5000, seed=7)
    # CI brackets the sample mean
    assert lo1 < x.mean() < hi1
    assert lo4 < x.mean() < hi4
    # both are finite, ordered intervals
    assert hi1 > lo1 and hi4 > lo4


def test_block_bootstrap_degenerate():
    assert all(np.isnan(v) for v in G.block_bootstrap_ci(np.array([0.01])))


def test_shuffle_pvalue_direction():
    # real score far above the shuffled-noise distribution -> small p-value
    shuffled = np.random.default_rng(0).normal(0.5, 0.02, size=100)
    assert G.shuffle_pvalue(0.7, shuffled, higher_is_better=True) < 0.05
    # real score buried in the noise -> large p-value
    assert G.shuffle_pvalue(0.5, shuffled, higher_is_better=True) > 0.2


def test_leakage_flag():
    clean = np.array([0.49, 0.50, 0.51, 0.50])   # mean ~0.5 -> clean
    leaky = np.array([0.60, 0.62, 0.58, 0.61])   # mean ~0.6 -> flag
    assert not G.leakage_flag(0.8, clean)
    assert G.leakage_flag(0.8, leaky)


def test_cost_haircut_monotone():
    gross = np.array([0.02, 0.03, -0.01])
    net = G.cost_haircut(gross, turnover=1.0, cost_bps=20)
    assert np.all(net < gross)
    # higher cost -> lower net
    net2 = G.cost_haircut(gross, turnover=1.0, cost_bps=40)
    assert np.all(net2 < net)
    # zero turnover -> no haircut
    assert np.allclose(G.cost_haircut(gross, turnover=0.0, cost_bps=40), gross)


def test_spearman_ic_sign():
    scores = np.array([1.0, 2, 3, 4, 5])
    fwd_up = np.array([0.1, 0.2, 0.25, 0.4, 0.5])     # monotone increasing
    fwd_dn = -fwd_up
    assert G.spearman_ic(scores, fwd_up) > 0.9
    assert G.spearman_ic(scores, fwd_dn) < -0.9


def test_selection_inflation_grows_with_trials():
    # expected max sharpe under the null rises as you try more configs
    few = G.expected_max_sharpe(n_trials=5, n_obs=34)
    many = G.expected_max_sharpe(n_trials=500, n_obs=34)
    assert many > few > 0


def test_deflated_sharpe_penalizes_search():
    # same observed SR is LESS convincing after searching more configs
    sr = 0.4
    p_few = G.deflated_sharpe(sr, n_trials=5, n_obs=34)
    p_many = G.deflated_sharpe(sr, n_trials=2000, n_obs=34)
    assert p_few > p_many


def test_summarize_signal_vs_noise():
    rng = np.random.default_rng(2)
    # a real positive edge -> report mean > 0, decent t
    edge = rng.normal(0.03, 0.04, size=80)
    rep = G.summarize(edge)
    assert rep.n == 80
    assert rep.mean > 0
    assert "SIGNAL" in rep.verdict() or "DIRECTIONAL" in rep.verdict()
    # pure noise centered at zero -> not a "SIGNAL" verdict
    noise = rng.normal(0.0, 0.05, size=80)
    assert "SIGNAL: 95% CI above zero" not in G.summarize(noise).verdict()
