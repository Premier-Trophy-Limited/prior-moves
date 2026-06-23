"""Tests for the leakage self-audit harness (scripts/leakage_audit.py).

These pin the harness behavior both founders rely on:

  * a dataset whose label is copied into a feature is flagged LEAK and the CLI
    exits nonzero;
  * a clean dataset (noise features, plus a learnable-but-honest signal) does
    not raise a false LEAK;
  * the flat-score-across-capacity detector flags an all-equal capacity map and
    passes a monotone-improving one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import leakage_audit as L  # noqa: E402


# --------------------------------------------------------------------------
# fixtures: small synthetic frames
# --------------------------------------------------------------------------

def _leaked_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Label copied (with tiny jitter) into a feature — a textbook leak."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="QE")
    y = rng.integers(0, 2, size=n).astype(float)
    return pd.DataFrame({
        "date": dates,
        "noise": rng.normal(size=n),
        "leaked_copy": y + rng.normal(scale=0.01, size=n),
        "y": y,
    })


def _clean_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Stationary label with one honest learnable signal and one noise feature.

    The signal is genuinely predictive in-sample and out-of-time, but not
    perfectly — exactly the shape an honest pipeline produces. No feature is a
    function of the contemporaneous label.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="QE")
    f_sig = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-1.2 * f_sig))
    y = (rng.uniform(size=n) < p).astype(float)
    return pd.DataFrame({
        "date": dates,
        "f_sig": f_sig,
        "f_noise": rng.normal(size=n),
        "y": y,
    })


# --------------------------------------------------------------------------
# (a) leaked dataset => LEAK + nonzero exit
# --------------------------------------------------------------------------

def test_leaked_feature_is_flagged():
    df = _leaked_frame()
    results = L.run_audit(df, "date", "y", ["noise", "leaked_copy"],
                          n_shuffle=30, n_splits=4)
    assert L.overall_verdict(results) == L.LEAK
    # the implausible-score check is the one that catches an in-feature copy
    by_name = {r.name: r for r in results}
    assert by_name["implausible-score"].verdict == L.LEAK


def test_leaked_dataset_exits_nonzero(tmp_path):
    df = _leaked_frame()
    path = tmp_path / "leaked.parquet"
    df.to_parquet(path)
    code = L.main([str(path), "--date-col", "date", "--label-col", "y",
                   "--feature-cols", "noise,leaked_copy",
                   "--n-shuffle", "20", "--n-splits", "4"])
    assert code == 1


def test_self_test_flag_reports_leak_and_exits_nonzero():
    assert L.main(["--self-test", "--n-shuffle", "20"]) == 1


# --------------------------------------------------------------------------
# (b) clean dataset => no false LEAK
# --------------------------------------------------------------------------

def test_clean_dataset_does_not_false_leak():
    df = _clean_frame()
    results = L.run_audit(df, "date", "y", ["f_sig", "f_noise"],
                          n_shuffle=40, n_splits=4)
    assert L.overall_verdict(results) != L.LEAK
    for r in results:
        assert r.verdict != L.LEAK, f"{r.name} false-flagged LEAK: {r.why}"


def test_clean_dataset_exits_zero(tmp_path):
    df = _clean_frame()
    path = tmp_path / "clean.parquet"
    df.to_parquet(path)
    code = L.main([str(path), "--date-col", "date", "--label-col", "y",
                   "--feature-cols", "f_sig,f_noise",
                   "--n-shuffle", "30", "--n-splits", "4"])
    assert code == 0


# --------------------------------------------------------------------------
# (c) flat-score-across-capacity detector
# --------------------------------------------------------------------------

def test_flat_capacity_map_is_flagged():
    # tiny vs huge model both ~0.705 across 677x params => LEAK
    flat = {300_000: 0.705, 203_000_000: 0.706}
    r = L.check_flat_score_capacity(flat)
    assert r.verdict == L.LEAK


def test_monotone_capacity_map_passes():
    # score climbs with capacity over orders of magnitude => honest, PASS
    rising = {300_000: 0.55, 5_000_000: 0.62, 203_000_000: 0.71}
    r = L.check_flat_score_capacity(rising)
    assert r.verdict == L.PASS


def test_narrow_capacity_span_warns_not_leaks():
    # capacity barely moves => cannot conclude leak, WARN
    narrow = {1_000_000: 0.70, 1_500_000: 0.70}
    r = L.check_flat_score_capacity(narrow)
    assert r.verdict == L.WARN


def test_no_capacity_map_is_skipped_pass():
    r = L.check_flat_score_capacity(None)
    assert r.verdict == L.PASS


# --------------------------------------------------------------------------
# (d) feature-timestamp + embedding-cutoff checks
# --------------------------------------------------------------------------

def test_feature_timestamp_after_label_is_leak():
    label_dates = np.array(["2020-03-31", "2020-06-30"], dtype="datetime64[ns]")
    avail = {"good": "2019-12-31", "bad": "2020-03-31"}  # bad == earliest label
    r = L.check_feature_timestamps(["good", "bad"], avail, label_dates)
    assert r.verdict == L.LEAK
    assert "bad" in r.why


def test_feature_timestamp_all_before_label_passes():
    label_dates = np.array(["2020-03-31", "2020-06-30"], dtype="datetime64[ns]")
    avail = {"good": "2019-12-31"}
    r = L.check_feature_timestamps(["good"], avail, label_dates)
    assert r.verdict == L.PASS


def test_embedding_cutoff_warns_on_pre_cutoff_labels():
    label_dates = np.array(["2019-01-31", "2025-01-31"], dtype="datetime64[ns]")
    r = L.check_embedding_cutoff(label_dates, "2024-01-01")
    assert r.verdict == L.WARN
    assert "1/2" in r.why


def test_embedding_cutoff_passes_when_all_post_cutoff():
    label_dates = np.array(["2025-02-28", "2025-05-31"], dtype="datetime64[ns]")
    r = L.check_embedding_cutoff(label_dates, "2024-01-01")
    assert r.verdict == L.PASS


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------

def test_auc_ranks_perfect_separation():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(L._auc(y, score) - 1.0) < 1e-9


def test_auc_chance_on_constant_score():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    score = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(L._auc(y, score) - 0.5) < 1e-9


def test_is_binary_detection():
    assert L._is_binary(np.array([0.0, 1.0, 1.0, 0.0]))
    assert not L._is_binary(np.array([0.1, 0.2, 0.3, 0.4]))


def test_auto_cols_detects_roles():
    df = _clean_frame(n=20)
    date_col, label_col, feats = L._auto_cols(df, None, None, None)
    assert date_col == "date"
    assert label_col == "y"
    assert set(feats) == {"f_sig", "f_noise"}


# --------------------------------------------------------------------------
# numpy fallback parity (forced off-sklearn path)
# --------------------------------------------------------------------------

def test_numpy_fallback_runs_when_sklearn_disabled(monkeypatch):
    monkeypatch.setattr(L, "_has_sklearn", lambda: False)
    df = _leaked_frame()
    results = L.run_audit(df, "date", "y", ["noise", "leaked_copy"],
                          n_shuffle=20, n_splits=4)
    # the in-feature copy is still caught without sklearn
    assert L.overall_verdict(results) == L.LEAK
