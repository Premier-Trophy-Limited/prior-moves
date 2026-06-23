"""D5 impact-model unit tests — no network, no prices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from super_investor import impact_model as IM  # noqa: E402


def test_feature_row_shape():
    r = IM.feature_row("macro", 4, True, 80.0, {"vix": 22.0, "hy_oas": 4.5})
    for f in IM.FEATURE_NAMES:
        assert f in r
    assert r["magnitude"] == 4.0
    assert r["in_impact_sector"] == 1.0
    assert r["type_macro"] == 1.0 and r["type_geopolitical"] == 0.0
    assert abs(r["prior_score"] - 0.8) < 1e-9          # scaled /100
    assert r["macro_vix"] == 22.0


def test_feature_row_missing_macro_defaults_zero():
    r = IM.feature_row("thematic", None, False, None, None)
    assert r["macro_vix"] == 0.0
    assert r["prior_score"] == 0.0
    assert r["magnitude"] == 0.0


def _synthetic():
    # 3 events, label correlated with in_impact_sector → model should learn it.
    rows, y, groups = [], [], []
    rng = np.random.default_rng(0)
    for ev in ("e1", "e2", "e3"):
        for _ in range(20):
            in_sec = bool(rng.integers(0, 2))
            rows.append(IM.feature_row("macro", 3, in_sec, rng.uniform(40, 90),
                                       {"vix": 20.0}))
            y.append((0.05 if in_sec else -0.01) + rng.normal(0, 0.005))
            groups.append(ev)
    return rows, y, groups


def test_fit_and_weights_sum_to_one():
    rows, y, groups = _synthetic()
    m = IM.fit(rows, y, groups)
    assert m.n_events == 3
    assert m.n_rows == 60
    tickers = ["AAA", "BBB", "CCC"]
    frows = [IM.feature_row("macro", 3, True, 70, {"vix": 20.0}),
             IM.feature_row("macro", 3, True, 60, {"vix": 20.0}),
             IM.feature_row("macro", 3, False, 50, {"vix": 20.0})]
    w = m.weights(tickers, frows)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # in-sector names should carry more weight than the out-of-sector one
    assert w["AAA"] >= w["CCC"]


def test_weights_all_negative_falls_back_equal():
    # craft a model whose predictions are all negative → equal-weight fallback
    m = IM.ImpactModel(coef=[0.0] * len(IM.FEATURE_NAMES), intercept=-1.0,
                       feature_names=list(IM.FEATURE_NAMES), n_events=3,
                       n_rows=10, cv_r2=None, y_mean=-1.0)
    w = m.weights(["X", "Y"], [IM.feature_row("macro", 1, True, 50, {}),
                               IM.feature_row("macro", 1, True, 50, {})])
    assert abs(w["X"] - 0.5) < 1e-9 and abs(w["Y"] - 0.5) < 1e-9


def test_save_load_roundtrip(tmp_path):
    rows, y, groups = _synthetic()
    m = IM.fit(rows, y, groups)
    p = tmp_path / "model.json"
    m.save(p)
    m2 = IM.ImpactModel.load(p)
    assert m2 is not None
    assert m2.coef == m.coef and m2.intercept == m.intercept
    assert IM.ImpactModel.load(tmp_path / "nope.json") is None
