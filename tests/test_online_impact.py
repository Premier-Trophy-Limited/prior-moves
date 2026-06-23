"""Online / continual-learning impact model — unit tests (no network, no prices).

Covers the leak-free streaming contract (predict-then-update, never peek
forward), the RLS recursion learning a known linear map, the surprise gate
monotonicity, and persistence round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from super_investor.online_impact import (  # noqa: E402
    MultiTimescaleOnline, OnlineRidge, OnlineSurprise)


def test_online_ridge_learns_linear_map():
    rng = np.random.default_rng(0)
    true_w = np.array([1.5, -2.0, 0.5])
    m = OnlineRidge(dim=3, lam=1.0, ridge=1.0)
    for _ in range(500):
        x = rng.normal(size=3)
        y = float(x @ true_w) + rng.normal(0, 0.01)
        m.update_one(x, y)
    assert np.allclose(m.w, true_w, atol=0.1)
    assert m.n_seen == 500


def test_predict_then_update_is_causal():
    # The prediction for step k must use ONLY weights from steps < k.
    rng = np.random.default_rng(1)
    m = OnlineRidge(dim=2, lam=0.99, ridge=10.0)
    seen_preds = []
    for _ in range(50):
        x = rng.normal(size=2)
        p_before = m.predict_one(x)        # uses w trained on prior points only
        seen_preds.append(p_before)
        m.update_one(x, float(x @ np.array([1.0, 1.0])))
    # first prediction is from the zero-seed (no future info leaked in)
    assert seen_preds[0] == 0.0


def test_surprise_gate_monotone_and_bounded():
    m = MultiTimescaleOnline(dim=2)
    gates = [m._gate(s) for s in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert all(0.0 <= g <= 1.0 for g in gates)
    assert gates == sorted(gates)          # higher surprise → more weight on fast
    assert gates[0] < gates[-1]


def test_multitimescale_blend_between_stores():
    rng = np.random.default_rng(2)
    m = MultiTimescaleOnline(dim=2, lam_slow=0.999, lam_fast=0.9)
    for _ in range(200):
        x = rng.normal(size=2)
        m.update_one(x, float(x @ np.array([0.5, -0.5])))
    x = np.array([1.0, 1.0])
    blended = m.predict_one(x, surprise=1.0)
    slow, fast = m.slow.predict_one(x), m.fast.predict_one(x)
    lo, hi = sorted((slow, fast))
    assert lo - 1e-9 <= blended <= hi + 1e-9


def test_online_surprise_zscore():
    s = OnlineSurprise()
    for v in (1.0, 1.0, 1.0, 1.0, 1.0):
        s.update(v)
    # constant stream → zero std → zero surprise
    assert s.update(1.0) == 0.0
    s2 = OnlineSurprise()
    for v in (0.0, 1.0, 2.0, 3.0, 4.0):
        s2.update(v)
    z = s2.current(10.0)                    # far outlier → large z
    assert z > 2.0


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    m = MultiTimescaleOnline(dim=3)
    for _ in range(30):
        x = rng.normal(size=3)
        m.update_one(x, float(x.sum()))
    p = tmp_path / "online.json"
    m.save(p)
    m2 = MultiTimescaleOnline.from_dict(__import__("json").loads(p.read_text()))
    x = rng.normal(size=3)
    assert abs(m.predict_one(x) - m2.predict_one(x)) < 1e-9
