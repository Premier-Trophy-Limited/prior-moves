"""Corporate-event corpus — unit tests (no network, no SEC, no prices).

Covers the deterministic magnitude mapping and the leak-safe label window
(embargo pushes the return window strictly AFTER the public filing date, and the
close-to-close return is causal — uses only closes at/before each anchor).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def test_magnitude_from_item_severity():
    import build_corporate_events as B
    assert B._magnitude_for_items(["4.02"]) == 5          # restatement = top
    assert B._magnitude_for_items(["2.02"]) == 2          # earnings = routine
    assert B._magnitude_for_items(["2.02", "4.02"]) == 5  # max wins
    assert B._magnitude_for_items(["9.99"]) == 1          # unknown floors to 1
    assert B._magnitude_for_items([]) == 1


def test_label_window_starts_after_embargo():
    import train_corporate_model as T
    filing = pd.Timestamp("2020-03-02")
    anchor = filing + pd.tseries.offsets.BDay(T.EMBARGO_BDAYS)
    # strictly after the public filing — no announcement-day bounce in the label
    assert anchor > filing
    assert (anchor - filing).days >= T.EMBARGO_BDAYS


def test_ret_is_causal_uses_only_past_closes():
    import train_corporate_model as T
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(range(100, 110), index=idx, dtype=float)
    # window [day2, day5] → close at/before each anchor: 102 -> 105
    r = T._ret(s, idx[2], idx[5])
    assert abs(r - (105.0 / 102.0 - 1.0)) < 1e-12
    # a future close after `end` must never enter the return
    r2 = T._ret(s, idx[2], idx[5] + pd.Timedelta(hours=1))
    assert abs(r2 - r) < 1e-12


def test_ret_empty_on_no_coverage():
    import math

    import train_corporate_model as T
    r = T._ret(pd.Series(dtype=float), pd.Timestamp("2020-01-01"),
               pd.Timestamp("2020-02-01"))
    assert math.isnan(r)
