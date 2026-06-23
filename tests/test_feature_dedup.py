"""Regression: per-investor feature columns must be unique.

`avg_hourly_earnings` is a FRED macro column (kept via base_cols) that also
matches the `avg_` prefix group, so it used to land in feature_cols twice and
crash LightGBM ("Feature (avg_hourly_earnings) appears more than one time").
Surfaced by scripts/optimize_2h.py phase 2. These tests pin the fix.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_DATA = REPO / "data" / "features"


def test_train_one_rejects_duplicate_feature_names():
    """The guard in train_one fails loud on a duplicate, not with a cryptic
    LightGBM error. Pure-logic — no data files needed."""
    import pandas as pd

    from super_investor.models.per_investor import train_one

    # Build rows that clear the data gate (train n>=50 with >=5 positives, and a
    # val window [2016-09-30, 2017-09-30) with >=1 positive) so execution reaches
    # the duplicate-feature guard rather than the insufficient-data early return.
    train_dates = pd.to_datetime(["2015-01-01"] * 60)   # < val_start -> train
    val_dates = pd.to_datetime(["2017-01-01"] * 6)      # in val window
    df = pd.DataFrame({
        "period_of_report": list(train_dates) + list(val_dates),
        "y": ([1] * 6 + [0] * 54) + ([1] * 1 + [0] * 5),
        "cusip_id": list(range(66)),
        "feat_a": [0.1] * 66,
        "dup": [1.0] * 66,
    })
    feature_cols = ["feat_a", "dup", "dup"]  # deliberate duplicate
    with pytest.raises(ValueError, match="duplicate feature names"):
        train_one("test", df, feature_cols, cat_cols=[])


@pytest.mark.skipif(not (_DATA / "macro_quarterly.parquet").exists(),
                    reason="feature parquets not present (data-dependent)")
def test_attach_features_unique_columns():
    """End-to-end: the real assembled feature set has no duplicate columns, and
    avg_hourly_earnings appears exactly once."""
    from super_investor.models.per_investor import _attach_features, _load_inputs

    ins = _load_inputs()
    _df, feature_cols, cat_cols = _attach_features(ins["labels"], ins)
    assert len(feature_cols) == len(set(feature_cols)), \
        f"duplicate feature cols: {sorted({c for c in feature_cols if feature_cols.count(c) > 1})}"
    assert feature_cols.count("avg_hourly_earnings") <= 1
    assert all(c in feature_cols for c in cat_cols)
