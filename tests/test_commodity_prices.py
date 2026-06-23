"""D4 — commodity merge logic, network-free (monkeypatch the fetch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_merge_commodities_noop_when_empty(monkeypatch):
    import fetch_macro
    monkeypatch.setattr(fetch_macro, "_commodity_frame",
                        lambda *a, **k: pd.DataFrame())
    macro = pd.DataFrame({"quarter_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                          "vix": [13.0, 14.0]})
    out = fetch_macro._merge_commodities(macro, "QE", "quarter_end", "2014-01-01")
    assert list(out.columns) == ["quarter_end", "vix"]
    assert len(out) == 2


def test_merge_commodities_adds_columns(monkeypatch):
    import fetch_macro
    comm = pd.DataFrame({
        "quarter_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "gold": [2200.0, 2330.0], "copper": [4.0, 4.5],
    })
    monkeypatch.setattr(fetch_macro, "_commodity_frame", lambda *a, **k: comm)
    macro = pd.DataFrame({"quarter_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                          "vix": [13.0, 14.0]})
    out = fetch_macro._merge_commodities(macro, "QE", "quarter_end", "2014-01-01")
    assert "gold" in out.columns and "copper" in out.columns
    assert out.loc[out["quarter_end"] == "2024-06-30", "gold"].iloc[0] == 2330.0
