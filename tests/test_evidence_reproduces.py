"""Guard: the handoff numbers cannot silently drift.

This asserts that the values reproduced from the frozen backtest artifacts
(via scripts/reproduce_evidence.py) still match the published claims canon in
web/data/consensus.json within tolerance. If a future edit re-marks a quarter,
re-runs the backtest, or edits the canon, and the two disagree, this test fails
loud — so the senior co-founder auditing the handoff always sees consistent
numbers across the site and the evidence pack.

These tests skip (not fail) if the frozen run artifacts are absent, so a fresh
checkout without the heavy run outputs does not red the suite.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_evidence.py"
CONSENSUS = REPO / "web" / "data" / "consensus.json"
BACKTEST = REPO / "runs" / "per_investor_wf" / "priorscore_backtest.parquet"
FACTOR = REPO / "runs" / "factor_attribution_2026-06-17.json"


def _load_reproduce_module():
    """Import scripts/reproduce_evidence.py as a module (it is not a package)."""
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("reproduce_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # register before exec so the @dataclass in the module can resolve its own
    # __module__ via sys.modules (dataclasses looks it up there at class build).
    sys.modules["reproduce_evidence"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reproduced() -> dict:
    if not BACKTEST.exists() or not FACTOR.exists():
        pytest.skip(
            "frozen run artifacts absent; regenerate with "
            "scripts/backtest_priorscore.py and scripts/factor_attribution.py"
        )
    module = _load_reproduce_module()
    _, values = module.build_rows()
    return values


@pytest.fixture(scope="module")
def canon() -> dict:
    return json.loads(CONSENSUS.read_text())


def test_mean_edge_matches_canon(reproduced: dict, canon: dict) -> None:
    """Reproduced net edge equals the published edge within 0.05 pts/q."""
    repro_pts = reproduced["mean_edge_pts_q"]
    canon_pts = float(canon["edge_pts_q"])
    assert abs(repro_pts - canon_pts) <= 0.05, (
        f"net edge drifted: reproduced {repro_pts:+.3f} pts/q vs "
        f"canon {canon_pts:+.3f} pts/q"
    )


def test_tstat_matches_canon(reproduced: dict, canon: dict) -> None:
    """Reproduced t-stat equals the published t within 0.05."""
    repro_t = reproduced["t_stat"]
    canon_t = float(canon["t_stat"])
    assert abs(repro_t - canon_t) <= 0.05, (
        f"t-stat drifted: reproduced {repro_t:.3f} vs canon {canon_t:.3f}"
    )


def test_n_quarters_matches_canon(reproduced: dict, canon: dict) -> None:
    """Sample size matches the published n exactly."""
    assert reproduced["n_quarters"] == int(canon["n_quarters"])


def test_quarters_positive_is_22_of_34(reproduced: dict) -> None:
    """The quarters-beat-SPY count is the published 22 of 34."""
    assert reproduced["quarters_positive"] == 22
    assert reproduced["n_quarters"] == 34


def test_factor_neutral_alpha_within_tolerance(reproduced: dict) -> None:
    """Factor-neutral selection alpha matches the handoff +0.75 pts/q (t=0.68)."""
    assert abs(reproduced["alpha_pts_q"] - 0.75) <= 0.05, (
        f"factor-neutral alpha drifted: {reproduced['alpha_pts_q']:+.3f} pts/q"
    )
    assert abs(reproduced["alpha_t"] - 0.68) <= 0.05


def test_hml_beta_is_the_significant_tilt(reproduced: dict) -> None:
    """The published value tilt: HML beta around -0.43 with a significant t."""
    assert abs(reproduced["hml_beta"] - (-0.43)) <= 0.05
    assert reproduced["hml_t"] <= -3.0


def test_ci_includes_zero(reproduced: dict) -> None:
    """The 95% block-bootstrap CI brackets zero, matching the honesty claim."""
    lo, hi = reproduced["ci95_pts_q"]
    assert lo < 0.0 < hi, f"CI [{lo:+.3f}, {hi:+.3f}] no longer includes zero"


def test_weight_search_n_trials(reproduced: dict) -> None:
    """The multiple-testing N (weight-search configs) is the published 200."""
    assert reproduced["n_trials"] == 200
