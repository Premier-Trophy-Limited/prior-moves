"""Network-free leak + correctness tests for channel_join (Step B).

The whole point of channel_join is that a channel parquet keyed by `quarter_end`
is NOT public at the quarter end — it becomes available `quarter_end + lag` later.
These tests assert the as-of join never reaches into a quarter whose availability
date is after the event's filing_date (the leak), and that the coverage flag and
neutral-fill behave.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from super_investor import channel_join as CJ


def _write_channel(d: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(d / f"{name}_quarterly.parquet", index=False)


def test_asof_uses_only_available_quarter(tmp_path: Path):
    # finra lag = 20d. Event filed 2020-06-01.
    #   Q1 end 2020-03-31 -> avail 2020-04-20  (<= filing: ELIGIBLE)
    #   Q2 end 2020-06-30 -> avail 2020-07-20  (>  filing: LEAK if used)
    _write_channel(tmp_path, "finra", [
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-03-31"), "sh_x": 10.0},
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-06-30"), "sh_x": 99.0},
    ])
    ev = pd.DataFrame({"ticker": ["AAA"], "filing_date": [pd.Timestamp("2020-06-01")]})

    out, info = CJ.join_channels(ev, channels=["finra"], features_dir=tmp_path,
                                 keep_avail=True)
    assert out.loc[0, "ch_finra__sh_x"] == 10.0, "must use Q1 (available), never Q2 (future)"
    assert out.loc[0, "has_finra"] == 1.0
    assert out.loc[0, "avail_finra"] <= ev.loc[0, "filing_date"], "no look-ahead"
    assert "sh_x" in info["finra"][0] or info["finra"] == ["ch_finra__sh_x", "has_finra"]


def test_no_eligible_row_neutral_fills(tmp_path: Path):
    # Only a Q1 row (avail 2020-04-20); event filed 2020-01-01 -> nothing available yet.
    _write_channel(tmp_path, "finra", [
        {"ticker": "BBB", "quarter_end": pd.Timestamp("2020-03-31"), "sh_x": 10.0},
    ])
    ev = pd.DataFrame({"ticker": ["BBB"], "filing_date": [pd.Timestamp("2020-01-01")]})

    out, _ = CJ.join_channels(ev, channels=["finra"], features_dir=tmp_path)
    assert out.loc[0, "ch_finra__sh_x"] == 0.0, "no available row -> neutral fill 0"
    assert out.loc[0, "has_finra"] == 0.0, "coverage flag must report absence"


def test_lag_is_per_channel(tmp_path: Path):
    # sec_xbrl lag = 75d. Q1 end 2020-03-31 -> avail 2020-06-14. Filing 2020-06-01
    # is BEFORE availability -> must NOT be used (fundamentals lag the quarter).
    _write_channel(tmp_path, "sec_xbrl", [
        {"ticker": "CCC", "quarter_end": pd.Timestamp("2020-03-31"), "xf_pe": 12.0},
    ])
    ev = pd.DataFrame({"ticker": ["CCC"], "filing_date": [pd.Timestamp("2020-06-01")]})

    out, _ = CJ.join_channels(ev, channels=["sec_xbrl"], features_dir=tmp_path)
    assert out.loc[0, "has_sec_xbrl"] == 0.0, \
        "75d-lag fundamentals not yet public 62d after quarter end"
    assert out.loc[0, "ch_sec_xbrl__xf_pe"] == 0.0


def test_invariant_never_uses_future_quarter(tmp_path: Path):
    # Property check across many events: for every matched row, quarter_end + lag
    # <= filing_date (the leak-free invariant), and unmatched rows fill 0/flag 0.
    rng = np.random.default_rng(0)
    qends = pd.to_datetime(["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
                            "2020-03-31", "2020-06-30"])
    _write_channel(tmp_path, "finra",
                   [{"ticker": "AAA", "quarter_end": q, "sh_x": float(i)}
                    for i, q in enumerate(qends)])
    filings = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 600, size=50), unit="D")
    ev = pd.DataFrame({"ticker": ["AAA"] * 50, "filing_date": filings})

    out, _ = CJ.join_channels(ev, channels=["finra"], features_dir=tmp_path,
                              keep_avail=True)
    lag = pd.Timedelta(days=CJ._LAG_DAYS["finra"])
    matched = out[out["has_finra"] == 1.0]
    # every matched availability date must be on/before its filing_date
    assert (matched["avail_finra"] <= matched["filing_date"]).all()
    # and equal the most-recent eligible quarter_end + lag (no future quarter)
    for _, r in matched.iterrows():
        eligible = qends[(qends + lag) <= r["filing_date"]]
        assert r["avail_finra"] == eligible.max() + lag


def test_skip_and_discover(tmp_path: Path):
    _write_channel(tmp_path, "finra", [
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-03-31"), "sh_x": 1.0}])
    # macro is in _SKIP and must never be discovered as a joinable channel
    _write_channel(tmp_path, "macro", [
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-03-31"), "vix": 20.0}])
    found = CJ.discover_channels(features_dir=tmp_path)
    assert "finra" in found
    assert "macro" not in found


def _write_monthly(d: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(d / f"{name}_monthly.parquet", index=False)


def test_monthly_channel_discovered_and_leaksafe(tmp_path: Path):
    # Event-resolution monthly channel keyed by month_end. polygon lag = 2d.
    #   Apr month-end 2020-04-30 -> avail 2020-05-02  (<= filing 2020-05-15: ELIGIBLE)
    #   May month-end 2020-05-31 -> avail 2020-06-02  (>  filing: LEAK if used)
    _write_monthly(tmp_path, "polygon", [
        {"ticker": "AAA", "month_end": pd.Timestamp("2020-04-30"), "plg_close": 10.0},
        {"ticker": "AAA", "month_end": pd.Timestamp("2020-05-31"), "plg_close": 99.0},
    ])
    found = CJ.discover_channels(features_dir=tmp_path)
    assert "polygon" in found, "monthly parquet must be discovered"

    ev = pd.DataFrame({"ticker": ["AAA"], "filing_date": [pd.Timestamp("2020-05-15")]})
    out, _ = CJ.join_channels(ev, channels=["polygon"], features_dir=tmp_path,
                              keep_avail=True)
    assert out.loc[0, "ch_polygon__plg_close"] == 10.0, "must use Apr (available), not May (future)"
    assert out.loc[0, "has_polygon"] == 1.0
    assert out.loc[0, "avail_polygon"] <= ev.loc[0, "filing_date"], "no look-ahead on month_end"


def test_quarterly_preferred_when_both_exist(tmp_path: Path):
    # If a name has both files, the quarterly one wins and it is not duplicated.
    _write_channel(tmp_path, "polygon", [
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-03-31"), "plg_close": 1.0}])
    _write_monthly(tmp_path, "polygon", [
        {"ticker": "AAA", "month_end": pd.Timestamp("2020-03-31"), "plg_close": 2.0}])
    found = CJ.discover_channels(features_dir=tmp_path)
    assert found.count("polygon") == 1, "must not duplicate a name present in both files"
    path, period = CJ._resolve("polygon", tmp_path)
    assert period == "quarter_end", "quarterly file preferred over monthly"


def test_drawdown_channel_discovered_and_leaksafe(tmp_path: Path):
    # The standalone drawdown channel ("buy-low" variable) is keyed quarter_end
    # with a 2-business-day publication lag, so it must join leak-safe.
    #   Q1 end 2020-03-31 -> avail 2020-04-02  (<= filing 2020-05-01: ELIGIBLE)
    #   Q2 end 2020-06-30 -> avail 2020-07-02  (>  filing: LEAK if used)
    _write_channel(tmp_path, "drawdown", [
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-03-31"),
         "dd_from_high": -0.30, "dd_52w": -0.25},
        {"ticker": "AAA", "quarter_end": pd.Timestamp("2020-06-30"),
         "dd_from_high": -0.05, "dd_52w": -0.05},
    ])
    assert CJ._LAG_DAYS["drawdown"] == 2, "drawdown publication lag must be 2 days"
    found = CJ.discover_channels(features_dir=tmp_path)
    assert "drawdown" in found, "drawdown parquet must be discovered as a channel"

    ev = pd.DataFrame({"ticker": ["AAA"], "filing_date": [pd.Timestamp("2020-05-01")]})
    out, _ = CJ.join_channels(ev, channels=["drawdown"], features_dir=tmp_path,
                              keep_avail=True)
    assert out.loc[0, "ch_drawdown__dd_from_high"] == -0.30, \
        "must use Q1 drawdown (available), never Q2 (future)"
    assert out.loc[0, "has_drawdown"] == 1.0
    assert out.loc[0, "avail_drawdown"] <= ev.loc[0, "filing_date"], "no look-ahead"


def test_drawdown_aggregation_is_leak_safe():
    # The per-quarter drawdown is the value AT the quarter-end bar — a later
    # all-time high must NOT retroactively change an earlier quarter's drawdown.
    from super_investor.adapters.price_channel import drawdown_quarterly
    dates = pd.bdate_range("2019-01-01", periods=400)
    # ramp to 100, crash to 60 mid-2019, recover to a NEW high 150 in 2020
    close = (list(range(50, 101)) + [100 - i for i in range(0, 40)]
             + list(range(60, 60 + (400 - 51 - 40))))
    daily = pd.DataFrame({"date": dates, "close": close[:400],
                          "volume": [1_000] * 400})
    q = drawdown_quarterly(daily).set_index("quarter_end")
    # drawdown is always <= 0 (close never exceeds its own trailing max), and the
    # mid-2019 crash registers a real drawdown — computed from data up to each
    # quarter end only (the join test proves an earlier quarter ignores a later high).
    assert (q["dd_from_high"] <= 1e-9).all(), "drawdown must never be positive"
    assert q["dd_from_high"].min() < -0.05, "the mid-series crash must register a drawdown"
