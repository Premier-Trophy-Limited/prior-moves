"""Tests for the no-look-ahead 13F backtest harness.

Three invariants, each a failing-first test:
  1. LOOK-AHEAD GUARD — every constituent filing's public date <= rebalance
     date, else the harness raises LookAheadError.
  2. KNOWN-ANSWER FIXTURE — a hand-computed small set reproduces holdings +
     P&L exactly.
  3. DETERMINISM — same inputs -> identical result (no wall-clock, no random,
     no network).

Plus a leak-demonstration test: keying the filter on period_of_report instead
of filed_at admits a not-yet-public filing and changes the edge, proving the
date filter matters.
"""
from __future__ import annotations

import pandas as pd
import pytest

from super_investor.leakproof_backtest import (
    LookAheadError,
    assert_no_lookahead,
    make_top_k_select,
    run_backtest,
)
from tests.fixtures.leakproof_holdings import HOLDINGS, price_fn


# --- 1. LOOK-AHEAD GUARD ----------------------------------------------------

def test_guard_raises_on_future_filing():
    """A holding whose public date is AFTER the rebalance date must trip."""
    R = pd.Timestamp("2020-05-15")
    leaked = HOLDINGS  # contains 2020-08-14 filings, all > R
    with pytest.raises(LookAheadError):
        assert_no_lookahead(leaked, R, date_field="filed_at")


def test_guard_passes_when_all_public():
    R = pd.Timestamp("2020-08-14")
    public = HOLDINGS[HOLDINGS["filed_at"] <= R]
    # Must not raise.
    assert_no_lookahead(public, R, date_field="filed_at")


def test_filter_excludes_unpublished_filing():
    """At Q1's public date the portfolio cannot contain a Q2-only ticker."""
    res = run_backtest(
        HOLDINGS,
        rebalance_dates=[pd.Timestamp("2020-05-15")],
        price_fn=price_fn,
    )
    rb = res.rebalances[0]
    # DDD only exists in the Q2 filing (public 08-14) -> must be absent.
    assert "DDD" not in rb.tickers
    assert set(rb.tickers) == {"AAA", "BBB"}


# --- 2. KNOWN-ANSWER FIXTURE ------------------------------------------------

def test_known_answer_holdings_and_pnl():
    res = run_backtest(
        HOLDINGS,
        rebalance_dates=[pd.Timestamp("2020-05-15"), pd.Timestamp("2020-08-14")],
        price_fn=price_fn,
    )
    assert res.quarters == 2

    r1, r2 = res.rebalances
    # Q1 public: new entries AAA(.20), BBB(.10) -> port .15, spy .05, edge .10
    assert set(r1.tickers) == {"AAA", "BBB"}
    assert r1.port_return == pytest.approx(0.15)
    assert r1.spy_return == pytest.approx(0.05)
    assert r1.edge == pytest.approx(0.10)

    # Q2 public: new entries DDD(-.10), AAA(.30) -> port .10, spy .08, edge .02
    assert set(r2.tickers) == {"AAA", "DDD"}
    assert r2.port_return == pytest.approx(0.10)
    assert r2.spy_return == pytest.approx(0.08)
    assert r2.edge == pytest.approx(0.02)

    # avg edge = (0.10 + 0.02)/2 = 0.06
    assert res.avg_edge == pytest.approx(0.06)
    # cumulative edge curve [0.10, 0.12] is monotone up -> no drawdown
    assert res.max_drawdown == pytest.approx(0.0)


# --- 3. DETERMINISM ---------------------------------------------------------

def test_determinism_identical_runs():
    dates = [pd.Timestamp("2020-05-15"), pd.Timestamp("2020-08-14")]
    a = run_backtest(HOLDINGS, rebalance_dates=dates, price_fn=price_fn)
    b = run_backtest(HOLDINGS, rebalance_dates=dates, price_fn=price_fn)
    assert a == b
    assert [r.edge for r in a.rebalances] == [r.edge for r in b.rebalances]


# --- LEAK DEMONSTRATION -----------------------------------------------------

def test_leak_inflates_when_filtering_on_period():
    """Rebalance 2020-07-01 sits between Q2 period-end (06-30) and its public
    date (08-14). Keying the filter on filed_at correctly excludes Q2; keying
    on period_of_report leaks it -> the holdings (and edge) change. Proves the
    date field is load-bearing.
    """
    R = pd.Timestamp("2020-07-01")

    clean = run_backtest(HOLDINGS, rebalance_dates=[R], price_fn=price_fn,
                         date_field="filed_at")
    leaked = run_backtest(HOLDINGS, rebalance_dates=[R], price_fn=price_fn,
                          date_field="period_of_report")

    clean_rb, leaked_rb = clean.rebalances[0], leaked.rebalances[0]
    # Clean: only Q1 filings are public at 07-01 -> AAA, BBB.
    assert set(clean_rb.tickers) == {"AAA", "BBB"}
    # Leaked: Q2 period (06-30) <= 07-01 admits the unpublished Q2 new entries.
    assert "DDD" in leaked_rb.tickers
    # The leak changes the constituent set -> not the same backtest.
    assert set(leaked_rb.tickers) != set(clean_rb.tickers)


# --- MODEL TOP-K SELECT (the 5.4-headline strategy) -------------------------

def _model_holdings() -> pd.DataFrame:
    """Predictions-shaped frame: per (investor, period, cusip) score `p`."""
    rows = [
        # Q1 2020-03-31, public 2020-05-15
        dict(investor_slug="alpha", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="AAA", ticker="AAA", p=0.90),
        dict(investor_slug="alpha", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="CCC", ticker="CCC", p=0.40),
        dict(investor_slug="beta", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="BBB", ticker="BBB", p=0.80),
        # Q2 2020-06-30, public 2020-08-14 (not yet public at 2020-05-15)
        dict(investor_slug="alpha", period_of_report="2020-06-30",
             filed_at="2020-08-14", cusip="DDD", ticker="DDD", p=0.99),
    ]
    df = pd.DataFrame(rows)
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    return df


def test_top_k_select_picks_highest_p_and_respects_filter():
    res = run_backtest(
        _model_holdings(),
        rebalance_dates=[pd.Timestamp("2020-05-15")],
        price_fn=price_fn,
        select_fn=make_top_k_select(top_k=1),
    )
    rb = res.rebalances[0]
    # top-1 per investor from the latest PUBLIC filing (Q1): alpha->AAA(.90),
    # beta->BBB(.80). DDD(.99) is Q2, not public at 05-15 -> excluded.
    assert set(rb.tickers) == {"AAA", "BBB"}
    assert "DDD" not in rb.tickers
    # known P&L: AAA .20, BBB .10 -> port .15, spy .05, edge .10
    assert rb.edge == pytest.approx(0.10)


def test_top_k_select_is_deterministic():
    sel = make_top_k_select(top_k=1)
    dates = [pd.Timestamp("2020-05-15")]
    a = run_backtest(_model_holdings(), rebalance_dates=dates, price_fn=price_fn, select_fn=sel)
    b = run_backtest(_model_holdings(), rebalance_dates=dates, price_fn=price_fn, select_fn=sel)
    assert a == b
