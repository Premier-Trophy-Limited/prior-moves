"""Known-answer fixture for the leak-proof backtest harness.

Three investors, two quarters, hand-computed returns so the test asserts
EXACT holdings and P&L — not approximate. No network, no files: every value
is literal here.

Timeline (the whole point — public availability lags the period):

  Q1 period_of_report = 2020-03-31, filings public 2020-05-15 (45-day lag)
  Q2 period_of_report = 2020-06-30, filings public 2020-08-14

Rebalance dates are placed at the PUBLIC date, so a clean filter admits the
quarter whose filings have published and rejects any not-yet-public filing.

The `period_of_report = 2020-06-30` rows carry `filed_at = 2020-08-14`. A
rebalance on 2020-05-15 (Q1's public date) must therefore see ONLY the Q1
rows. A leaky filter keyed on period_of_report would wrongly admit the Q2
rows whose period (06-30) is... still after 05-15, so to demonstrate the leak
we add a THIRD, late-period filing: period 2020-06-30 but a clerical early
publish — no. Instead the leak demo uses a rebalance placed BETWEEN a
period-end and its public date (see PRICES / test_leak).
"""
from __future__ import annotations

import pandas as pd


# One row per (investor, period, cusip). label drives the default select rule.
# filed_at is the public-availability date (period_end + ~45d).
HOLDINGS = pd.DataFrame(
    [
        # --- Q1 2020-03-31, public 2020-05-15 ---
        dict(investor_slug="alpha", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="AAA", ticker="AAA",
             label="new_entry", value_usd=1_000_000),
        dict(investor_slug="beta", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="BBB", ticker="BBB",
             label="new_entry", value_usd=2_000_000),
        dict(investor_slug="alpha", period_of_report="2020-03-31",
             filed_at="2020-05-15", cusip="CCC", ticker="CCC",
             label="hold", value_usd=500_000),
        # --- Q2 2020-06-30, public 2020-08-14 ---
        dict(investor_slug="alpha", period_of_report="2020-06-30",
             filed_at="2020-08-14", cusip="DDD", ticker="DDD",
             label="new_entry", value_usd=3_000_000),
        dict(investor_slug="beta", period_of_report="2020-06-30",
             filed_at="2020-08-14", cusip="AAA", ticker="AAA",
             label="new_entry", value_usd=1_500_000),
    ]
)
HOLDINGS["period_of_report"] = pd.to_datetime(HOLDINGS["period_of_report"])
HOLDINGS["filed_at"] = pd.to_datetime(HOLDINGS["filed_at"])


# Deterministic price table: return over the 1-quarter hold starting at the
# given rebalance date. Keyed (ticker, rebalance_date_str) -> hold return.
# Hand-picked so the known-answer P&L is trivial to verify.
_RETURNS = {
    # Rebalance 2020-05-15 (Q1 public): AAA & BBB are the new entries.
    ("AAA", "2020-05-15"): 0.20,
    ("BBB", "2020-05-15"): 0.10,
    ("SPY", "2020-05-15"): 0.05,
    # Rebalance 2020-08-14 (Q2 public): DDD & AAA are the new entries.
    ("DDD", "2020-08-14"): -0.10,
    ("AAA", "2020-08-14"): 0.30,
    ("SPY", "2020-08-14"): 0.08,
    # Leak-demo rebalance 2020-07-01 (AFTER Q2 period-end 06-30, BEFORE its
    # public date 08-14). A period-keyed filter LEAKS the Q2 filing here.
    ("DDD", "2020-07-01"): -0.10,
    ("AAA", "2020-07-01"): 0.30,
    ("BBB", "2020-07-01"): 0.10,
    ("CCC", "2020-07-01"): 0.0,
    ("SPY", "2020-07-01"): 0.08,
}


def price_fn(ticker: str, asof: pd.Timestamp, hold_quarters: int = 1) -> float:
    """Deterministic injected price function — no network, no files."""
    key = (ticker, pd.Timestamp(asof).strftime("%Y-%m-%d"))
    return _RETURNS.get(key, float("nan"))
