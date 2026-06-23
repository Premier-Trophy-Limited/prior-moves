"""Congressional Periodic Transaction Reports (PTRs) — Senate + House.

STOCK Act of 2012 requires members of Congress to disclose stock trades
within 45 days. Two well-maintained aggregators expose the data as JSON:

- Senate: github.com/timothycarambat/senate-stock-watcher-data
- House:  github.com/TattooedHead/house-stock-watcher-data

Both publish `all_transactions.json` (per-transaction rows). House additionally
includes ``disclosure_date`` and ``amount_mid`` (midpoint of the bracketed
range "$15,001-$50,000" → 32,500), which the Senate feed lacks.

The 45-day disclosure window means the signal lags the trade by 1-2 months,
but the typical 13F quarterly cycle is ~45 days too — so by the time a 13F
filer's holdings publish, the congressional trades from that same quarter are
also visible. Useful as cross-validation: when politicians and super-investors
hit the same name in the same quarter, that's a stronger pick.

Per-(ticker, quarter_end) aggregates emitted with ``cg_`` prefix:
  cg_n_buys, cg_n_sells, cg_n_exchanges,
  cg_n_unique_filers, cg_dollar_mid_total,
  cg_buy_sell_ratio  (buys/(buys+sells), Laplace-smoothed)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("super_investor.adapters.congress")

_SENATE_URL = (
    "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/"
    "master/aggregate/all_transactions.json"
)
_HOUSE_URL = (
    "https://raw.githubusercontent.com/TattooedHead/house-stock-watcher-data/"
    "main/data/all_transactions.json"
)

# Map raw "type" strings to canonical action: buy / sell / exchange / other
_BUY_PATTERNS = re.compile(
    r"\b(purchase|buy|acquire|acquisition)\b", re.IGNORECASE
)
_SELL_PATTERNS = re.compile(
    r"\b(sale|sell|sold)\b", re.IGNORECASE
)
_EXCHANGE_PATTERNS = re.compile(
    r"\b(exchange|transfer)\b", re.IGNORECASE
)

# Senate amount ranges → midpoint dollar value (House provides this column
# natively, Senate doesn't). Buckets per the official PTR disclosure form.
_SENATE_AMOUNT_MID = {
    "$1,001 - $15,000": 8_000,
    "$15,001 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000,
    "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000,
    "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "Over $50,000,000": 50_000_000,
}


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)
    ),
)
def _fetch_json(url: str, cache_path: Path | None = None) -> list[dict]:
    """Download a remote JSON file (with on-disk cache)."""
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_bytes())
    with httpx.Client(timeout=120.0, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        data = r.json()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(json.dumps(data).encode())
    return data


def _classify_action(raw_type: str) -> str:
    t = str(raw_type or "")
    if _BUY_PATTERNS.search(t):
        return "buy"
    if _SELL_PATTERNS.search(t):
        return "sell"
    if _EXCHANGE_PATTERNS.search(t):
        return "exchange"
    return "other"


def _coerce_date(s: str | None) -> pd.Timestamp:
    if not s:
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def normalize_senate(rows: list[dict]) -> pd.DataFrame:
    """Senate JSON → normalized DataFrame."""
    out: list[dict] = []
    for r in rows:
        ticker = str(r.get("ticker") or "").strip().upper()
        # Skip non-ticker placeholders ("--", "<>", "")
        if not ticker or ticker in ("--", "<>"):
            continue
        out.append({
            "chamber": "senate",
            "filer": str(r.get("senator") or ""),
            "ticker": ticker,
            "transaction_date": _coerce_date(r.get("transaction_date")),
            "disclosure_date": pd.NaT,
            "action": _classify_action(r.get("type")),
            "amount_mid": _SENATE_AMOUNT_MID.get(str(r.get("amount") or "").strip(), float("nan")),
            "owner": str(r.get("owner") or ""),
        })
    return pd.DataFrame(out)


def normalize_house(rows: list[dict]) -> pd.DataFrame:
    """House JSON → normalized DataFrame."""
    out: list[dict] = []
    for r in rows:
        ticker = str(r.get("ticker") or "").strip().upper()
        if not ticker or ticker in ("--", "<>"):
            continue
        # House amount_mid is already a number; coerce safely
        amt = r.get("amount_mid")
        try:
            amt = float(amt) if amt is not None else float("nan")
        except Exception:
            amt = float("nan")
        out.append({
            "chamber": "house",
            "filer": str(r.get("representative") or ""),
            "ticker": ticker,
            "transaction_date": _coerce_date(r.get("transaction_date")),
            "disclosure_date": _coerce_date(r.get("disclosure_date")),
            "action": _classify_action(r.get("type")),
            "amount_mid": amt,
            "owner": str(r.get("owner") or ""),
        })
    return pd.DataFrame(out)


def aggregate_to_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-transaction rows up to (ticker, quarter_end) features."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["transaction_date", "ticker"]).copy()
    if df.empty:
        return pd.DataFrame()
    # Drop transactions dated in the future (data-entry / parse errors in the
    # STOCK Act disclosures occasionally carry a forward date).
    now = pd.Timestamp.utcnow().tz_localize(None)
    df = df[df["transaction_date"] <= now]
    if df.empty:
        return pd.DataFrame()
    df["quarter_end"] = (
        df["transaction_date"].dt.to_period("Q").dt.end_time.dt.normalize()
    )

    df["is_buy"] = (df["action"] == "buy").astype(int)
    df["is_sell"] = (df["action"] == "sell").astype(int)
    df["is_exchange"] = (df["action"] == "exchange").astype(int)

    agg = (
        df.groupby(["ticker", "quarter_end"])
        .agg(
            cg_n_buys=("is_buy", "sum"),
            cg_n_sells=("is_sell", "sum"),
            cg_n_exchanges=("is_exchange", "sum"),
            cg_n_unique_filers=("filer", "nunique"),
            cg_dollar_mid_total=("amount_mid", "sum"),
        )
        .reset_index()
    )
    # Laplace-smoothed buy/sell ratio (add 1 to each side so a single trade
    # doesn't generate degenerate 0/1 values for sparse tickers).
    agg["cg_buy_sell_ratio"] = (agg["cg_n_buys"] + 1.0) / (
        agg["cg_n_buys"] + agg["cg_n_sells"] + 2.0
    )
    return agg


def pull_all(
    cache_dir: Path | None = None,
    senate_url: str = _SENATE_URL,
    house_url: str = _HOUSE_URL,
) -> pd.DataFrame:
    """Fetch both chambers' transaction feeds, normalize, concatenate."""
    senate_cache = cache_dir / "senate_all.json" if cache_dir else None
    house_cache = cache_dir / "house_all.json" if cache_dir else None
    senate = normalize_senate(_fetch_json(senate_url, senate_cache))
    house = normalize_house(_fetch_json(house_url, house_cache))
    return pd.concat([senate, house], ignore_index=True)
