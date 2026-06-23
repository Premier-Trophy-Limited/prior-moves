"""yfinance price history adapter for mimic backtest.

For each ticker we cache the full daily history once and re-use across the
backtest. Quarter-end → next-quarter-end returns are derived on demand.

Cache: data/prices_cache/<ticker>.parquet  (columns: date, close, volume)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger("super_investor.adapters.prices")


def get_history(ticker: str, cache_dir: Path,
                start: str = "2014-01-01", end: str | None = None,
                cache_only: bool = False, provider: str = "yfinance") -> pd.DataFrame:
    """Daily (date, close, volume) history, cached per ticker.

    ``provider`` selects the live source on a cache miss: ``"yfinance"`` (default,
    unchanged), ``"polygon"``, or ``"tiingo"``. All providers write the SAME
    parquet schema, so existing ``cache_only`` backtests are provider-agnostic —
    once a ticker is cached, the source it came from is irrelevant.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df
    if cache_only:
        # Offline mode: never hit the network. Used by the leak-free event
        # backtest, which prices the whole mirror (incl. junk bond-CUSIP rows)
        # and must stay deterministic — a cache miss is simply NaN, not a slow
        # live yfinance lookup of hundreds of delisted/invalid symbols.
        return pd.DataFrame(columns=["date", "close", "volume"])

    if provider in ("polygon", "tiingo"):
        if provider == "polygon":
            from .polygon import fetch_daily_bars
        else:
            from .tiingo import fetch_daily_bars
        hist = fetch_daily_bars(ticker, start=start, end=end, cache_dir=cache_dir)
        if hist.empty:
            return pd.DataFrame(columns=["date", "close", "volume"])
        hist = hist[["date", "close", "volume"]].copy()
        hist.to_parquet(cache_path, index=False)
        return hist

    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        hist = t.history(start=start, end=end, auto_adjust=True)
    except Exception as e:
        log.warning("get_history(%s): %s: %s", ticker, type(e).__name__, e)
        return pd.DataFrame(columns=["date", "close", "volume"])
    if hist.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    hist = hist.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    hist = hist[["date", "close", "volume"]]
    hist.to_parquet(cache_path, index=False)
    return hist


def quarterly_return(ticker: str, quarter_end: pd.Timestamp, cache_dir: Path,
                     hold_quarters: int = 1, hold_months: float | None = None,
                     cache_only: bool = False) -> float:
    """Close-to-close return from quarter_end through quarter_end + horizon.

    Horizon is ``hold_months`` months if given, else ``3 * hold_quarters``.
    Lets the backtest run multiple holding periods (1mo / 3mo / 6mo) so the
    user isn't implicitly locked into a one-quarter hold.

    Robust to non-trading days: uses the last available close on or before each anchor.
    """
    hist = get_history(ticker, cache_dir, cache_only=cache_only)
    if hist.empty:
        return float("nan")
    months = hold_months if hold_months is not None else 3 * hold_quarters
    end_date = quarter_end + pd.DateOffset(months=int(months)) if float(months).is_integer() \
        else quarter_end + pd.Timedelta(days=int(round(months * 30.44)))

    def _price_at(asof: pd.Timestamp) -> float:
        prior = hist[hist["date"] <= asof]
        if prior.empty:
            return float("nan")
        return float(prior.iloc[-1]["close"])

    p0 = _price_at(quarter_end)
    p1 = _price_at(end_date)
    if p0 <= 0 or not (p0 == p0) or not (p1 == p1):
        return float("nan")
    return float(p1 / p0 - 1.0)
