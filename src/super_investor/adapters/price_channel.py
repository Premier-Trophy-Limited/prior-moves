"""Turn daily price bars into a leak-safe per-(ticker, quarter_end) feature row.

Shared by the Polygon / Tiingo / IEX channel builders — they all produce the
same (date, close, volume) frame, so the quarter-bucketing lives here once.

Features per quarter (prefixed by provider, e.g. ``plg_`` / ``tng_``):
  <p>_close        last close in the quarter
  <p>_ret_q        close-to-close return over the quarter
  <p>_vol_q        annualized realized vol of daily log returns in the quarter
  <p>_advol        mean daily volume in the quarter
  <p>_mom_12m      trailing 12-month (4-quarter) price momentum
  <p>_dd_from_high drawdown vs all-time trailing high, AT the quarter end ("buy-low")
  <p>_dd_52w       drawdown vs trailing-252-trading-day high, AT the quarter end

All of these are knowable AT the quarter end (no forward data), so the joiner's
availability lag (quarter_end + a couple of settlement days) keeps it leak-safe.
The drawdown columns let the model see how far a name has fallen from its high —
Howard's "buy low" variable — without ever peeking past the quarter end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Trailing-window length for the 52-week high, in trading days (~252/yr).
_DD_52W_WINDOW = 252
# Minimum bars before a trailing-high drawdown is meaningful (a fresh listing
# would otherwise read dd≈0 forever); below this the 52w column is left NaN.
_DD_MIN_BARS = 20


def _with_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row dd_from_high (vs cumulative trailing max) and dd_52w (vs the
    trailing-252-bar max). Both <= 0; 0 means at a new high. `df` must be sorted
    ascending by date with a positive `close`."""
    cummax = df["close"].cummax()
    df["dd_from_high"] = np.where(cummax > 0, df["close"] / cummax - 1.0, np.nan)
    roll_max = df["close"].rolling(_DD_52W_WINDOW, min_periods=_DD_MIN_BARS).max()
    df["dd_52w"] = np.where(roll_max > 0, df["close"] / roll_max - 1.0, np.nan)
    return df


def _prep_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Clean (date, close, volume): tz-naive, positive close, sorted, + drawdown
    + log-return + quarter period. Empty frame in -> empty frame out."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = daily[["date", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.dropna(subset=["close"]).sort_values("date")
    df = df[df["close"] > 0]
    if df.empty:
        return df
    df = _with_drawdown(df)
    df["logret"] = np.log(df["close"]).diff()
    df["q"] = df["date"].dt.to_period("Q")
    return df


def aggregate_to_quarterly(daily: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """daily[date, close, volume] -> quarterly feature rows (no ticker column).

    Returns columns: quarter_end + the seven prefixed features. Empty in -> empty out.
    """
    df = _prep_daily(daily)
    if df.empty:
        return pd.DataFrame(columns=["quarter_end"])

    g = df.groupby("q")
    out = pd.DataFrame({
        f"{prefix}_close": g["close"].last(),
        f"{prefix}_vol_q": g["logret"].std() * np.sqrt(252),
        f"{prefix}_advol": g["volume"].mean(),
        # dd is the value AT the quarter-end bar (last row of the quarter)
        f"{prefix}_dd_from_high": g["dd_from_high"].last(),
        f"{prefix}_dd_52w": g["dd_52w"].last(),
    })
    out[f"{prefix}_ret_q"] = out[f"{prefix}_close"].pct_change()
    out[f"{prefix}_mom_12m"] = out[f"{prefix}_close"].pct_change(4)
    out = out.reset_index()
    # period -> the quarter-END timestamp (the as-of anchor the joiner lags from)
    out["quarter_end"] = out["q"].dt.to_timestamp(how="end").dt.normalize()
    out = out.drop(columns=["q"])
    cols = ["quarter_end", f"{prefix}_close", f"{prefix}_ret_q",
            f"{prefix}_vol_q", f"{prefix}_advol", f"{prefix}_mom_12m",
            f"{prefix}_dd_from_high", f"{prefix}_dd_52w"]
    return out[cols].replace([np.inf, -np.inf], np.nan)


def drawdown_quarterly(daily: pd.DataFrame) -> pd.DataFrame:
    """daily[date, close, volume] -> per-quarter drawdown rows (no ticker column).

    The standalone, key-free drawdown channel: just the two leak-safe columns
    ``dd_from_high`` / ``dd_52w`` at each quarter end. Empty in -> empty out.
    """
    df = _prep_daily(daily)
    if df.empty:
        return pd.DataFrame(columns=["quarter_end"])
    g = df.groupby("q")
    out = pd.DataFrame({
        "dd_from_high": g["dd_from_high"].last(),
        "dd_52w": g["dd_52w"].last(),
    }).reset_index()
    out["quarter_end"] = out["q"].dt.to_timestamp(how="end").dt.normalize()
    out = out.drop(columns=["q"])
    return out[["quarter_end", "dd_from_high", "dd_52w"]].replace(
        [np.inf, -np.inf], np.nan)


def current_drawdown(daily: pd.DataFrame) -> dict[str, float] | None:
    """Latest (most-recent-bar) drawdown snapshot for the live "buy-the-dip"
    display tilt — NOT used in any backtest. Returns {dd_from_high, dd_52w} or
    None if there isn't enough history."""
    df = _prep_daily(daily)
    if df.empty:
        return None
    last = df.iloc[-1]
    dd_high = last["dd_from_high"]
    dd_52 = last["dd_52w"]
    if pd.isna(dd_high):
        return None
    return {
        "dd_from_high": float(dd_high),
        "dd_52w": float(dd_52) if pd.notna(dd_52) else float(dd_high),
    }
