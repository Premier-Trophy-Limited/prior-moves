"""Leak-safe as-of channel joiner — aggregate every cached signal into the
event feature space.

The repo has ~35 cached `data/features/*_quarterly.parquet` channels (news tone,
FINRA short interest, options skew, XBRL fundamentals, insider flow, social, FT
Alphaville, congress, …), each keyed (quarter_end, ticker). The corporate-event
model only used 21 features and ignored all of them. This module joins them onto
an event frame (ticker, filing_date) without look-ahead.

THE CORE LEAK RISK: a parquet keyed by `quarter_end` is NOT public at the quarter
end. 13F breadth is filed ~45d later; 10-Q XBRL ~40-75d; FINRA short ~16d. Joining
on `quarter_end <= filing_date` would let a feature use data that was not yet public
at the event. So every channel joins on an AVAILABILITY date = quarter_end + a
per-channel publication lag, and only the most recent row whose availability date is
on/before filing_date is used.

Pure / network-free: reads cached parquets only, returns a widened frame. No prices,
no scoring.py. $0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
FEATURES = REPO / "data" / "features"

# Publication lag (business-conservative) from quarter_end to the date the quarterly
# aggregate is fully public. Event-like channels (filings, news, social) are public
# within days of quarter end; fundamentals and institutional ownership lag months.
_DEFAULT_LAG = 75
_LAG_DAYS: dict[str, int] = {
    # event-like — public within days of the quarter end
    "edgar_8k": 5, "sec_13d": 5, "congress": 5, "openinsider": 5, "form4": 8,
    "dataroma": 45, "news_rss": 5, "nyt": 5, "guardian": 5, "alpha_vantage": 5,
    "gdelt": 5, "alphaville": 5, "hackernews": 5, "reddit": 5, "reddit_history": 5,
    "stocktwits": 5, "stocktwits_trending": 5, "substack": 5, "wikipedia": 5,
    "damodaran": 5, "tipranks": 5, "edgar_riskfactors": 8,
    # filing-text embeddings — text is public at the filing date; +1 day avail
    # means an event can only see a strictly-earlier filing's embedding (leak guard)
    "filings_embeddings": 1,
    # market-microstructure — short settlement / options reporting
    "finra": 20, "options": 5,
    # price-derived channels — quarter-end close/vol/momentum/drawdown, public ~2bd later
    "polygon": 2, "tiingo": 2, "iex": 2, "drawdown": 2,
    # fundamentals / ownership — statutory filing lags
    "sec_xbrl": 75, "stockanalysis": 75, "yfinance": 75, "finviz": 75,
    "sec_13f_breadth": 50, "form4_agg": 45,
}

# Channels to skip: non-ticker macro (already wired separately), empty, or dup.
_SKIP = {"macro", "cftc_cot", "google_trends", "insider_monkey", "macrotrends",
         "gurufocus", "seekingalpha", "whalewisdom"}

# Period-anchor columns, in priority order. A channel parquet carries exactly one;
# the join lags availability off it. Monthly channels (event-resolution) use
# month_end/period_end; quarterly channels use quarter_end.
_PERIOD_COLS = ("quarter_end", "month_end", "period_end")

# Non-feature columns dropped before the join (ids, raw text blobs).
_DROP_COLS = {"ticker", "quarter_end", "month_end", "period_end", "cusip",
              "subreddit", "joined_text", "joined_titles"}


def _resolve(name: str, features_dir: Path) -> tuple[Path, str] | None:
    """Locate a channel's parquet (quarterly preferred, then monthly) and its
    period-anchor column. Returns (path, period_col) or None if not joinable."""
    for suffix in ("_quarterly.parquet", "_monthly.parquet"):
        p = Path(features_dir) / f"{name}{suffix}"
        if not p.exists():
            continue
        try:
            head = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if head.empty or "ticker" not in head.columns:
            continue
        period = next((c for c in _PERIOD_COLS if c in head.columns), None)
        if period is None:
            continue
        return p, period
    return None


def _naive(s: pd.Series) -> pd.Series:
    """Coerce a datetime series to tz-naive (channels mix tz-aware/naive)."""
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def _numeric_features(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns (drop ids, dates, and object/text columns)."""
    out = []
    for c in df.columns:
        if c in _DROP_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def discover_channels(features_dir: Path = FEATURES) -> list[str]:
    """Names of joinable channels (cached, ticker-keyed, non-empty, not skipped).

    Discovers both quarterly and monthly (event-resolution) channels. A name found
    in both prefers the quarterly file (via _resolve) — they're not duplicated.
    """
    names: list[str] = []
    seen: set[str] = set()
    for pattern in ("*_quarterly.parquet", "*_monthly.parquet"):
        for p in sorted(Path(features_dir).glob(pattern)):
            name = p.name.replace("_quarterly.parquet", "").replace("_monthly.parquet", "")
            if name in _SKIP or name in seen:
                continue
            resolved = _resolve(name, features_dir)
            if resolved is None:
                continue
            head = pd.read_parquet(resolved[0])
            if not _numeric_features(head):
                continue
            seen.add(name)
            names.append(name)
    return names


def channel_columns(name: str, features_dir: Path = FEATURES) -> list[str]:
    """The feature column names a channel contributes (prefixed by channel for
    uniqueness), plus its coverage flag — without doing the join."""
    resolved = _resolve(name, features_dir)
    if resolved is None:
        return []
    df = pd.read_parquet(resolved[0])
    feats = _numeric_features(df)
    return [f"ch_{name}__{c}" for c in feats] + [f"has_{name}"]


def join_channels(events: pd.DataFrame, channels: list[str] | None = None,
                  date_col: str = "filing_date",
                  features_dir: Path = FEATURES,
                  keep_avail: bool = False) -> tuple[pd.DataFrame, dict]:
    """As-of join the requested channels onto `events` (must have ticker + date_col).

    Returns (widened_frame, info) where info maps channel -> list of added feature
    column names. Each channel contributes ch_<name>__<col> features (neutral-filled
    0 where no row was available) and a has_<name> coverage flag in {0,1}.

    Leak-safe: a channel row is eligible only if quarter_end + lag <= filing_date.
    keep_avail=True adds a per-channel `avail_<name>` column (the matched availability
    date) for leak diagnostics/tests.
    """
    features_dir = Path(features_dir)
    if channels is None:
        channels = discover_channels(features_dir)
    ev = events.copy()
    ev[date_col] = _naive(ev[date_col])
    ev = ev.sort_values(date_col).reset_index(drop=True)
    info: dict[str, list[str]] = {}
    new_cols: dict[str, np.ndarray] = {}  # accumulate, concat once (avoid fragmentation)

    for name in channels:
        resolved = _resolve(name, features_dir)
        if resolved is None:
            continue
        p, period_col = resolved
        ch = pd.read_parquet(p)
        feats = _numeric_features(ch)
        if not feats or "ticker" not in ch.columns:
            continue
        lag = _LAG_DAYS.get(name, _DEFAULT_LAG)
        ch = ch[["ticker", period_col] + feats].copy()
        # availability = period end + publication lag; monthly channels lag off
        # month_end the same way quarterly ones lag off quarter_end.
        ch["avail"] = _naive(ch[period_col]) + pd.Timedelta(days=lag)
        ch = ch.dropna(subset=["avail"]).sort_values("avail")
        # collapse duplicate (ticker, avail) keeping last (latest restatement)
        ch = ch.drop_duplicates(["ticker", "avail"], keep="last")
        ren = {c: f"ch_{name}__{c}" for c in feats}
        ch = ch.rename(columns=ren)
        feat_cols = list(ren.values())

        merged = pd.merge_asof(
            ev, ch[["ticker", "avail"] + feat_cols],
            left_on=date_col, right_on="avail", by="ticker", direction="backward")
        cov = merged["avail"].notna().astype(float)
        for c in feat_cols:
            # neutral-fill missing with 0 (coverage flag carries absence);
            # ratio channels can carry +/-inf (div-by-zero) -> neutralize too.
            new_cols[c] = (merged[c].replace([np.inf, -np.inf], np.nan)
                           .fillna(0.0).to_numpy())
        new_cols[f"has_{name}"] = cov.to_numpy()
        if keep_avail:
            new_cols[f"avail_{name}"] = merged["avail"].to_numpy()
        info[name] = feat_cols + [f"has_{name}"]

    if new_cols:
        ev = pd.concat([ev, pd.DataFrame(new_cols, index=ev.index)], axis=1)
    return ev, info
