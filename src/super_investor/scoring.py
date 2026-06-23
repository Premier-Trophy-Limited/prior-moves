"""PriorScore — the single composite ranking formula for Prior Moves.

The headline "Top Picks" list ranks every candidate ticker by ONE number,
**PriorScore (0-100)**, so rank and the displayed conviction badge never
contradict each other. (Previous bug: rank was driven by summed weighted-p
while the badge showed max-p, so #1 could show 70% and #13 show 82%.)

PriorScore is a weighted blend of five percentile-normalized components:

    PriorScore = 100 * (
        0.40 * conviction      # how strongly the models predict it
      + 0.20 * consensus       # how many high-track-record investors agree
      + 0.20 * momentum        # recent price action (real-time tilt)
      + 0.10 * cross_signal    # congress / insider / activist corroboration
      + 0.10 * liquidity       # tradability — the "Buffett-effect" / impact gate
    )

Each raw component is converted to a 0-1 percentile rank ACROSS the surviving
candidate set before weighting, so the score is a relative ranking within the
quarter's universe, not an absolute probability.

A hard **listing filter** runs before scoring: a candidate survives only if it
has a recent market price OR appears in the most-recent 13F holdings of any
tracked investor. This removes delisted / taken-private names (Vigil, DNB,
Berry, etc.) that the raw model still scored from stale feature rows.

Weights live in ``WEIGHTS`` — a frozen, documented dataclass — so they are
easy to tune and audit.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
PRICES_DIR = REPO / "data" / "prices_cache"

# A ticker counts as "recently traded" (listing filter) if its latest cached
# price is on/after this date. Env-overridable (SI_PRICE_RECENCY=YYYY-MM-DD,
# also settable via build_aggregated_forward --price-recency); the default is
# pinned, not wall-clock, so reruns stay reproducible — bump it each quarter
# with the refresh.
PRICE_RECENCY_CUTOFF = pd.Timestamp(os.environ.get("SI_PRICE_RECENCY", "2026-04-01"))

# ETFs / index funds — a stock-picking product must not recommend these, and
# multi-strat quants (Citadel, Millennium, D.E. Shaw) hold them heavily, so they
# flood the consensus aggregate and dilute the backtest toward market return.
ETF_TICKERS: frozenset[str] = frozenset({
    "SPY", "QQQ", "QQQM", "IWM", "DIA", "VOO", "IVV", "VTI", "VEA", "VWO", "VUG",
    "VTV", "VIG", "VYM", "VO", "VB", "EEM", "EFA", "IEMG", "IEFA", "ACWI",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "IBB", "XBI", "VGT", "IGV", "SKYY", "HACK", "ARKK", "ARKG",
    "GLD", "SLV", "GDX", "USO", "UNG", "TLT", "IEF", "SHY", "HYG", "LQD", "AGG",
    "BND", "BNDX", "TIP", "MUB", "SCHD", "JEPI", "JEPQ", "DGRO", "NOBL", "KWEB",
    "FXI", "MCHI", "EWZ", "EWJ", "INDA", "XOP", "OIH", "ITB", "XHB", "KRE", "KBE",
    "BITO", "IBIT", "VXUS", "RSP", "MDY", "SPLG", "SPYG", "SPYV", "QUAL", "MTUM",
    "USMV", "VLUE", "SPLV", "EFV", "EFG", "IJR", "IJH", "VEU", "SPDW",
})


def is_etf(ticker: str) -> bool:
    return str(ticker).upper().strip() in ETF_TICKERS


# Spot crypto ETFs — the ONE ETF category we DO surface. Since the Jan-2024 spot
# approvals, 13F filers report these (IBIT/FBTC/ETHA…), so they are how
# institutional crypto positioning legitimately enters an equity model. They stay
# in ETF_TICKERS (they ARE ETFs) but are whitelisted past the fund filter.
CRYPTO_ETF: frozenset[str] = frozenset({
    "IBIT", "FBTC", "GBTC", "ARKB", "BITB", "HODL", "BTCO", "EZBC", "BRRR", "BTCW",
    "BITO", "ETHA", "FETH", "ETHE", "ETHW", "EZET",
})


def is_crypto_etf(ticker: str) -> bool:
    return str(ticker).upper().strip() in CRYPTO_ETF


# Investors whose models we keep (and whose 13F feeds the breadth features) but
# do NOT mirror in the consensus Top Picks: pure multi-strat market-makers whose
# 13F is thousands of churning, low-conviction/hedging positions. Including
# millennium dropped the backtest t-stat 2.3 -> 1.4 by flooding the consensus.
#
# Roster is env-pinnable for reproducible pick generation:
#   SI_MIRROR_EXCLUDE_EXTRA="slug1,slug2"  -> add to the base exclusion
#   SI_MIRROR_EXCLUDE="slugA,slugB"        -> REPLACE the base exclusion entirely
# Both scoring + aggregation read this, so a pinned roster reproduces the same
# Top Picks every run. Style rationale: the 13F-mirror thesis holds only for
# discretionary concentrated-equity pickers; macro/quant-high-turnover/credit/
# insurance 13Fs are stale slivers of much larger non-equity books.
import os as _os

_BASE_MIRROR_EXCLUDE = {"millennium", "citadel", "de_shaw", "point72"}


def _resolve_mirror_exclude() -> frozenset[str]:
    repl = _os.environ.get("SI_MIRROR_EXCLUDE", "").strip()
    if repl:
        return frozenset(s.strip() for s in repl.split(",") if s.strip())
    extra = _os.environ.get("SI_MIRROR_EXCLUDE_EXTRA", "").strip()
    base = set(_BASE_MIRROR_EXCLUDE)
    if extra:
        base |= {s.strip() for s in extra.split(",") if s.strip()}
    return frozenset(base)


MIRROR_EXCLUDE: frozenset[str] = _resolve_mirror_exclude()


@dataclass(frozen=True)
class ScoreWeights:
    """Component weights for the composite PriorScore. Sum to 1.0."""

    conviction: float = 0.40
    consensus: float = 0.20
    momentum: float = 0.20
    cross_signal: float = 0.10
    liquidity: float = 0.10

    # sub-weights inside conviction (max vs mean of model probabilities)
    conviction_max_share: float = 0.60
    conviction_mean_share: float = 0.40

    # sub-weights inside momentum (1-month vs 3-month return). Tilt toward
    # 1-month so a recent spike (Howard's "Tencent +10% in a day") counts.
    momentum_1m_share: float = 0.60
    momentum_3m_share: float = 0.40

    def validate(self) -> None:
        total = (
            self.conviction + self.consensus + self.momentum
            + self.cross_signal + self.liquidity
        )
        assert abs(total - 1.0) < 1e-9, f"weights must sum to 1.0, got {total}"


def _weights_from_env() -> ScoreWeights:
    """Build component weights from SI_W_* env vars (for the optimization search),
    renormalized to sum 1.0. Unset -> validated defaults. Lets the autonomous
    search explore the weight simplex without code edits."""
    keys = ["conviction", "consensus", "momentum", "cross_signal", "liquidity"]
    raw = {k: _os.environ.get(f"SI_W_{k.upper()}") for k in keys}
    if not any(v is not None for v in raw.values()):
        return ScoreWeights()
    base = ScoreWeights()
    vals = {k: float(raw[k]) if raw[k] is not None else getattr(base, k) for k in keys}
    s = sum(vals.values()) or 1.0
    return ScoreWeights(conviction=vals["conviction"] / s, consensus=vals["consensus"] / s,
                        momentum=vals["momentum"] / s, cross_signal=vals["cross_signal"] / s,
                        liquidity=vals["liquidity"] / s)


WEIGHTS = _weights_from_env()


# ---------------------------------------------------------------------------
# Price-derived features (momentum + liquidity) from the local prices cache
# ---------------------------------------------------------------------------


import functools


@functools.lru_cache(maxsize=4096)
def _load_prices(ticker: str) -> pd.DataFrame | None:
    """Read one ticker's price parquet, dates parsed, sorted.

    lru_cached: the PriorScore backtest calls price_features per
    (ticker, quarter) — 35 quarters × ~1,400 tickers re-read the same ~1,400
    files without this. Cached frames are treated as immutable — callers
    must .copy() before mutating.
    """
    safe = str(ticker).upper().replace("/", "-")
    p = PRICES_DIR / f"{safe}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if "close" not in df.columns or "date" not in df.columns:
        return None
    df = df.dropna(subset=["close"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df if len(df) else None


def price_features(ticker: str, asof: pd.Timestamp | None = None) -> dict:
    """Return momentum + liquidity + recency features for one ticker.

    Keys:
      has_recent_price : bool — traded within the last 60 calendar days
      ret_1m, ret_3m   : trailing simple returns (fractions, e.g. 0.10 = +10%)
      dollar_vol       : mean daily close*volume over last ~21 sessions
      last_date        : most recent price date
    """
    out = {
        "has_recent_price": False,
        "ret_1m": np.nan,
        "ret_3m": np.nan,
        "dollar_vol": np.nan,
        "last_close": np.nan,
        "last_date": pd.NaT,
    }
    df = _load_prices(ticker)
    if df is None or df.empty:
        return out
    if asof is not None:
        df = df[df["date"] <= asof]
        if df.empty:
            return out
    last_date = df["date"].iloc[-1]
    out["last_date"] = last_date
    # listed = last cached print on/after the pinned recency cutoff
    out["has_recent_price"] = bool(pd.Timestamp(last_date) >= PRICE_RECENCY_CUTOFF)
    closes = df["close"].astype(float).to_numpy()
    last = float(closes[-1])
    # Guard against a single bad print (unadjusted split / stale tick): if the
    # latest close is a wild outlier vs the trailing-20 median, fall back to the
    # most recent in-range close. A real stock doesn't 5x (or 1/5) in one
    # session — such a value is a data artifact (e.g. SNDK showing ~$1,590).
    if len(closes) >= 6:
        window = closes[-20:]
        med = float(np.median(window))
        if med > 0 and not (med / 5.0 <= last <= 5.0 * med):
            in_range = window[(window >= med / 5.0) & (window <= 5.0 * med)]
            last = float(in_range[-1]) if len(in_range) else med
    out["last_close"] = last if last == last else np.nan
    if len(closes) > 21 and closes[-22] > 0:
        out["ret_1m"] = last / closes[-22] - 1.0
    if len(closes) > 63 and closes[-64] > 0:
        out["ret_3m"] = last / closes[-64] - 1.0
    if "volume" in df.columns:
        tail = df.tail(21)
        dv = (tail["close"].astype(float) * tail["volume"].astype(float)).mean()
        out["dollar_vol"] = float(dv) if dv == dv else np.nan
    return out


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


_CALIB_GRID: tuple[np.ndarray, np.ndarray] | None = None


def apply_calibration(p) -> "np.ndarray":
    """Map raw model probabilities through the isotonic calibration grid.

    Loads ``runs/per_investor_wf/calibration_grid.parquet`` (built by
    ``scripts/calibrate.py``) once and interpolates. Raw gradient-boosted
    scores rank well but are badly over-confident (the top decile predicts
    ~49% but only ~11% actually become new entries); this returns the
    historically-honest probability. Identity passthrough if the grid is
    missing. Monotone — does NOT change rank ordering.
    """
    global _CALIB_GRID
    arr = np.asarray(p, dtype=float)
    if _CALIB_GRID is None:
        grid_path = REPO / "runs" / "per_investor_wf" / "calibration_grid.parquet"
        if grid_path.exists():
            g = pd.read_parquet(grid_path)
            _CALIB_GRID = (g["x"].to_numpy(float), g["y"].to_numpy(float))
        else:
            _CALIB_GRID = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    xs, ys = _CALIB_GRID
    return np.interp(arr, xs, ys)


def _pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank 0-1 over rows that HAVE data; NaN → 0.5 (neutral).

    Missing data must neither help nor hurt — a name with no momentum signal
    should sit at the median, not be rewarded (old ``na_option='bottom'`` bug
    pushed NaN to a high percentile) nor punished to zero.
    """
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    # rank only the non-null values; leave NaN out of the ranking entirely
    r = x.rank(pct=True)
    return r.fillna(0.5)


# ---------------------------------------------------------------------------
# The composite
# ---------------------------------------------------------------------------


def compute_prior_scores(
    candidates: pd.DataFrame,
    weights: ScoreWeights = WEIGHTS,
) -> pd.DataFrame:
    """Compute PriorScore for a candidate DataFrame.

    Required columns:
      ticker, max_p, mean_p, consensus_raw, momentum_1m, momentum_3m,
      cross_signal_count, dollar_vol

    Returns the frame with added columns:
      conviction_pct, consensus_pct, momentum_pct, cross_pct, liquidity_pct,
      prior_score (0-100)
    sorted by prior_score descending.
    """
    weights.validate()
    df = candidates.copy()

    # 1. Conviction — blend max + mean model probability, then percentile
    conviction_raw = (
        weights.conviction_max_share * df["max_p"].astype(float)
        + weights.conviction_mean_share * df["mean_p"].astype(float)
    )
    df["conviction_pct"] = _pct_rank(conviction_raw)

    # 2. Consensus — sqrt of summed quality weights (diminishing returns),
    #    then percentile
    df["consensus_pct"] = _pct_rank(np.sqrt(df["consensus_raw"].clip(lower=0)))

    # 3. Momentum — recency-tilted blend of 1m + 3m return, then percentile
    momentum_raw = (
        weights.momentum_1m_share * df["momentum_1m"].astype(float)
        + weights.momentum_3m_share * df["momentum_3m"].astype(float)
    )
    df["momentum_pct"] = _pct_rank(momentum_raw)

    # 4. Cross-signal — count of corroborating channels, percentile
    df["cross_pct"] = _pct_rank(df["cross_signal_count"].astype(float))

    # 5. Liquidity — log dollar volume, percentile (Buffett-effect gate)
    log_dv = np.log10(df["dollar_vol"].astype(float).clip(lower=1.0))
    df["liquidity_pct"] = _pct_rank(log_dv)

    df["prior_score"] = 100.0 * (
        weights.conviction * df["conviction_pct"]
        + weights.consensus * df["consensus_pct"]
        + weights.momentum * df["momentum_pct"]
        + weights.cross_signal * df["cross_pct"]
        + weights.liquidity * df["liquidity_pct"]
    )

    # --- Buffett-effect / market-impact haircut ---------------------------
    # The liquidity component nudges; this GATE bites. A name too thin to
    # actually deploy meaningful capital into (low average daily $-volume) is
    # not an actionable "pick" for a real investor, no matter how high the
    # model's conviction. We multiply the score by a haircut that scales from
    # 1.0 (deep liquidity) down to 0.7 (microcap), and hard-flag the tier.
    dv = df["dollar_vol"].astype(float)
    LOW = 5_000_000.0     # < $5M ADV  → severe impact, deepest haircut
    MID = 50_000_000.0    # < $50M ADV → some impact
    def _haircut(v: float) -> float:
        if not (v == v):           # NaN — unknown liquidity, mild caution
            return 0.92
        if v >= MID:
            return 1.0
        if v <= LOW:
            return 0.70
        # log-linear between LOW and MID → 0.70..1.0
        frac = (math.log10(v) - math.log10(LOW)) / (math.log10(MID) - math.log10(LOW))
        return 0.70 + 0.30 * frac
    df["impact_haircut"] = dv.apply(_haircut)

    def _tier(v: float) -> str:
        if not (v == v):
            return "unknown"
        if v >= 200_000_000:
            return "mega"
        if v >= MID:
            return "large"
        if v >= LOW:
            return "mid"
        return "thin"
    df["liquidity_tier"] = dv.apply(_tier)

    df["prior_score"] = df["prior_score"] * df["impact_haircut"]

    return df.sort_values("prior_score", ascending=False).reset_index(drop=True)


def score_label(score: float) -> str:
    """0-100 PriorScore → 1-5 star string (single source of truth for badges)."""
    if score >= 75:
        return "★★★★★"
    if score >= 60:
        return "★★★★☆"
    if score >= 45:
        return "★★★☆☆"
    if score >= 30:
        return "★★☆☆☆"
    return "★☆☆☆☆"
