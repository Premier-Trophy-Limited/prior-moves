"""Portfolio constructor — turn the PriorScore Top Picks into a sized,
risk-controlled allocation you can actually execute.

Given a capital amount and risk settings, this:
  1. filters the ranked list (min liquidity tier, has a tradeable price),
  2. weights candidates by PriorScore,
  3. caps any single position (water-fills the excess to the rest),
  4. caps any single sector (scales it down, remainder to cash),
  5. converts target weights to whole-share orders at the latest close,
  6. reports leftover cash.

Two presets are tuned in ``PRESETS`` — an aggressive small book ($15k) and a
conservative large book ($200k, tighter caps + deeper-liquidity gate, because
you can't push $20k into a thin name without moving it).

This is a *recommendation* tool. It is not advice and places no orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import _load_prices

TIER_RANK = {"mega": 4, "large": 3, "mid": 2, "thin": 1, "unknown": 0}


@dataclass(frozen=True)
class RiskProfile:
    capital: float
    n_positions: int = 15
    max_position: float = 0.12      # cap per name (fraction of invested)
    max_sector: float = 0.35        # cap per sector
    cash_buffer: float = 0.05       # keep this fraction in cash
    min_tier: str = "mid"           # drop names thinner than this
    leverage: float = 1.0           # gross exposure multiple (1.0 = unlevered)
    min_price: float = 0.0          # drop names below this close (penny-stock gate)
    equal_weight: bool = False      # True = equal-weight (matches the validated
                                    # backtest, which is np.mean of top-N). False =
                                    # score-weighted then capped.
    label: str = "custom"


PRESETS: dict[str, RiskProfile] = {
    # Your $15k — can be more concentrated; mid-caps allowed.
    "aggressive_15k": RiskProfile(
        capital=15_000, n_positions=10, max_position=0.16, max_sector=0.45,
        cash_buffer=0.03, min_tier="mid", label="Aggressive · $15k",
    ),
    # Parent's $200k — capital preservation first: more names, tighter caps,
    # only deep-liquidity names so fills don't move the stock.
    "conservative_200k": RiskProfile(
        capital=200_000, n_positions=20, max_position=0.08, max_sector=0.25,
        cash_buffer=0.08, min_tier="large", label="Conservative · $200k",
    ),
    # Full IBKR paper buying power (~$1M). Diversified (30 names), deep-liquidity
    # only, modestly levered (1.25x). Rationale: the model's edge is real but thin
    # (t≈2), so leverage is kept low and tested on paper only — never cranked.
    # min_price=$5 gates out penny stocks (13F filers don't hold them → no signal,
    # and spreads/manipulation are prohibitive); min_tier="large" keeps fills from
    # moving a name at ~$60k/position. No options/futures — equities only.
    "paper_1m": RiskProfile(
        capital=1_000_000, n_positions=30, max_position=0.06, max_sector=0.30,
        cash_buffer=0.05, min_tier="large", leverage=1.25, min_price=5.0,
        label="Paper · $1M (1.25x liquid)",
    ),
    # PROFIT-MAX: concentrate to the *validated* config (the backtest measures
    # top-15: net +2.0 pts/q vs SPY, t≈1.5, CI includes zero, win 61%) and lever
    # it 1.5x. 1.5x is
    # the responsible ceiling for a long-only book: a -15% quarter -> -22.5%
    # equity, still well above the ~30% Reg-T maintenance line (2x would risk a
    # forced liquidation at the bottom). Margin drag ~0.7 pt/q is priced in.
    "paper_1m_max": RiskProfile(
        capital=1_000_000, n_positions=15, max_position=0.10, max_sector=0.40,
        cash_buffer=0.02, min_tier="large", leverage=1.5, min_price=5.0,
        label="Paper · $1M MAX (top-15, 1.5x)",
    ),
    # VALIDATED MATCH: equal-weight, UNLEVERED top-15 — exactly what the backtest
    # measures (np.mean of the top-15, no leverage). Use this for the PUBLIC paper
    # proof so the live track record actually validates the published +2.0pts/q
    # edge. (paper_1m_max trades a different, unvalidated strategy: score-weighted,
    # sector-capped, 1.5x levered — its track record does NOT prove the backtest.)
    "paper_validated": RiskProfile(
        capital=1_000_000, n_positions=15, max_position=1.0, max_sector=1.0,
        cash_buffer=0.0, min_tier="large", leverage=1.0, min_price=5.0,
        equal_weight=True, label="Paper · validated (equal-weight, unlevered top-15)",
    ),
}


def latest_price(ticker: str) -> float:
    df = _load_prices(ticker)
    if df is None or df.empty:
        return float("nan")
    return float(df["close"].astype(float).iloc[-1])


def _waterfill_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap each weight at ``cap``, redistributing the excess to under-cap names
    proportionally, iterating until stable. Weights stay summed to their input
    total (1.0)."""
    w = weights.clip(lower=0).astype(float).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(200):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap - 1e-12
        pool = float(w[under].sum())
        if pool <= 0:
            break
        w[under] = w[under] + excess * (w[under] / pool)
    return w


def _apply_sector_cap(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Scale any sector whose summed weight exceeds ``cap`` down to ``cap``.
    Freed weight becomes cash (NOT redistributed) so both the position cap and
    the sector cap are guaranteed to hold — the conservative choice."""
    df = df.copy()
    if "sector" not in df.columns:
        return df
    sec = df.groupby("sector")["weight"].transform("sum")
    over = sec > cap + 1e-9
    if over.any():
        df.loc[over, "weight"] = df.loc[over, "weight"] * (cap / sec[over])
    return df


def build_portfolio(agg: pd.DataFrame, profile: RiskProfile) -> dict:
    """Return a sized allocation + summary for one risk profile."""
    df = agg.copy()
    min_rank = TIER_RANK.get(profile.min_tier, 0)
    df = df[df["liquidity_tier"].map(lambda t: TIER_RANK.get(str(t), 0)) >= min_rank]

    # price: prefer the last_close baked into the aggregated parquet (works on
    # the deployed image, which has no prices_cache); fall back to the cache.
    if "last_close" in df.columns:
        df["price"] = pd.to_numeric(df["last_close"], errors="coerce")
        miss = df["price"].isna() | (df["price"] <= 0)
        if miss.any():
            df.loc[miss, "price"] = df.loc[miss, "ticker"].apply(latest_price)
    else:
        df["price"] = df["ticker"].apply(latest_price)
    df = df[df["price"].notna() & (df["price"] > 0)]
    # penny-stock gate: 13F filers don't hold sub-$5 names, so there is no signal
    # there, and spreads/manipulation make them un-executable at size.
    if profile.min_price > 0:
        df = df[df["price"] >= profile.min_price]
    if df.empty:
        return {"profile": profile, "holdings": pd.DataFrame(), "summary": {}}

    df = df.sort_values("prior_score", ascending=False).head(profile.n_positions).copy()

    # weight: equal-weight (matches the validated equal-weight backtest) OR
    # score-weighted then position-capped (water-fill). Sector cap applied after.
    if profile.equal_weight:
        df["weight"] = 1.0 / len(df)
    else:
        df["weight"] = _waterfill_cap(df["prior_score"], profile.max_position)
    df = _apply_sector_cap(df, profile.max_sector)

    # leverage multiplies gross exposure (1.0 = unlevered). Caps below scale with
    # it so the greedy fill can deploy the full levered target.
    investable = profile.capital * (1.0 - profile.cash_buffer) * profile.leverage
    df["target_$"] = df["weight"] * investable
    df["shares"] = np.floor(df["target_$"] / df["price"]).astype(int)

    # greedy leftover-cash deploy: buy one more share of the highest-PriorScore
    # name that still fits the remaining cash AND keeps its position + sector
    # caps. Floors the idle cash left by share-rounding without breaching caps.
    df = df.sort_values("prior_score", ascending=False).reset_index(drop=True)
    cap_base = profile.capital * profile.leverage
    pos_cap_usd = profile.max_position * cap_base
    sec_cap_usd = profile.max_sector * cap_base
    for _ in range(2000):
        spent = float((df["shares"] * df["price"]).sum())
        remaining = investable - spent
        sec_spend = (df.assign(c=df["shares"] * df["price"])
                     .groupby("sector")["c"].sum().to_dict()) if "sector" in df.columns else {}
        bought = False
        for i in df.index:
            price = float(df.at[i, "price"])
            cur = df.at[i, "shares"] * price
            sec = df.at[i, "sector"] if "sector" in df.columns else "_"
            if (price <= remaining
                    and cur + price <= pos_cap_usd + 1e-6
                    and sec_spend.get(sec, 0.0) + price <= sec_cap_usd + 1e-6):
                df.at[i, "shares"] += 1
                bought = True
                break
        if not bought:
            break

    df = df[df["shares"] > 0].copy()
    df["cost_$"] = df["shares"] * df["price"]
    df["weight_actual"] = df["cost_$"] / (profile.capital * profile.leverage)

    deployed = float(df["cost_$"].sum())
    summary = {
        "label": profile.label,
        "capital": profile.capital,
        "leverage": profile.leverage,
        "n_holdings": int(len(df)),
        "deployed_$": deployed,
        "gross_exposure_pct": float(deployed / profile.capital) if profile.capital else 0.0,
        "cash_$": float(investable - deployed),
        "cash_pct": float((investable - deployed) / profile.capital) if profile.capital else 0.0,
        "min_price": profile.min_price,
        "max_position": profile.max_position,
        "max_sector": profile.max_sector,
        "min_tier": profile.min_tier,
    }
    cols = ["ticker", "name", "sector", "prior_score", "liquidity_tier",
            "price", "shares", "cost_$", "weight_actual"]
    holdings = df[[c for c in cols if c in df.columns]].reset_index(drop=True)
    return {"profile": profile, "holdings": holdings, "summary": summary}


def to_ibkr_basket(holdings: pd.DataFrame) -> str:
    """IBKR Basket Trader CSV (Action,Quantity,Symbol,SecType,Exchange,Currency,
    OrderType). Import via TWS → Trade → BasketTrader → Load. Market-on-open,
    SMART routing, USD US equities."""
    lines = ["Action,Quantity,Symbol,SecType,Exchange,Currency,TimeInForce,OrderType"]
    for _, r in holdings.iterrows():
        lines.append(
            f"BUY,{int(r['shares'])},{r['ticker']},STK,SMART,USD,DAY,MKT"
        )
    return "\n".join(lines) + "\n"


def to_plain_csv(holdings: pd.DataFrame) -> str:
    d = holdings.copy()
    if "weight_actual" in d.columns:
        d["weight_%"] = (d["weight_actual"] * 100).round(1)
        d = d.drop(columns=["weight_actual"])
    for c in ("price", "cost_$"):
        if c in d.columns:
            d[c] = d[c].round(2)
    return d.to_csv(index=False)
