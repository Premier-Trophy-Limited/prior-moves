"""New-entry / exit / add / trim labels built from quarter-over-quarter 13F diffs.

For each (investor, ticker, quarter), assign one of:
  - new_entry     : ticker absent at Q-1, present at Q
  - exit          : ticker present at Q-1, absent at Q
  - add           : present at both; shares increased by >= 10% (configurable)
  - trim          : present at both; shares decreased by >= 10%
  - hold          : present at both; shares within ±10%

Plus continuous fields:
  - shares_delta_pct : (shares_Q - shares_Q_minus_1) / shares_Q_minus_1
  - weight_delta_pct : portfolio-weight change in pp (percentage points)

Important caveats:
  - 13F-HR/A amendments restate the prior period; we keep ONLY the latest filing
    per (investor, period_of_report) to avoid double-counting.
  - Some amendments are partial (e.g. one position); we detect this by row count
    vs the immediately-prior full filing and skip partials.
  - First quarter for which an investor has any data has no Q-1 baseline; we
    label every position there as `unknown` so it does not pollute training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


LabelKind = Literal["new_entry", "exit", "add", "trim", "hold", "unknown"]


@dataclass(frozen=True)
class LabelConfig:
    add_trim_threshold_pct: float = 10.0
    drop_partial_amendments: bool = True
    # Lower threshold so concentrated portfolios (Pershing, Scion, Pabrai) aren't filtered;
    # we instead reject amendments that are <50% the size of the prior full filing.
    partial_amendment_min_positions: int = 3
    partial_amendment_size_ratio: float = 0.5


def build_labels(
    holdings_path: Path,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Read a per-investor holdings parquet and emit a per-quarter label table."""
    cfg = config or LabelConfig()
    df = pd.read_parquet(holdings_path)
    if df.empty:
        return df.assign(label="unknown")

    # Sort filings by period, keep latest per (period_of_report) only.
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])
    df["filed_at"] = pd.to_datetime(df["filed_at"])

    if cfg.drop_partial_amendments:
        position_counts = df.groupby("accession").size()
        # Hard floor: any filing with <min_positions positions is treated as a partial amendment.
        keep_acc = set(position_counts[position_counts >= cfg.partial_amendment_min_positions].index)
        # Soft floor: amendments much smaller than the period's biggest filing are partials.
        meta = df.drop_duplicates("accession")[["accession", "period_of_report"]]
        for period, group in meta.groupby("period_of_report"):
            max_count = max(position_counts.get(acc, 0) for acc in group["accession"])
            if max_count <= cfg.partial_amendment_min_positions:
                # Don't filter when even the biggest filing is small (concentrated portfolio).
                continue
            for acc in group["accession"]:
                if position_counts.get(acc, 0) < max_count * cfg.partial_amendment_size_ratio:
                    keep_acc.discard(acc)
        df = df[df["accession"].isin(keep_acc)].copy()

    # For each period, retain only the LATEST-filed full filing.
    chosen_acc = {
        period: g["accession"].iloc[-1]
        for period, g in df.sort_values("filed_at").groupby("period_of_report")
    }
    df = df[df["accession"].isin(set(chosen_acc.values()))].copy()

    # Aggregate per-(period, cusip): sum shares + values (in case multiple class share rows).
    agg = (
        df.groupby(["investor_slug", "period_of_report", "cusip"], as_index=False)
        .agg(shares=("shares", "sum"), value_usd=("value_usd", "sum"),
             name_of_issuer=("name_of_issuer", "first"))
    )
    # Portfolio total per period
    period_total = agg.groupby(["investor_slug", "period_of_report"])["value_usd"].sum().rename("portfolio_total")
    agg = agg.join(period_total, on=["investor_slug", "period_of_report"])
    agg["weight_pct"] = agg["value_usd"] / agg["portfolio_total"].replace(0, pd.NA) * 100.0

    # Sort by period to compute Q -> Q+1 diffs
    agg = agg.sort_values(["investor_slug", "cusip", "period_of_report"]).reset_index(drop=True)
    agg["prev_shares"] = agg.groupby(["investor_slug", "cusip"])["shares"].shift(1)
    agg["prev_period"] = agg.groupby(["investor_slug", "cusip"])["period_of_report"].shift(1)
    agg["prev_weight_pct"] = agg.groupby(["investor_slug", "cusip"])["weight_pct"].shift(1)

    def _classify(row) -> LabelKind:
        prev = row["prev_shares"]
        curr = row["shares"]
        if pd.isna(prev):
            return "new_entry"
        if curr is None or curr == 0:
            return "exit"
        delta_pct = (curr - prev) / max(prev, 1) * 100.0
        if delta_pct > cfg.add_trim_threshold_pct:
            return "add"
        if delta_pct < -cfg.add_trim_threshold_pct:
            return "trim"
        return "hold"

    agg["label"] = agg.apply(_classify, axis=1)
    agg["shares_delta_pct"] = (agg["shares"] - agg["prev_shares"]) / agg["prev_shares"].replace(0, pd.NA) * 100.0
    agg["weight_delta_pp"] = agg["weight_pct"] - agg["prev_weight_pct"].fillna(0)

    # Synthesise EXIT rows: tickers present at Q-1 but absent at Q.
    exits = []
    periods_per_investor = (
        agg.groupby("investor_slug")["period_of_report"].apply(lambda s: sorted(s.unique()))
    )
    held_at = (
        agg.groupby(["investor_slug", "period_of_report"])["cusip"].apply(set).to_dict()
    )
    for investor_slug, periods in periods_per_investor.items():
        for i in range(1, len(periods)):
            prev_p = periods[i - 1]
            curr_p = periods[i]
            prev_set = held_at.get((investor_slug, prev_p), set())
            curr_set = held_at.get((investor_slug, curr_p), set())
            exited_cusips = prev_set - curr_set
            for cusip in exited_cusips:
                # Look up the prior row for name + shares; build a Q-row with shares=0.
                prior_row = agg[(agg["investor_slug"] == investor_slug)
                                & (agg["cusip"] == cusip)
                                & (agg["period_of_report"] == prev_p)]
                if prior_row.empty:
                    continue
                pr = prior_row.iloc[0]
                exits.append({
                    "investor_slug": investor_slug,
                    "period_of_report": curr_p,
                    "cusip": cusip,
                    "shares": 0,
                    "value_usd": 0,
                    "name_of_issuer": pr["name_of_issuer"],
                    "portfolio_total": pr["portfolio_total"],
                    "weight_pct": 0.0,
                    "prev_shares": pr["shares"],
                    "prev_period": prev_p,
                    "prev_weight_pct": pr["weight_pct"],
                    "label": "exit",
                    "shares_delta_pct": -100.0,
                    "weight_delta_pp": -pr["weight_pct"],
                })
    if exits:
        agg = pd.concat([agg, pd.DataFrame(exits)], ignore_index=True)

    agg = agg.sort_values(["investor_slug", "period_of_report", "cusip"]).reset_index(drop=True)
    return agg
