"""Iterative compare loop — my picks vs investor's actual picks.

For each (investor, quarter) in held-out:
  1. Load model's holdout predictions for that investor
  2. Rank candidate (ticker/cusip) by p_new_entry descending
  3. Take top-K picks
  4. Compare to actual new_entry set for that investor that quarter
  5. Emit: precision@K, recall@K, overlap, hits list, misses list, false-positives list

The user's mental model: "at Q-end I would have picked these N stocks; at
Q+45d 13F drops and I see which ones investor X actually entered". This file
makes that diff machine-readable.

Output per (investor, quarter):
  runs/per_investor/<slug>/picks_<YYYY-MM-DD>.parquet
  with columns rank, cusip, ticker, name_of_issuer, p_new_entry, actual_label

Plus an aggregate summary at runs/per_investor/SUMMARY_picks.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]


# Default top-K per investor. Quant funds with 100+ new entries per quarter
# get larger K so precision/recall is meaningful; concentrated value funds
# get smaller K so noise doesn't dilute.
TIER_K = {
    # Concentrated value / event-driven
    "ackman": 5, "buffett": 10, "burry": 10, "klarman": 10, "tepper": 10,
    "loeb": 10, "pabrai": 5, "einhorn": 10,
    # Macro
    "soros": 25, "druckenmiller": 25,
    # Quant / systematic
    "greenblatt": 50, "renaissance": 80, "two_sigma": 80,
}


@dataclass(frozen=True)
class QuarterPicksSummary:
    investor: str
    quarter: pd.Timestamp
    n_candidates: int
    n_actual_new_entries: int
    top_k: int
    precision_at_k: float
    recall_at_k: float
    overlap_at_k: int
    hit_tickers: list[str]
    miss_tickers: list[str]
    false_positive_tickers: list[str]


def compare_quarter(holdout_df: pd.DataFrame, investor: str, quarter: pd.Timestamp,
                    top_k: int = 10) -> QuarterPicksSummary:
    sub = holdout_df[holdout_df["period_of_report"] == quarter].copy()
    if sub.empty:
        return QuarterPicksSummary(
            investor, quarter, 0, 0, top_k, 0.0, 0.0, 0, [], [], [],
        )
    sub = sub.sort_values("p", ascending=False).reset_index(drop=True)
    picks = sub.head(top_k)

    actual_pos = sub[sub["y"] == 1]
    actual_set = set(actual_pos["cusip"]) if "cusip" in actual_pos.columns else set()
    pick_set = set(picks["cusip"]) if "cusip" in picks.columns else set()

    hits = pick_set & actual_set
    misses = actual_set - pick_set
    fps = pick_set - actual_set

    n_actual = len(actual_set)
    precision = (len(hits) / top_k) if top_k > 0 else 0.0
    recall = (len(hits) / n_actual) if n_actual > 0 else 0.0

    # Convert cusip → ticker / name where possible for human readability
    cusip_to_ticker = sub.set_index("cusip")["ticker"].to_dict() if "ticker" in sub.columns else {}
    cusip_to_name = sub.set_index("cusip")["name_of_issuer"].to_dict() if "name_of_issuer" in sub.columns else {}

    def _label(cusip: str) -> str:
        t = cusip_to_ticker.get(cusip)
        n = cusip_to_name.get(cusip)
        if t:
            return f"{t} ({(n or '')[:30]})"
        return f"cusip={cusip} ({(n or '')[:30]})"

    return QuarterPicksSummary(
        investor=investor,
        quarter=quarter,
        n_candidates=int(len(sub)),
        n_actual_new_entries=int(n_actual),
        top_k=top_k,
        precision_at_k=float(precision),
        recall_at_k=float(recall),
        overlap_at_k=int(len(hits)),
        hit_tickers=sorted(_label(c) for c in hits),
        miss_tickers=sorted(_label(c) for c in misses),
        false_positive_tickers=sorted(_label(c) for c in fps),
    )


def compare_investor(repo: Path, investor: str, top_k: int = 10
                     ) -> tuple[list[QuarterPicksSummary], pd.DataFrame]:
    slug_dir = repo / "runs" / "per_investor" / investor
    preds_path = slug_dir / "holdout_predictions.parquet"
    if not preds_path.exists():
        return [], pd.DataFrame()
    df = pd.read_parquet(preds_path)
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])

    # Re-join CUSIP→ticker + name from labels if missing
    if "ticker" not in df.columns or "name_of_issuer" not in df.columns:
        labels_path = repo / "data" / "13f" / "labels.parquet"
        if labels_path.exists():
            full = pd.read_parquet(labels_path)
            full["period_of_report"] = pd.to_datetime(full["period_of_report"])
            name_lookup = full.drop_duplicates(subset=["cusip"]).set_index("cusip")["name_of_issuer"]
            df["name_of_issuer"] = df["cusip"].map(name_lookup)
        cusip_map_path = repo / "data" / "tickers" / "cusip_to_ticker.parquet"
        if cusip_map_path.exists():
            cm = pd.read_parquet(cusip_map_path)[["cusip", "ticker"]].dropna()
            df = df.merge(cm, on="cusip", how="left")

    summaries: list[QuarterPicksSummary] = []
    for q in sorted(df["period_of_report"].unique()):
        s = compare_quarter(df, investor, pd.Timestamp(q), top_k=top_k)
        summaries.append(s)
        # Persist per-quarter picks file
        sub = df[df["period_of_report"] == q].sort_values("p", ascending=False).reset_index(drop=True)
        sub = sub.assign(rank=range(1, len(sub) + 1))
        cols = ["rank", "cusip", "ticker", "name_of_issuer", "p", "y", "label"]
        keep = [c for c in cols if c in sub.columns]
        out_path = slug_dir / f"picks_{pd.Timestamp(q).date().isoformat()}.parquet"
        sub[keep].to_parquet(out_path, index=False)
    overview = pd.DataFrame([{
        "investor": s.investor,
        "quarter": s.quarter,
        "n_candidates": s.n_candidates,
        "n_actual_new_entries": s.n_actual_new_entries,
        "top_k": s.top_k,
        "precision_at_k": s.precision_at_k,
        "recall_at_k": s.recall_at_k,
        "overlap_at_k": s.overlap_at_k,
    } for s in summaries])
    return summaries, overview


def compare_all(repo: Path = REPO, top_k: int = 10,
                use_tier_k: bool = True) -> dict:
    out_root = repo / "runs" / "per_investor"
    if not out_root.exists():
        raise FileNotFoundError(f"{out_root} missing; run scripts/train_per_investor.py first")

    all_overviews = []
    detail: dict[str, list[dict]] = {}
    for slug_dir in sorted(out_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        investor = slug_dir.name
        # Use per-investor K when requested (quant funds need K=50+, value K=10)
        k = TIER_K.get(investor, top_k) if use_tier_k else top_k
        summaries, overview = compare_investor(repo, investor, top_k=k)
        if overview.empty:
            continue
        all_overviews.append(overview)
        detail[investor] = [asdict(s) for s in summaries]

    if not all_overviews:
        return {"overview": pd.DataFrame(), "detail": detail}

    summary_df = pd.concat(all_overviews, ignore_index=True)
    summary_df.to_parquet(out_root / "picks_summary.parquet", index=False)

    # Markdown overview
    md_lines = ["# Iterative compare — top-K picks vs actual new_entries\n"]
    md_lines.append(f"top_k = {top_k}\n")
    md_lines.append("| investor | quarter | n_cand | n_actual | overlap@K | precision@K | recall@K |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for _, r in summary_df.sort_values(["investor", "quarter"]).iterrows():
        md_lines.append(
            f"| {r['investor']} | {pd.Timestamp(r['quarter']).date()} | "
            f"{int(r['n_candidates'])} | {int(r['n_actual_new_entries'])} | "
            f"{int(r['overlap_at_k'])} | {r['precision_at_k']:.2f} | {r['recall_at_k']:.2f} |"
        )
    (out_root / "picks_summary.md").write_text("\n".join(md_lines) + "\n")
    return {"overview": summary_df, "detail": detail}
