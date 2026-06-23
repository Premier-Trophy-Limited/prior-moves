"""LightGBM baseline classifier: P(investor enters ticker T at quarter Q | macro, investor_id, cusip).

This is the first cut, using only what's already on disk (no Finnhub backfill,
no text features). The point is to establish:

1. Does macro context alone predict super-investor new entries above the base rate?
2. Which investor profiles separate easiest (per-investor AUC)?
3. What's the calibration of P(new_entry) — fitted directly vs needing Platt?

The training set is the existing labels.parquet rows (~324k). Positive class =
`label == "new_entry"` (~25k, 7.8%). Negative class = the other 4 labels.

Features used:
    investor_slug (categorical, 12 levels)
    cusip (categorical, ~23k levels — hashed to int)
    period_of_report → year, quarter_index
    macro features joined from FRED snapshot

Label-leak guards: prev_shares, prev_period, prev_weight_pct, shares_delta_pct,
weight_delta_pp, value_usd, shares ALL excluded — they trivially encode the
label (new_entry => prev_shares is NaN by construction).

Time split: train on quarters ≤ 2024-06-30, holdout on 2024-09-30 forward.
That's ~8 quarters of holdout (Q3 2024 → Q1 2026).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


REPO = Path(__file__).resolve().parents[3]
HOLDOUT_START_DEFAULT = pd.Timestamp("2024-09-30")


@dataclass
class BaselineResult:
    n_train: int
    n_val: int
    n_holdout: int
    n_features: int
    feature_importance: dict[str, float]
    val_auc: float
    val_pr_auc: float
    val_brier: float
    holdout_auc: float
    holdout_pr_auc: float
    holdout_brier: float
    per_investor_holdout: dict[str, dict]


def _load_labels(repo: Path = REPO) -> pd.DataFrame:
    df = pd.read_parquet(repo / "data" / "13f" / "labels.parquet")
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])
    return df


def _load_macro(repo: Path = REPO) -> pd.DataFrame:
    df = pd.read_parquet(repo / "data" / "features" / "macro_quarterly.parquet")
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    return df


def build_dataset(repo: Path = REPO, holdout_start: pd.Timestamp = HOLDOUT_START_DEFAULT
                  ) -> tuple[pd.DataFrame, list[str], list[str]]:
    labels = _load_labels(repo)
    macro = _load_macro(repo).set_index("quarter_end")

    df = labels.copy()
    df["y"] = (df["label"] == "new_entry").astype(np.int8)
    df["year"] = df["period_of_report"].dt.year
    df["q_index"] = df["period_of_report"].dt.quarter
    df["cusip_id"] = df["cusip"].map({c: i for i, c in enumerate(sorted(df["cusip"].unique()))})
    df["investor_id"] = df["investor_slug"].map(
        {s: i for i, s in enumerate(sorted(df["investor_slug"].unique()))})

    # Join macro at period_of_report (broadcast to all rows in that quarter)
    df = df.join(macro, on="period_of_report", how="left")

    feature_cols = ["investor_id", "cusip_id", "year", "q_index"] + list(macro.columns)
    categorical_cols = ["investor_id", "cusip_id", "year", "q_index"]

    keep_cols = feature_cols + ["y", "period_of_report", "investor_slug", "cusip", "label"]
    return df[keep_cols].copy(), feature_cols, categorical_cols


def train_baseline(repo: Path = REPO,
                   holdout_start: pd.Timestamp = HOLDOUT_START_DEFAULT,
                   val_quarters: int = 4,
                   params: dict | None = None) -> BaselineResult:
    df, feature_cols, cat_cols = build_dataset(repo, holdout_start)

    # Temporal split: train = (oldest, holdout_start - val_quarters); val = last
    # val_quarters before holdout_start; holdout = >= holdout_start.
    val_start = holdout_start - pd.DateOffset(months=3 * val_quarters)
    train = df[df["period_of_report"] < val_start].copy()
    val = df[(df["period_of_report"] >= val_start) & (df["period_of_report"] < holdout_start)].copy()
    holdout = df[df["period_of_report"] >= holdout_start].copy()

    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "verbose": -1,
            "max_bin": 255,
        }

    dtrain = lgb.Dataset(train[feature_cols], label=train["y"], categorical_feature=cat_cols)
    dval = lgb.Dataset(val[feature_cols], label=val["y"], categorical_feature=cat_cols, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    p_val = booster.predict(val[feature_cols])
    p_holdout = booster.predict(holdout[feature_cols])

    val_metrics = _metrics(val["y"].to_numpy(), p_val)
    holdout_metrics = _metrics(holdout["y"].to_numpy(), p_holdout)

    # Per-investor holdout breakdown
    holdout = holdout.assign(p=p_holdout)
    per_inv: dict[str, dict] = {}
    for inv, sub in holdout.groupby("investor_slug"):
        if sub["y"].sum() >= 5 and sub["y"].nunique() == 2:
            per_inv[inv] = {
                "n": int(len(sub)),
                "n_positive": int(sub["y"].sum()),
                "auc": float(roc_auc_score(sub["y"], sub["p"])),
                "pr_auc": float(average_precision_score(sub["y"], sub["p"])),
                "brier": float(brier_score_loss(sub["y"], sub["p"])),
            }

    importance = booster.feature_importance(importance_type="gain")
    fi = dict(sorted(
        ((f, float(v)) for f, v in zip(feature_cols, importance)),
        key=lambda x: -x[1]
    ))

    return BaselineResult(
        n_train=len(train), n_val=len(val), n_holdout=len(holdout),
        n_features=len(feature_cols), feature_importance=fi,
        val_auc=val_metrics["auc"], val_pr_auc=val_metrics["pr_auc"], val_brier=val_metrics["brier"],
        holdout_auc=holdout_metrics["auc"], holdout_pr_auc=holdout_metrics["pr_auc"],
        holdout_brier=holdout_metrics["brier"],
        per_investor_holdout=per_inv,
    )


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    if y_true.sum() < 1 or y_true.sum() == len(y_true):
        return {"auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }
