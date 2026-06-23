"""D5 — learned name-level event-impact model.

Honest scope: a SMALL, regularized model (Ridge) over a handful of features —
event type + magnitude, whether a held name sits in the event's impact sectors,
its PriorScore conviction, and the as-of macro regime — predicting forward
excess vs SPY per name. Used to WEIGHT the event basket instead of equal-weight.

Small-n by construction (a handful of clean curated events). We keep the model
deliberately tiny, report leave-one-event-out CV honestly, and the backtest
always shows the equal-weight baseline beside the learned weights. This is the
concrete, leak-checkable instance of a continually-learning impact model — the
exact surface a future online version plugs into.

NEVER modifies scoring.py: PriorScore enters read-only as one feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVENT_TYPES = ("macro", "geopolitical", "thematic")
MACRO_FEATS = ("vix", "hy_oas", "real_10y", "unemployment", "core_cpi_yoy_pct")

FEATURE_NAMES = (
    ["magnitude", "in_impact_sector", "prior_score"]
    + [f"type_{t}" for t in EVENT_TYPES]
    + [f"macro_{m}" for m in MACRO_FEATS]
)

# --- Corporate-event (8-K / 13D) feature space (Tier 1 Track A, name-level) ---
# A separate, larger feature vector for the thousands of name-level corporate
# events. Each row = one (issuer, filing_date): item one-hots + activist flag +
# the issuer's own conviction + as-of macro regime → issuer forward excess.
CORP_ITEMS = ("2.02", "4.02", "5.02", "1.01", "2.01", "3.01",
              "7.01", "8.01", "1.02", "2.03", "2.05", "4.01", "5.07")

CORP_FEATURE_NAMES = (
    ["magnitude", "holder_breadth", "is_activist"]
    + [f"macro_{m}" for m in MACRO_FEATS]
    + [f"item_{it.replace('.', '_')}" for it in CORP_ITEMS]
)

# holder_breadth = how many of the 44 tracked investors held the name as-of the
# event (a cheap, leak-safe conviction proxy from labels.parquet — deliberately
# NOT the validated PriorScore, which is a different, frozen consensus model).
# LightGBM monotone hint: more of the greats holding it shouldn't hurt.
_CORP_MONOTONE = {"holder_breadth": 1}


def feature_row(event_type: str, magnitude, in_impact_sector: bool,
                prior_score, macro: dict | None) -> dict:
    """One feature dict for one (event, name) pair. PriorScore scaled to ~0-1;
    missing macro fields default to 0 (guarded)."""
    feats = {
        "magnitude": float(magnitude or 0),
        "in_impact_sector": 1.0 if in_impact_sector else 0.0,
        "prior_score": float(prior_score or 0.0) / 100.0,
    }
    for t in EVENT_TYPES:
        feats[f"type_{t}"] = 1.0 if event_type == t else 0.0
    macro = macro or {}
    for m in MACRO_FEATS:
        v = macro.get(m)
        feats[f"macro_{m}"] = float(v) if v is not None and v == v else 0.0
    return feats


def corp_feature_row(magnitude, holder_breadth, is_activist,
                     macro: dict | None, items_onehot: dict | None,
                     extra: dict | None = None) -> dict:
    """Feature dict for one corporate event (issuer, filing_date). holder_breadth
    is the count of tracked investors holding the name as-of the event (a leak-safe
    conviction proxy), scaled by /44 to ~0-1.

    `extra` carries arbitrary aggregated-channel features (ch_<chan>__<col>,
    has_<chan>, engineered interactions) joined leak-safely by channel_join.py. They
    are appended verbatim; fit(feature_names=...) is generic, so the feature space
    grows by whatever the iterate loop includes. The 21-feature baseline block
    (magnitude/breadth/activist/macro/items) is always present."""
    feats = {
        "magnitude": float(magnitude or 0),
        "holder_breadth": float(holder_breadth or 0.0) / 44.0,
        "is_activist": 1.0 if is_activist else 0.0,
    }
    macro = macro or {}
    for m in MACRO_FEATS:
        v = macro.get(m)
        feats[f"macro_{m}"] = float(v) if v is not None and v == v else 0.0
    items_onehot = items_onehot or {}
    for it in CORP_ITEMS:
        k = f"item_{it.replace('.', '_')}"
        feats[k] = float(items_onehot.get(k, 0.0))
    if extra:
        import math
        for k, v in extra.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            feats[k] = fv if math.isfinite(fv) else 0.0
    return feats


@dataclass
class ImpactModel:
    coef: list
    intercept: float
    feature_names: list
    n_events: int
    n_rows: int
    cv_r2: float | None
    y_mean: float
    model_kind: str = "ridge"
    booster_txt: str | None = None   # LightGBM model string when model_kind=="lightgbm"

    def _X(self, rows: list[dict]) -> np.ndarray:
        return np.array([[r.get(f, 0.0) for f in self.feature_names] for r in rows],
                        dtype=float)

    def predict(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.array([])
        if self.model_kind == "lightgbm" and self.booster_txt:
            import lightgbm as lgb
            booster = lgb.Booster(model_str=self.booster_txt)
            return booster.predict(self._X(rows))
        return self._X(rows) @ np.array(self.coef, dtype=float) + self.intercept

    def weights(self, tickers: list, rows: list[dict]) -> dict:
        """Predicted forward excess → long-only basket weights. Negative
        predictions clip to 0 (we never short the basket); normalize. If every
        prediction is <=0, fall back to equal-weight (honest no-signal default)."""
        if not tickers:
            return {}
        pred = self.predict(rows)
        pos = np.clip(pred, 0.0, None)
        if pos.size == 0 or pos.sum() <= 0:
            w = np.ones(len(tickers)) / len(tickers)
        else:
            w = pos / pos.sum()
        return {str(t): float(wi) for t, wi in zip(tickers, w)}

    def to_dict(self) -> dict:
        return {
            "coef": list(self.coef), "intercept": self.intercept,
            "feature_names": list(self.feature_names), "n_events": self.n_events,
            "n_rows": self.n_rows, "cv_r2": self.cv_r2, "y_mean": self.y_mean,
            "model_kind": self.model_kind, "booster_txt": self.booster_txt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImpactModel":
        return cls(coef=d["coef"], intercept=d["intercept"],
                   feature_names=d["feature_names"], n_events=d["n_events"],
                   n_rows=d["n_rows"], cv_r2=d.get("cv_r2"), y_mean=d["y_mean"],
                   model_kind=d.get("model_kind", "ridge"),
                   booster_txt=d.get("booster_txt"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ImpactModel | None":
        path = Path(path)
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))


def _grouped_cv_r2(estimator_factory, X, yv, groups):
    """Honest out-of-sample R² grouped by event (or year). LeaveOneGroupOut for
    a handful of groups, GroupKFold(5) when there are many."""
    n_groups = len(set(groups))
    if n_groups < 3:
        return None, n_groups
    try:
        from sklearn.model_selection import (
            GroupKFold, LeaveOneGroupOut, cross_val_score)
        cv = LeaveOneGroupOut() if n_groups <= 12 else GroupKFold(n_splits=5)
        scores = cross_val_score(estimator_factory(), X, yv,
                                 groups=np.array(groups), cv=cv, scoring="r2")
        return float(np.mean(scores)), n_groups
    except Exception:
        return None, n_groups


def fit(rows: list[dict], y: list[float], groups: list[str],
        alpha: float = 1.0, feature_names=None, model_kind: str = "ridge",
        monotone: dict | None = None) -> ImpactModel:
    """Fit the impact model on (rows, y). `groups` drive honest grouped CV (can
    the model predict a group — event or year — it never saw?).

    model_kind="ridge"    → linear Ridge (the honest baseline).
    model_kind="lightgbm" → gradient-boosted trees with optional monotone
                            constraints; used once n supports it.
    """
    feature_names = list(feature_names or FEATURE_NAMES)
    X = np.array([[r.get(f, 0.0) for f in feature_names] for r in rows], dtype=float)
    yv = np.array(y, dtype=float)

    if model_kind == "lightgbm":
        import lightgbm as lgb
        mono = monotone or {}
        mono_vec = [int(mono.get(f, 0)) for f in feature_names]
        params = dict(objective="regression", n_estimators=300, learning_rate=0.03,
                      num_leaves=31, min_child_samples=40, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0,
                      monotone_constraints=mono_vec, verbosity=-1)

        def _factory():
            return lgb.LGBMRegressor(**params)

        cv_r2, n_groups = _grouped_cv_r2(_factory, X, yv, groups)
        booster = _factory().fit(X, yv)
        return ImpactModel(
            coef=[0.0] * len(feature_names), intercept=float(yv.mean()),
            feature_names=feature_names, n_events=n_groups, n_rows=len(rows),
            cv_r2=cv_r2, y_mean=float(yv.mean()), model_kind="lightgbm",
            booster_txt=booster.booster_.model_to_string())

    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(X, yv)
    cv_r2, n_groups = _grouped_cv_r2(lambda: Ridge(alpha=alpha), X, yv, groups)
    return ImpactModel(coef=m.coef_.tolist(), intercept=float(m.intercept_),
                       feature_names=feature_names, n_events=n_groups,
                       n_rows=len(rows), cv_r2=cv_r2, y_mean=float(yv.mean()),
                       model_kind="ridge")
