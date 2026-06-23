"""Online / continual-learning impact model — the Zoral-thesis surface.

The batch model (impact_model.py) fits once on the whole window and freezes. This
model never freezes: its weights update as each new filing/event arrives, in
strict streaming order, with no retraining from scratch. Two ideas made literal,
with a leak-free benchmark attached:

  Fusi (multi-timescale memory): a SLOW store (forgetting factor lambda≈0.999 —
  stable, the long-run weights the backtest trusts) and a FAST store (lambda≈0.95
  — plastic, tracks the current regime). Prediction blends the two.

  Friston (surprise-weighted learning): when the as-of regime is far from what the
  recent window predicted (high surprise z), lean on the fast/plastic store; when
  the world is calm, trust the slow store. Surprise gates the blend.

Math: recursive least squares with forgetting (closed-form, numpy only). Per step
(x, y):
    g = P x / (lambda + xᵀ P x)
    w = w + g (y - xᵀ w)
    P = (P - g xᵀ P) / lambda
P is the running inverse-covariance, seeded to (1/ridge) I.

This is deliberately a FIRST cut. The open problems are left for Aryaa:
  - catastrophic-forgetting controls (the slow store still drifts; no explicit
    consolidation / replay).
  - an optimal lambda schedule (fixed two-timescale here; the real Fusi cascade is
    a chain of stores with geometrically spaced timescales).
  - surprise estimation (here a simple online z on the macro block; Friston's
    free-energy formulation is richer).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OnlineRidge:
    """Recursive least squares with a forgetting factor. One plastic linear map."""

    dim: int
    lam: float = 0.99          # forgetting factor (1.0 = never forget)
    ridge: float = 10.0        # prior precision (seeds P = (1/ridge) I)
    p_trace_cap: float = 1e6   # covariance-windup bound (see update_one)
    w: np.ndarray = field(default=None)
    P: np.ndarray = field(default=None)
    n_seen: int = 0

    def __post_init__(self):
        if self.w is None:
            self.w = np.zeros(self.dim)
        if self.P is None:
            self.P = np.eye(self.dim) / self.ridge

    def predict_one(self, x: np.ndarray) -> float:
        return float(x @ self.w)

    def update_one(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float)
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        if denom <= 0 or not math.isfinite(denom):
            return
        g = Px / denom
        err = float(y) - float(x @ self.w)
        self.w = self.w + g * err
        self.P = (self.P - np.outer(g, Px)) / self.lam
        # symmetrize (guards numerical drift) and bound the covariance windup
        self.P = 0.5 * (self.P + self.P.T)
        tr = float(np.trace(self.P))
        if not math.isfinite(tr) or tr > self.p_trace_cap:
            self.P = self.P * (self.p_trace_cap / tr) if tr > 0 else np.eye(self.dim) / self.ridge
        self.n_seen += 1


@dataclass
class MultiTimescaleOnline:
    """Slow + fast RLS stores with a surprise-gated blend (Fusi × Friston)."""

    dim: int
    lam_slow: float = 0.999
    lam_fast: float = 0.95
    ridge: float = 10.0
    surprise_gain: float = 1.0   # how hard surprise shifts the blend toward fast
    slow: OnlineRidge = field(default=None)
    fast: OnlineRidge = field(default=None)

    def __post_init__(self):
        if self.slow is None:
            self.slow = OnlineRidge(self.dim, lam=self.lam_slow, ridge=self.ridge)
        if self.fast is None:
            self.fast = OnlineRidge(self.dim, lam=self.lam_fast, ridge=self.ridge)

    def _gate(self, surprise: float) -> float:
        """Blend weight on the FAST store. Calm regime → ~0 (trust slow);
        surprising regime → →1 (trust plastic). Logistic in surprise z."""
        s = max(0.0, float(surprise)) * self.surprise_gain
        return 1.0 / (1.0 + math.exp(-(s - 1.0)))  # ~0.27 at s=0, →1 as s grows

    def predict_one(self, x: np.ndarray, surprise: float = 0.0) -> float:
        x = np.asarray(x, dtype=float)
        g = self._gate(surprise)
        return (1.0 - g) * self.slow.predict_one(x) + g * self.fast.predict_one(x)

    def update_one(self, x: np.ndarray, y: float, surprise: float = 0.0) -> None:
        # Both stores always learn; they differ only in how fast they forget.
        # Surprise gates the PREDICT blend, not the RLS recursion (keeps the
        # closed form clean). An obvious extension for Aryaa: also modulate
        # lam_fast by surprise so the plastic store forgets faster under stress.
        self.slow.update_one(x, y)
        self.fast.update_one(x, y)

    # --- persistence ---
    def to_dict(self) -> dict:
        return {
            "dim": self.dim, "lam_slow": self.lam_slow, "lam_fast": self.lam_fast,
            "ridge": self.ridge, "surprise_gain": self.surprise_gain,
            "slow_w": self.slow.w.tolist(), "fast_w": self.fast.w.tolist(),
            "n_seen": self.slow.n_seen,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "MultiTimescaleOnline":
        m = cls(dim=d["dim"], lam_slow=d["lam_slow"], lam_fast=d["lam_fast"],
                ridge=d["ridge"], surprise_gain=d.get("surprise_gain", 1.0))
        m.slow.w = np.asarray(d["slow_w"], dtype=float)
        m.fast.w = np.asarray(d["fast_w"], dtype=float)
        m.slow.n_seen = m.fast.n_seen = int(d.get("n_seen", 0))
        return m


class OnlineSurprise:
    """Running mean/std of a scalar regime feature → an online z-score (the
    Friston surprise signal). Welford-style, no stored history."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> float:
        if not math.isfinite(x):
            return 0.0
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return self.current(x)

    def current(self, x: float) -> float:
        if self.n < 5:
            return 0.0
        std = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0
        if std <= 0:
            return 0.0
        return abs(float(x) - self.mean) / std
