"""Shared evaluation gates for both trading tracks (ROADMAP.md S1).

One honest measuring stick for any candidate signal, long-hold (Track A) or
short-horizon (Track B). Pure functions on numpy arrays / pandas frames — no
data loading, no repo-state coupling — so a backtest, the event study, the
weight search, and a future intraday study all score themselves the same way.

This consolidates math that was duplicated and subtly inconsistent across
scripts/stress_test.py (IID quarter bootstrap), scripts/backtest_event_signal.py
(IID event bootstrap), scripts/leakage_canary.py (hard-coded shuffle threshold),
and scripts/optimize_2h.py (selection haircut). Two corrections land here:

  * Returns series are serially correlated, so a plain IID bootstrap understates
    the CI width. `block_bootstrap_ci` uses a circular block bootstrap.
  * "Beats default" means nothing without deflating for how many configs were
    tried. `selection_haircut` / `deflated_sharpe` price the multiple-testing
    inflation explicitly.

Nothing here touches scoring.py, the published consensus, or any frozen state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# normal helpers (no scipy dependency — keep the gate library import-light)
# --------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 on (0, 1); used for the expected-max-of-N selection
    benchmark. Clamps the open interval to avoid +/-inf at the boundary.
    """
    if not (0.0 < p < 1.0):
        p = min(max(p, 1e-15), 1.0 - 1e-15)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


# --------------------------------------------------------------------------
# basic return statistics
# --------------------------------------------------------------------------

def tstat(x: np.ndarray) -> float:
    """One-sample t-statistic of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / (sd / math.sqrt(len(x))))


def sharpe(x: np.ndarray, periods_per_year: int = 4) -> float:
    """Annualized Sharpe of a per-period excess-return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def max_drawdown(curve: np.ndarray) -> float:
    """Worst peak-to-trough on an equity curve (negative number)."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) == 0:
        return float("nan")
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


# --------------------------------------------------------------------------
# block bootstrap CI — serial-correlation-aware
# --------------------------------------------------------------------------

def block_bootstrap_ci(x: np.ndarray, block: int = 4, n_boot: int = 20000,
                       alpha: float = 0.05, seed: int = 12345) -> tuple[float, float]:
    """Circular-block-bootstrap CI for the mean of a serially-correlated series.

    Resampling whole blocks of length `block` preserves short-range
    autocorrelation that an IID bootstrap destroys (which is why the IID
    versions in stress_test.py / backtest_event_signal.py give too-narrow CIs
    on quarterly returns). block=1 reduces to the IID bootstrap.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    nobs = len(x)
    if nobs < 2:
        return (float("nan"), float("nan"))
    block = max(1, min(block, nobs))
    n_blocks = int(math.ceil(nobs / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, nobs, size=(n_boot, n_blocks))
    # each block is `block` consecutive elements from a random start, wrapping
    # circularly (the % nobs) so blocks near the end don't truncate.
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % nobs
    samples = x[idx].reshape(n_boot, n_blocks * block)[:, :nobs]
    means = samples.mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# --------------------------------------------------------------------------
# label-shuffle leakage canary — generic permutation p-value
# --------------------------------------------------------------------------

def shuffle_pvalue(observed: float, shuffled: np.ndarray,
                   higher_is_better: bool = True) -> float:
    """Empirical p-value of `observed` against a null of shuffled-label scores.

    Generalizes leakage_canary.py's hard 0.55 threshold into a proper
    permutation p-value: the fraction of shuffled runs that match-or-beat the
    real score. A clean (leak-free) pipeline yields a LARGE p-value (the real
    score is indistinguishable from noise on shuffled labels means the *metric*
    can't be gamed by leakage); a small p-value on shuffled labels means the
    pipeline scores well even on noise -> leakage. Returns p in [0, 1].
    """
    shuffled = np.asarray(shuffled, dtype=float)
    shuffled = shuffled[~np.isnan(shuffled)]
    if len(shuffled) == 0:
        return float("nan")
    if higher_is_better:
        hits = int((shuffled >= observed).sum())
    else:
        hits = int((shuffled <= observed).sum())
    return (hits + 1) / (len(shuffled) + 1)  # +1 smoothing, never exactly 0


def leakage_flag(real_score: float, shuffled_scores: np.ndarray,
                 noise_level: float = 0.5, abs_margin: float = 0.05) -> bool:
    """True if the shuffled-label score sits materially above the noise floor.

    For AUC, noise_level=0.5 and abs_margin=0.05 reproduces the canary's
    "shuffled AUC > 0.55 = leak" rule, but driven off the MEAN of several
    shuffles (robust to one bad draw), matching leakage_canary.py's intent.
    """
    s = np.asarray(shuffled_scores, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return False
    return bool(s.mean() > noise_level + abs_margin)


# --------------------------------------------------------------------------
# transaction-cost haircut
# --------------------------------------------------------------------------

def cost_haircut(gross_returns: np.ndarray, turnover: np.ndarray | float,
                 cost_bps: float) -> np.ndarray:
    """Net per-period returns after a per-period two-sided cost.

    cost charged each period = turnover * cost_bps / 1e4, where turnover is the
    fraction of the book traded (1.0 = full replacement). Short-horizon books
    are spread-dominated, so this is where a daily strategy usually dies — the
    gate makes that explicit instead of reporting gross.
    """
    gross = np.asarray(gross_returns, dtype=float)
    turn = np.broadcast_to(np.asarray(turnover, dtype=float), gross.shape)
    return gross - turn * (cost_bps / 1e4)


# --------------------------------------------------------------------------
# cross-sectional information coefficient (signal quality)
# --------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1-D array (ties shared), for Spearman without scipy.

    np.unique returns sorted values + an inverse map; each value's average rank
    is the midpoint of its run, so ties get the shared mean rank.
    """
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def spearman_ic(scores: np.ndarray, fwd_returns: np.ndarray) -> float:
    """Spearman rank correlation between a cross-section of scores and the
    forward returns they're meant to predict. The core signal-quality metric
    shared by both tracks (one period's cross-section)."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(fwd_returns, dtype=float)
    m = ~(np.isnan(s) | np.isnan(r))
    s, r = s[m], r[m]
    if len(s) < 3:
        return float("nan")
    rs, rr = _rank(s), _rank(r)
    if rs.std() == 0 or rr.std() == 0:
        return float("nan")
    return float(np.corrcoef(rs, rr)[0, 1])


# --------------------------------------------------------------------------
# overfit / multiple-testing deflation
# --------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, n_obs: int, var_trials: float = 1.0) -> float:
    """Expected MAX in-sample Sharpe from `n_trials` independent strategies that
    all have TRUE Sharpe = 0, given `n_obs` observations.

    This is the selection benchmark from Bailey & Lopez de Prado's Deflated
    Sharpe Ratio: search enough random configs and the best one looks good by
    luck alone. A candidate must clear THIS, not zero.
    """
    if n_trials < 2 or n_obs < 2:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = math.e
    # SR estimate standard error ~ 1/sqrt(n_obs) when true SR=0
    se = math.sqrt(var_trials) / math.sqrt(n_obs)
    z = ((1 - gamma) * norm_ppf(1 - 1.0 / n_trials)
         + gamma * norm_ppf(1 - 1.0 / (n_trials * e)))
    return se * z


def deflated_sharpe(observed_sr: float, n_trials: int, n_obs: int) -> float:
    """Probability the observed (per-obs, non-annualized) Sharpe beats what
    pure selection luck over `n_trials` would produce — i.e. P(real edge).

    ~1.0 = the result clears the multiple-testing bar; ~0.5 or below = it is in
    the range you'd get by trying this many configs on noise. Honest go/no-go
    for the weight search and any leaderboard-selected signal.

    Simplification vs the full Bailey-Lopez de Prado DSR: this uses the normal
    SR standard error 1/sqrt(n_obs) and omits the skew/kurtosis adjustment, so
    on heavy-tailed return series it is mildly optimistic. Good enough as a gate
    that errs toward demanding MORE evidence than zero, not as a p-value to cite.
    """
    if n_obs < 2:
        return float("nan")
    bench = expected_max_sharpe(n_trials, n_obs)
    se = 1.0 / math.sqrt(n_obs)
    return norm_cdf((observed_sr - bench) / se)


def selection_haircut(in_sample_sr: float, n_trials: int, n_obs: int) -> float:
    """The Sharpe you should EXPECT out-of-sample after subtracting the
    selection inflation: max(0, observed - expected-max-under-null)."""
    return max(0.0, in_sample_sr - expected_max_sharpe(n_trials, n_obs))


# --------------------------------------------------------------------------
# one-call summary
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GateReport:
    n: int
    mean: float
    t: float
    sharpe_ann: float
    ci95: tuple[float, float]
    win_rate: float
    max_dd: float

    def verdict(self) -> str:
        if self.n < 2:
            return "INSUFFICIENT DATA"
        if self.ci95[0] > 0:
            return "SIGNAL: 95% CI above zero"
        if self.mean > 0:
            return "DIRECTIONAL, NOT SIGNIFICANT: positive mean, CI includes zero"
        return "NO EDGE: mean <= 0"


def summarize(excess_returns: np.ndarray, periods_per_year: int = 4,
              block: int = 4, seed: int = 12345) -> GateReport:
    """Roll the per-period excess-return series into one honest report."""
    x = np.asarray(excess_returns, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return GateReport(0, float("nan"), float("nan"), float("nan"),
                          (float("nan"), float("nan")), float("nan"), float("nan"))
    curve = np.cumprod(1.0 + x)
    return GateReport(
        n=n,
        mean=float(x.mean()),
        t=tstat(x),
        sharpe_ann=sharpe(x, periods_per_year),
        ci95=block_bootstrap_ci(x, block=block, seed=seed),
        win_rate=float((x > 0).mean()),
        max_dd=max_drawdown(curve),
    )
