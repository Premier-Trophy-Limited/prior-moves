"""Deterministic, no-look-ahead 13F backtest harness.

The headline credibility claim — "+2.0 pts/quarter net vs the S&P 500 over
35 quarters (directional; CI includes zero)" — is only trustworthy if
quarter Q's portfolio is built ONLY
from filings whose PUBLIC-availability date (``filed_at``) is on or before
Q's rebalance date. 13F has a ~45-day reporting lag, so filtering on the
period-end date (``period_of_report``) instead silently leaks the future and
inflates the edge.

This module makes "no look-ahead" a TESTED INVARIANT, not a hope:

  * ``assert_no_lookahead`` — a hard guard. Every constituent filing's public
    date must be <= the rebalance date, else ``LookAheadError``.
  * ``run_backtest`` — a pure function. Given holdings, rebalance dates, and an
    injected ``price_fn``, it filters filings by public-availability date,
    selects the portfolio, and measures return vs SPY. No wall-clock, no
    randomness, no network — identical inputs yield an identical result.

It complements ``scripts/leakage_canary.py`` (a label-shuffle check on the
MODEL) by guarding the BACKTEST's date filter, which the canary never touches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


REPO = Path(__file__).resolve().parents[2]


# price_fn(ticker, rebalance_date, hold_quarters) -> hold-period return (float).
# Injected so tests stay offline and the real run reads the price cache.
PriceFn = Callable[[str, pd.Timestamp, int], float]

# select_fn(public_holdings, rebalance_date) -> list of tickers to hold.
SelectFn = Callable[[pd.DataFrame, pd.Timestamp], "list[str]"]


class LookAheadError(AssertionError):
    """A filing dated after the rebalance leaked into the portfolio."""


@dataclass(frozen=True)
class Rebalance:
    date: pd.Timestamp
    tickers: tuple[str, ...]
    port_return: float
    spy_return: float
    edge: float
    n_rejected: int  # filings excluded because not yet public at `date`


@dataclass(frozen=True)
class BacktestResult:
    rebalances: tuple[Rebalance, ...]
    quarters: int
    avg_edge: float
    max_drawdown: float


def assert_no_lookahead(
    holdings: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    date_field: str = "filed_at",
) -> None:
    """Hard guard: every row's public date must be <= the rebalance date.

    Raises ``LookAheadError`` naming the offenders. This is the invariant the
    whole harness exists to protect — call it on the post-filter set so a
    selection bug can never smuggle an unpublished filing into the portfolio.
    """
    if holdings.empty:
        return
    dates = pd.to_datetime(holdings[date_field])
    future = holdings[dates > rebalance_date]
    if not future.empty:
        sample = (
            future[["investor_slug", "period_of_report", date_field]]
            .head(5)
            .to_dict("records")
        )
        raise LookAheadError(
            f"{len(future)} filing(s) have {date_field} > rebalance "
            f"{rebalance_date.date()} — look-ahead leak. e.g. {sample}"
        )


def consensus_new_entries(public: pd.DataFrame, rebalance_date: pd.Timestamp) -> list[str]:
    """Default strategy = the existing 'new buys' rule, no-look-ahead-safe.

    For each investor, take their most recent PUBLIC filing (latest
    period_of_report among rows already filtered to filed_at <= R), keep the
    ``new_entry`` rows, and equal-weight the de-duplicated ticker set. This is
    the product's "what the best funds are buying next" basket, built only
    from what was knowable at the rebalance.
    """
    if public.empty:
        return []
    latest_period = public.groupby("investor_slug")["period_of_report"].transform("max")
    latest = public[public["period_of_report"] == latest_period]
    new = latest[latest["label"] == "new_entry"]
    tickers = [t for t in new["ticker"].dropna().tolist() if isinstance(t, str) and t]
    return sorted(set(tickers))


def make_top_k_select(top_k: int = 10) -> SelectFn:
    """Model top-K strategy as a select_fn — the picks behind the headline edge.

    At rebalance R the harness has already filtered ``holdings`` to filings
    public at R (``filed_at <= R``). For each investor we take their most
    recent PUBLIC filing (max period_of_report among the survivors), rank its
    candidate rows by the model score ``p`` descending, and keep the top-K.
    Equal-weight the union across investors. No date logic here — the
    no-look-ahead filter lives entirely in ``run_backtest``.
    """
    def _select(public: pd.DataFrame, rebalance_date: pd.Timestamp) -> list[str]:
        if public.empty or "p" not in public.columns:
            return []
        latest_period = public.groupby("investor_slug")["period_of_report"].transform("max")
        latest = public[public["period_of_report"] == latest_period]
        picks: list[str] = []
        for _, g in latest.groupby("investor_slug"):
            top = g.sort_values("p", ascending=False, kind="stable").head(top_k)
            picks.extend(
                t for t in top["ticker"].tolist() if isinstance(t, str) and t
            )
        return sorted(set(picks))

    return _select


def _max_drawdown(cum: list[float]) -> float:
    """Largest peak-to-trough drop on the cumulative-edge equity curve."""
    peak = float("-inf")
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def run_backtest(
    holdings: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    price_fn: PriceFn,
    select_fn: SelectFn = consensus_new_entries,
    hold_quarters: int = 1,
    date_field: str = "filed_at",
    spy_ticker: str = "SPY",
) -> BacktestResult:
    """Run a no-look-ahead backtest and return the ledger.

    For each rebalance date R (processed in sorted order):
      1. FILTER to filings public at R: ``holdings[date_field] <= R``.
      2. GUARD: ``assert_no_lookahead`` on the survivors (belt and braces).
      3. SELECT the portfolio via ``select_fn`` (default: consensus new buys).
      4. MEASURE equal-weight hold return vs SPY -> per-quarter edge.

    ``date_field`` defaults to ``filed_at`` (the public date). Pointing it at
    ``period_of_report`` deliberately re-introduces the leak — used only to
    demonstrate that the filter is load-bearing.
    """
    holdings = holdings.copy()
    holdings[date_field] = pd.to_datetime(holdings[date_field])

    rebalances: list[Rebalance] = []
    for R in sorted(pd.Timestamp(d) for d in rebalance_dates):
        public = holdings[holdings[date_field] <= R]
        n_rejected = len(holdings) - len(public)

        # Surface the dangerous rejects: filings for a quarter that has already
        # ENDED but is not yet public. These are exactly the rows a naive
        # period-keyed filter would have leaked.
        if "period_of_report" in holdings.columns:
            too_recent = holdings[
                (pd.to_datetime(holdings["period_of_report"]) <= R)
                & (holdings[date_field] > R)
            ]
            if not too_recent.empty:
                periods = sorted(
                    pd.to_datetime(too_recent["period_of_report"]).dt.date.unique()
                )
                print(
                    f"[no-look-ahead] rebalance {R.date()}: REJECTED "
                    f"{len(too_recent)} filing(s) period<=R but not yet public "
                    f"(periods {periods}) — would leak the future"
                )

        # Guard the post-filter set: nothing past R may survive.
        assert_no_lookahead(public, R, date_field=date_field)

        tickers = select_fn(public, R)
        rets = [price_fn(t, R, hold_quarters) for t in tickers]
        rets = [r for r in rets if r == r]  # drop NaN (no price)
        port = sum(rets) / len(rets) if rets else float("nan")
        spy = price_fn(spy_ticker, R, hold_quarters)
        edge = port - spy if (port == port and spy == spy) else float("nan")

        rebalances.append(Rebalance(
            date=R,
            tickers=tuple(tickers),
            port_return=port,
            spy_return=spy,
            edge=edge,
            n_rejected=n_rejected,
        ))

    edges = [r.edge for r in rebalances if r.edge == r.edge]
    avg_edge = sum(edges) / len(edges) if edges else float("nan")
    cum, run = [], 0.0
    for e in edges:
        run += e
        cum.append(run)
    mdd = _max_drawdown(cum) if cum else 0.0

    return BacktestResult(
        rebalances=tuple(rebalances),
        quarters=len(rebalances),
        avg_edge=avg_edge,
        max_drawdown=mdd,
    )


# --------------------------------------------------------------------------- #
# Real-data driver. Lives in-module so the change set stays harness-only.
# Reads the on-disk price cache (no network when cached). Joins the public
# `filed_at` date onto the labels — the join the leak-free backtest requires
# and that the model's `labels.parquet` (period-keyed only) lacks.
# --------------------------------------------------------------------------- #

_REPORTING_LAG_DAYS = 45  # statutory 13F deadline after quarter-end


def _load_real_holdings(repo: Path) -> pd.DataFrame:
    """labels + public filed_at + ticker, keyed for a no-look-ahead run."""
    labels = pd.read_parquet(repo / "data" / "13f" / "labels.parquet")
    labels["period_of_report"] = pd.to_datetime(labels["period_of_report"])

    # filed_at is constant per (investor, period); pull it from the per-investor
    # holdings parquets and attach it to the period-keyed labels.
    frames = []
    for p in sorted((repo / "data" / "13f").glob("*.parquet")):
        if p.name in ("labels.parquet", "index.parquet"):
            continue
        df = pd.read_parquet(p, columns=["investor_slug", "period_of_report", "filed_at"])
        frames.append(df)
    filed = pd.concat(frames, ignore_index=True)
    filed["filed_at"] = pd.to_datetime(filed["filed_at"])
    # 13F-HR/A amendments are filed AFTER the original; holdings parquets are
    # written oldest-first. Keep the LATEST filing per (investor, period) so an
    # amended quarter uses the amendment's (later) filed_at — preserving the
    # no-look-ahead guarantee instead of inheriting the superseded original date.
    filed = filed.sort_values("filed_at").drop_duplicates(
        ["investor_slug", "period_of_report"], keep="last"
    )
    filed["period_of_report"] = pd.to_datetime(filed["period_of_report"])

    out = labels.merge(filed, on=["investor_slug", "period_of_report"], how="left")
    # Fall back to the statutory deadline when a filed_at is missing.
    fallback = out["period_of_report"] + pd.Timedelta(days=_REPORTING_LAG_DAYS)
    out["filed_at"] = out["filed_at"].fillna(fallback)

    cm_path = repo / "data" / "tickers" / "cusip_to_ticker.parquet"
    cm = pd.read_parquet(cm_path)[["cusip", "ticker"]].dropna()
    out = out.drop(columns=["ticker"], errors="ignore").merge(cm, on="cusip", how="left")
    return out


def _filed_at_map(repo: Path) -> pd.DataFrame:
    """(investor_slug, period_of_report) -> filed_at, from the holdings parquets."""
    frames = []
    for p in sorted((repo / "data" / "13f").glob("*.parquet")):
        if p.name in ("labels.parquet", "index.parquet"):
            continue
        df = pd.read_parquet(p, columns=["investor_slug", "period_of_report", "filed_at"])
        frames.append(df)
    filed = pd.concat(frames, ignore_index=True)
    filed["filed_at"] = pd.to_datetime(filed["filed_at"])
    # 13F-HR/A amendments are filed AFTER the original; holdings parquets are
    # written oldest-first. Keep the LATEST filing per (investor, period) so an
    # amended quarter uses the amendment's (later) filed_at — preserving the
    # no-look-ahead guarantee instead of inheriting the superseded original date.
    filed = filed.sort_values("filed_at").drop_duplicates(
        ["investor_slug", "period_of_report"], keep="last"
    )
    filed["period_of_report"] = pd.to_datetime(filed["period_of_report"])
    return filed


def load_model_predictions(
    repo: Path,
    source_subdir: str = "per_investor_wf",
) -> pd.DataFrame:
    """Read existing walk-forward LightGBM predictions — NO retrain, NO network.

    Returns one row per (investor, period, cusip) candidate with the model
    score ``p`` and the public ``filed_at`` date attached (the join the
    period-keyed predictions lack). ``filed_at`` powers the no-look-ahead
    trade-timing filter in ``run_backtest``; the model's own features stay
    period-aligned exactly as trained.
    """
    root = repo / "runs" / source_subdir
    if not root.exists():
        raise FileNotFoundError(f"{root} missing; train per-investor models first")

    frames = []
    for slug_dir in sorted(root.iterdir()):
        preds = slug_dir / "holdout_predictions.parquet"
        if not slug_dir.is_dir() or not preds.exists():
            continue
        df = pd.read_parquet(preds)
        df["investor_slug"] = slug_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no holdout_predictions.parquet under {root}")
    out = pd.concat(frames, ignore_index=True)
    out["period_of_report"] = pd.to_datetime(out["period_of_report"])

    filed = _filed_at_map(repo)
    out = out.merge(filed, on=["investor_slug", "period_of_report"], how="left")
    fallback = out["period_of_report"] + pd.Timedelta(days=_REPORTING_LAG_DAYS)
    out["filed_at"] = out["filed_at"].fillna(fallback)
    return out


def _real_price_fn(repo: Path) -> PriceFn:
    from super_investor.adapters.prices import quarterly_return
    cache_dir = repo / "data" / "prices_cache"

    def _fn(ticker: str, asof: pd.Timestamp, hold_quarters: int) -> float:
        return quarterly_return(ticker, asof, cache_dir, hold_quarters=hold_quarters)

    return _fn


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quarters", type=int, default=23, help="most recent N quarters")
    ap.add_argument("--hold-quarters", type=int, default=1)
    ap.add_argument(
        "--strategy", choices=("consensus", "topk"), default="consensus",
        help="consensus new-buys, or model top-K (the headline picks)",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--source-subdir", default="per_investor_wf",
        help="runs/<subdir>/ holding the walk-forward predictions (topk only)",
    )
    ap.add_argument(
        "--date-field", default="filed_at",
        help="public date field; pass 'period_of_report' to DEMONSTRATE the leak",
    )
    args = ap.parse_args()

    if args.strategy == "topk":
        holdings = load_model_predictions(REPO, source_subdir=args.source_subdir)
        select_fn = make_top_k_select(args.top_k)
        label = f"top-K={args.top_k} ({args.source_subdir})"
    else:
        holdings = _load_real_holdings(REPO)
        select_fn = consensus_new_entries
        label = "consensus new-buys"

    periods = sorted(holdings["period_of_report"].dropna().unique())
    # Rebalance at each quarter-end. Same dates for both runs so the ONLY thing
    # that differs honest-vs-leaky is the date field that admits a filing.
    rebalance_dates = [pd.Timestamp(p) for p in periods][-args.quarters:]

    res = run_backtest(
        holdings,
        rebalance_dates=rebalance_dates,
        price_fn=_real_price_fn(REPO),
        select_fn=select_fn,
        hold_quarters=args.hold_quarters,
        date_field=args.date_field,
    )

    edges = [r.edge for r in res.rebalances if r.edge == r.edge]
    print(f"\n=== Leak-proof 13F backtest [{label}] (date_field={args.date_field}) ===")
    print(f"quarters       : {res.quarters} ({len(edges)} with price data)")
    print(f"avg edge vs SPY: {res.avg_edge:+.4f}  ({res.avg_edge * 100:+.2f} pts/quarter)")
    print(f"max drawdown   : {res.max_drawdown:.4f}  (cumulative-edge curve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
