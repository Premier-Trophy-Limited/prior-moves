"""yfinance historical adapter — backfill multi-year quarterly features.

The existing ``yfinance_news`` adapter is snapshot-only: it captures the
*current* news + price-target window, which only ever populates the latest
quarter. For training across many historical quarters we need data that
yfinance actually exposes in time-series form:

* ``Ticker.upgrades_downgrades`` — analyst rating change events (full history)
* ``Ticker.quarterly_financials`` — revenue + earnings + tax rate, per quarter
* ``Ticker.quarterly_balance_sheet`` — assets + debt + equity, per quarter
* ``Ticker.quarterly_cashflow`` — operating + free cash flow, per quarter
* ``Ticker.dividends`` — full dividend history (daily series)

Output: per (ticker, quarter_end) row aggregating the four panels above into
numeric features prefixed ``yh_`` so the model can join cleanly. Idempotent
merge by (ticker, quarter_end) so backfill is restart-safe.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YfinanceQuarterRow:
    ticker: str
    quarter_end: pd.Timestamp
    # Analyst ratings
    n_upgrades_q: int
    n_downgrades_q: int
    n_initiations_q: int
    # Income statement
    revenue: float
    operating_income: float
    net_income: float
    eps_basic: float
    # Balance sheet
    total_assets: float
    total_debt: float
    stockholders_equity: float
    cash_and_equivalents: float
    # Cash flow
    operating_cash_flow: float
    free_cash_flow: float
    # Dividends
    dividends_paid_q: float


def _coerce_float(v) -> float:
    try:
        if v is None or pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Snap a timestamp to the calendar-quarter end."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.to_period("Q").end_time.normalize()


def _column_pick(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first matching row from a yfinance quarterly statement.

    yfinance's quarterly_financials uses fiscal-period columns and row labels
    that vary by company (e.g. "Total Revenue" vs "Operating Revenue"). We
    walk the candidate names and return the first hit, NaN-filling if none
    match.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            return df.loc[n]
    # Case-insensitive fallback
    lc = {str(idx).lower(): idx for idx in df.index}
    for n in names:
        if n.lower() in lc:
            return df.loc[lc[n.lower()]]
    return pd.Series(dtype=float)


class YfinanceHistoricalClient:
    """Cache-backed yfinance historical pull at per-ticker granularity."""

    def __init__(self, cache_dir: Path | None = None, throttle_sec: float = 0.4):
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._throttle = throttle_sec

    def _cache_path(self, ticker: str) -> Path | None:
        if not self._cache_dir:
            return None
        # yfinance accepts BRK.B / BRK-B / TECK/B etc., but the filesystem
        # interprets "/" as a path separator. Normalize for cache only.
        safe = ticker.upper().replace("/", "-").replace(" ", "_")
        return self._cache_dir / f"{safe}.json"

    def _fetch_raw(self, ticker: str) -> dict:
        """Hit yfinance once per ticker, capture every quarterly panel."""
        import yfinance as yf
        t = yf.Ticker(ticker)
        out: dict = {}
        # upgrades_downgrades — full history
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                ud = ud.reset_index()
                # The DatetimeIndex becomes "GradeDate"; normalize to ISO string
                date_col = ud.columns[0]
                ud[date_col] = pd.to_datetime(ud[date_col], errors="coerce")
                out["upgrades_downgrades"] = [
                    {
                        "date": (
                            r[date_col].isoformat()
                            if pd.notna(r[date_col])
                            else None
                        ),
                        "action": str(r.get("Action", "")),
                        "from_grade": str(r.get("FromGrade", "")),
                        "to_grade": str(r.get("ToGrade", "")),
                        "firm": str(r.get("Firm", "")),
                    }
                    for _, r in ud.iterrows()
                ]
        except Exception as e:
            out["_ud_err"] = str(e)
        # Quarterly statements — yfinance returns DataFrame with PeriodIndex
        for attr in ("quarterly_financials", "quarterly_balance_sheet", "quarterly_cashflow"):
            try:
                df = getattr(t, attr, None)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df.columns = [
                    pd.Timestamp(c).isoformat() if pd.notna(c) else "" for c in df.columns
                ]
                # Convert NaN to None for json
                df = df.where(pd.notna(df), None)
                out[attr] = df.to_dict(orient="index")
            except Exception as e:
                out[f"_{attr}_err"] = str(e)
        # Dividends
        try:
            div = t.dividends
            if div is not None and not div.empty:
                div.index = pd.to_datetime(div.index)
                if div.index.tz is not None:
                    div.index = div.index.tz_localize(None)
                out["dividends"] = {
                    d.isoformat(): float(v) for d, v in div.items()
                }
        except Exception as e:
            out["_div_err"] = str(e)
        return out

    def pull_ticker(self, ticker: str) -> dict:
        """Return raw yfinance data for one ticker, cached."""
        cache_path = self._cache_path(ticker)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())
        time.sleep(self._throttle)
        data = self._fetch_raw(ticker)
        if cache_path:
            cache_path.write_text(json.dumps(data, default=str))
        return data


def aggregate_ticker_to_quarters(ticker: str, raw: dict) -> list[YfinanceQuarterRow]:
    """Roll one ticker's raw yfinance payload up to per-quarter rows."""
    # Index quarterly statements first; collect the set of quarter_ends covered
    quarters: dict[pd.Timestamp, dict[str, float]] = {}

    def _attach(name: str, mapping: dict, statement: dict | None):
        if not statement:
            return
        # statement is { row_name -> { col_iso -> value } }
        # We want { quarter_end -> { field_name -> value } } using mapping row→field
        for row_name, col_map in statement.items():
            if not isinstance(col_map, dict):
                continue
            target_field = mapping.get(row_name.lower())
            if not target_field:
                continue
            for col_iso, val in col_map.items():
                try:
                    qe = _quarter_end(pd.Timestamp(col_iso))
                except Exception:
                    continue
                if qe not in quarters:
                    quarters[qe] = {}
                quarters[qe][target_field] = _coerce_float(val)

    # Income statement field mapping
    income_map = {
        "total revenue": "revenue",
        "operating revenue": "revenue",
        "operating income": "operating_income",
        "operating income or loss": "operating_income",
        "net income": "net_income",
        "net income from continuing operations": "net_income",
        "basic eps": "eps_basic",
        "diluted eps": "eps_basic",  # fall back
    }
    _attach("financials", income_map, raw.get("quarterly_financials"))

    balance_map = {
        "total assets": "total_assets",
        "total debt": "total_debt",
        "stockholders equity": "stockholders_equity",
        "total equity gross minority interest": "stockholders_equity",
        "cash and cash equivalents": "cash_and_equivalents",
        "cash cash equivalents and short term investments": "cash_and_equivalents",
    }
    _attach("balance", balance_map, raw.get("quarterly_balance_sheet"))

    cashflow_map = {
        "operating cash flow": "operating_cash_flow",
        "cash flow from continuing operating activities": "operating_cash_flow",
        "free cash flow": "free_cash_flow",
    }
    _attach("cashflow", cashflow_map, raw.get("quarterly_cashflow"))

    # Upgrades / downgrades per quarter
    ud_counts: dict[pd.Timestamp, dict[str, int]] = {}
    for ev in raw.get("upgrades_downgrades", []) or []:
        d = ev.get("date")
        if not d:
            continue
        try:
            qe = _quarter_end(pd.Timestamp(d))
        except Exception:
            continue
        action = (ev.get("action") or "").lower()
        bucket = ud_counts.setdefault(qe, {"up": 0, "down": 0, "init": 0})
        if "up" in action:
            bucket["up"] += 1
        elif "down" in action:
            bucket["down"] += 1
        elif "init" in action or "main" in action:
            bucket["init"] += 1

    # Dividends per quarter (sum)
    div_q: dict[pd.Timestamp, float] = {}
    for d_iso, val in (raw.get("dividends") or {}).items():
        try:
            qe = _quarter_end(pd.Timestamp(d_iso))
        except Exception:
            continue
        div_q[qe] = div_q.get(qe, 0.0) + _coerce_float(val)

    all_quarters = set(quarters) | set(ud_counts) | set(div_q)
    rows: list[YfinanceQuarterRow] = []
    for qe in sorted(all_quarters):
        f = quarters.get(qe, {})
        u = ud_counts.get(qe, {"up": 0, "down": 0, "init": 0})
        rows.append(YfinanceQuarterRow(
            ticker=ticker.upper(),
            quarter_end=qe,
            n_upgrades_q=int(u["up"]),
            n_downgrades_q=int(u["down"]),
            n_initiations_q=int(u["init"]),
            revenue=f.get("revenue", float("nan")),
            operating_income=f.get("operating_income", float("nan")),
            net_income=f.get("net_income", float("nan")),
            eps_basic=f.get("eps_basic", float("nan")),
            total_assets=f.get("total_assets", float("nan")),
            total_debt=f.get("total_debt", float("nan")),
            stockholders_equity=f.get("stockholders_equity", float("nan")),
            cash_and_equivalents=f.get("cash_and_equivalents", float("nan")),
            operating_cash_flow=f.get("operating_cash_flow", float("nan")),
            free_cash_flow=f.get("free_cash_flow", float("nan")),
            dividends_paid_q=div_q.get(qe, 0.0),
        ))
    return rows


def rows_to_dataframe(rows: list[YfinanceQuarterRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Numeric features prefixed yh_ to avoid collision with finnhub fh_ / form4 f4_
    rename_map = {c: f"yh_{c}" for c in df.columns if c not in ("ticker", "quarter_end")}
    df = df.rename(columns=rename_map)
    return df
