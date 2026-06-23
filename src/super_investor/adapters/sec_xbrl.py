"""SEC XBRL companyfacts — real quarterly fundamentals for the whole universe.

Free, no API key (User-Agent required). One JSON per company holds every
XBRL concept it has ever reported:

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

We extract a compact set of fundamentals per (ticker, quarter_end) and emit
with the ``xf_`` prefix. This covers EVERY SEC filer — far broader than the
~200-ticker stockanalysis/finviz snapshots.

Concepts pulled (US-GAAP):
  Revenues / RevenueFromContractWithCustomerExcludingAssessedTax
  NetIncomeLoss
  Assets, Liabilities, StockholdersEquity
  EarningsPerShareDiluted
  OperatingIncomeLoss
  CashAndCashEquivalentsAtCarryingValue
  LongTermDebtNoncurrent
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("super_investor.adapters.sec_xbrl")

UA = "super-investor-mirror research@example.com"
BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# concept -> (output column, is_flow). Flow concepts (income statement) report
# over a period; we keep only ~single-quarter durations. Stock concepts
# (balance sheet) are point-in-time (only an 'end' date, no 'start').
CONCEPTS = {
    "Revenues": ("xf_revenue", True),
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("xf_revenue", True),
    "NetIncomeLoss": ("xf_net_income", True),
    "OperatingIncomeLoss": ("xf_op_income", True),
    "EarningsPerShareDiluted": ("xf_eps_diluted", True),
    "Assets": ("xf_assets", False),
    "Liabilities": ("xf_liabilities", False),
    "StockholdersEquity": ("xf_equity", False),
    "CashAndCashEquivalentsAtCarryingValue": ("xf_cash", False),
    "LongTermDebtNoncurrent": ("xf_lt_debt", False),
}


def _get_json(cik: str, cache_dir: Path | None = None) -> dict | None:
    if cik is None or str(cik).strip() in ("", "None"):
        return None
    try:
        cik10 = f"{int(cik):010d}"
    except (TypeError, ValueError):
        return None
    url = f"{BASE}/CIK{cik10}.json"
    cp = cache_dir / f"CIK{cik10}.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists() and cp.stat().st_size > 100:
        import json
        try:
            return json.loads(cp.read_text())
        except Exception:
            return None
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            return None
        if cp:
            cp.write_text(r.text)
        return r.json()
    except Exception as e:
        log.warning("_get_json(%s): %s: %s", cik, type(e).__name__, e)
        return None


def fetch_company_facts(ticker: str, cik: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Return per-(quarter_end) fundamentals for one company."""
    data = _get_json(cik, cache_dir=cache_dir)
    if not data:
        return pd.DataFrame()
    facts = (data.get("facts") or {}).get("us-gaap") or {}
    rows: dict[pd.Timestamp, dict] = {}
    for concept, (out_col, is_flow) in CONCEPTS.items():
        node = facts.get(concept)
        if not node:
            continue
        units = node.get("units") or {}
        series = units.get("USD") or units.get("USD/shares") or next(iter(units.values()), [])
        for item in series:
            end = item.get("end")
            val = item.get("val")
            if end is None or val is None:
                continue
            if is_flow:
                # keep only ~single-quarter durations (80-100 days). Drops
                # YTD (6/9mo) and annual (12mo) figures that share a quarter-end.
                start = item.get("start")
                if not start:
                    continue
                try:
                    dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
                except Exception:
                    continue
                if not (80 <= dur <= 100):
                    continue
            try:
                q_end = pd.Timestamp(end).to_period("Q").end_time.normalize()
            except Exception:
                continue
            rows.setdefault(q_end, {})
            if out_col not in rows[q_end] or rows[q_end][out_col] is None:
                rows[q_end][out_col] = float(val)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame([{"quarter_end": k, **v} for k, v in rows.items()])
    out["ticker"] = ticker.upper()
    # derived ratios
    if "xf_net_income" in out and "xf_revenue" in out:
        out["xf_net_margin"] = out["xf_net_income"] / out["xf_revenue"].replace(0, pd.NA)
    if "xf_lt_debt" in out and "xf_equity" in out:
        out["xf_debt_to_equity"] = out["xf_lt_debt"] / out["xf_equity"].replace(0, pd.NA)
    out["quarter_end"] = pd.to_datetime(out["quarter_end"], utc=True)
    return out


def aggregate(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # one row per (ticker, quarter); keep last (most complete)
    df = df.sort_values("quarter_end").drop_duplicates(["ticker", "quarter_end"], keep="last")
    return df
