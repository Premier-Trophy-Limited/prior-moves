"""SEC EDGAR 8-K adapter — material-event filings → per-(ticker, quarter) counts.

8-K = "current report" — public companies must file within 4 business days of
any material event. Items signal the event class:

  1.01  Entry into a Material Definitive Agreement (M&A, partnership)
  1.02  Termination of a Material Definitive Agreement
  2.01  Completion of Acquisition or Disposition
  2.02  Results of Operations and Financial Condition (earnings)
  2.03  Creation of a Direct Financial Obligation (debt)
  2.05  Costs Associated with Exit or Disposal Activities
  3.01  Notice of Delisting
  4.01  Changes in Registrant's Certifying Accountant
  4.02  Non-Reliance on Previously Issued Financial Statements (restatement!)
  5.02  Departure / Election of Directors or Officers
  5.07  Submission of Matters to a Vote of Security Holders
  7.01  Regulation FD Disclosure
  8.01  Other Events (catch-all, often guidance updates)
  9.01  Financial Statements and Exhibits (NOISE — attached docs)

Per (ticker, quarter_end) we aggregate counts of each meaningful item code
plus a total. Output prefix ``e8k_`` so the per-investor model can join cleanly
alongside the existing ``fh_`` / ``f4_`` / ``yh_`` blocks.

Uses the same SEC submissions JSON the Form 4 adapter already caches — no
new network burden when the issuer has been touched before.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_MIN_REQUEST_GAP_S = 0.11
_USER_AGENT_DEFAULT = "super-investor-mirror researcher@example.com"

# Item codes worth tracking (skip 9.01 noise + administrative items)
_TRACKED_ITEMS = (
    "1.01",
    "1.02",
    "2.01",
    "2.02",
    "2.03",
    "2.05",
    "3.01",
    "4.01",
    "4.02",
    "5.02",
    "5.07",
    "7.01",
    "8.01",
)


class Edgar8KClient:
    """Polite EDGAR client — list 8-K filings + counts items per quarter."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None):
        ua = user_agent or os.environ.get("SEC_USER_AGENT") or _USER_AGENT_DEFAULT
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_t = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - gap)
        self._last_t = time.monotonic()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)
        ),
    )
    def _get(self, url: str) -> bytes:
        self._throttle()
        with httpx.Client(timeout=20.0, headers=self._headers, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.content

    def _submissions(self, cik: str) -> dict | None:
        cik_padded = str(cik).lstrip("0").zfill(10)
        cache_path = (
            self._cache_dir / "submissions" / f"CIK{cik_padded}.json"
            if self._cache_dir
            else None
        )
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_bytes())
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        try:
            raw = self._get(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw)
        return json.loads(raw)

    def pull_ticker_8k(self, ticker: str, cik: str) -> pd.DataFrame:
        """Return one DataFrame of (ticker, filing_date, items[]) for issuer."""
        payload = self._submissions(cik)
        if not payload:
            return pd.DataFrame()
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filings: list[dict] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            items_raw = (recent.get("items", []) + [""] * len(forms))[i]
            try:
                fd = pd.Timestamp(recent["filingDate"][i])
            except Exception:
                continue
            filings.append({
                "ticker": ticker.upper(),
                "issuer_cik": cik,
                "filing_date": fd,
                "items": [x.strip() for x in str(items_raw).split(",") if x.strip()],
            })
        return pd.DataFrame(filings)


def aggregate_8k_to_quarters(filings_df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-filing 8-K rows up to per-(ticker, quarter_end) counts."""
    if filings_df is None or filings_df.empty:
        return pd.DataFrame()
    df = filings_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["quarter_end"] = df["filing_date"].dt.to_period("Q").dt.end_time.dt.normalize()

    # One row per (ticker, quarter, item) → pivot to wide columns
    long_rows: list[dict] = []
    for _, r in df.iterrows():
        long_rows.append({
            "ticker": r["ticker"],
            "quarter_end": r["quarter_end"],
            "item": "_total",
        })
        for item in r["items"] or []:
            if item not in _TRACKED_ITEMS:
                continue
            long_rows.append({
                "ticker": r["ticker"],
                "quarter_end": r["quarter_end"],
                "item": item,
            })
    if not long_rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(long_rows)
    pivot = (
        long_df.groupby(["ticker", "quarter_end", "item"])
        .size()
        .unstack("item", fill_value=0)
        .reset_index()
    )
    # Prefix the count columns
    rename_map = {
        "_total": "e8k_n_filings",
        **{c: f"e8k_item_{c.replace('.', '_')}" for c in _TRACKED_ITEMS},
    }
    pivot = pivot.rename(columns=rename_map)
    # Ensure all expected columns exist
    for c in rename_map.values():
        if c not in pivot.columns:
            pivot[c] = 0
    return pivot
