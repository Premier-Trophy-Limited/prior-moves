"""Additional SEC EDGAR forms: Form 4 (insider trading), Schedule 13D/13G (activist).

These complement the 13F-HR adapter with higher-frequency signals:

- **Form 4**: officers/directors/10% owners must file within 2 business days of
  any insider trade. Per (ticker, quarter): count of buys, sells, option
  exercises; total $ value; top-insider concentration. Strong signal for value
  investors (Buffett quote: "There's only one reason insiders buy: they think
  the stock will go up.")

- **Schedule 13D**: any investor crossing 5% ownership with activist intent
  must file within 10 days. THE Ackman/Loeb/Icahn signal — these filings are
  the picks themselves.

- **Schedule 13G**: same 5% threshold but passive intent (institutional
  long-onlies). Less actionable but still a position signal.

EDGAR endpoints reused from sec_13f.py:
  - https://data.sec.gov/submissions/CIK{cik}.json — per-CIK filing list
  - https://efts.sec.gov/LATEST/search-index?q=...&forms=... — full-text search
  - https://data.sec.gov/submissions/CIK{cik}.json — per-CIK filing list

For Form 4 + 13D/G the canonical pattern is to query EDGAR's full-text search
by ticker/CUSIP, not per-investor CIK (since the FILER is the insider/activist,
not the company). Each filing references the SUBJECT company by CIK.

Output schemas:
  insider_transactions: ticker, period, n_buys, n_sells, n_exercises,
                        total_buy_usd, total_sell_usd, top_insider_concentration
  activist_disclosures: ticker, period, filing_date, filer_cik, filer_name,
                        shares_owned, pct_owned, schedule_type (13D|13G|13D/A|13G/A),
                        purpose_text
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from super_investor.adapters.sec_13f import _MIN_REQUEST_GAP_S, _USER_AGENT_DEFAULT
import os

log = logging.getLogger("super_investor.adapters.edgar_forms")


_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"


@dataclass(frozen=True)
class EdgarFilingRef:
    cik: str
    accession: str
    accession_nodash: str
    form: str
    filed_at: pd.Timestamp
    period_of_report: pd.Timestamp | None
    subject_cik: str = ""    # company being reported on (Form 4) — None for 13F-HR


class EdgarFormsClient:
    """Polite client for non-13F EDGAR forms (Form 4, 13D, 13G)."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None):
        ua = user_agent or os.environ.get("SEC_USER_AGENT") or _USER_AGENT_DEFAULT
        if "@" not in ua:
            raise ValueError("SEC_USER_AGENT must include contact email")
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._last_t: float = 0.0
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        now = time.monotonic()
        delta = now - self._last_t
        if delta < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - delta)
        self._last_t = time.monotonic()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _get(self, url: str, params: dict | None = None) -> bytes:
        self._throttle()
        with httpx.Client(timeout=30.0, headers=self._headers, follow_redirects=True) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.content

    def _get_cached(self, url: str, cache_subpath: str, params: dict | None = None) -> bytes:
        if self._cache_dir:
            p = self._cache_dir / cache_subpath
            if p.exists():
                return p.read_bytes()
            content = self._get(url, params=params)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            return content
        return self._get(url, params=params)

    def search_filings_for_subject(
        self,
        subject_cik: str,
        forms: tuple[str, ...],
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> list[EdgarFilingRef]:
        """Use EDGAR full-text search to find Form 4 / 13D / 13G filings naming the
        subject company by CIK. Returns metadata; download body separately.
        """
        subject_cik = subject_cik.zfill(10)
        params = {
            "q": "",
            "ciks": subject_cik,
            "forms": ",".join(forms),
            "dateRange": "custom",
            "startdt": (start_date or pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d"),
            "enddt": (end_date or pd.Timestamp.now()).strftime("%Y-%m-%d"),
        }
        cache_key = (
            f"search/{subject_cik}/{'_'.join(forms)}_"
            f"{params['startdt']}_{params['enddt']}.json"
        )
        raw = self._get_cached(_EDGAR_SEARCH, cache_key, params=params)
        payload = json.loads(raw)
        hits = payload.get("hits", {}).get("hits", [])
        out: list[EdgarFilingRef] = []
        for h in hits:
            s = h.get("_source", {})
            ciks_field = s.get("ciks", [])
            filer_cik = (ciks_field[0] if ciks_field else "").zfill(10)
            accession = h.get("_id", "").split(":")[0]
            if not accession:
                continue
            out.append(EdgarFilingRef(
                cik=filer_cik,
                accession=accession,
                accession_nodash=accession.replace("-", ""),
                form=s.get("forms", [""])[0],
                filed_at=pd.Timestamp(s.get("file_date", "")),
                period_of_report=pd.Timestamp(s.get("period_of_report", "")) if s.get("period_of_report") else None,
                subject_cik=subject_cik,
            ))
        return out


# ---------------------------------------------------------------------------
# Aggregation helpers — convert per-filing rows to per-(ticker, quarter) signals
# ---------------------------------------------------------------------------

def aggregate_form4_to_quarterly(form4_df: pd.DataFrame) -> pd.DataFrame:
    """Roll up Form 4 transactions to per-(ticker, quarter) counts + dollar volume.

    Expected input columns:
        ticker, filed_at, transaction_code, transaction_shares,
        transaction_price_per_share, insider_name

    transaction_code values per Form 4 spec:
        P = open-market purchase  (strongest bullish signal)
        S = open-market sale
        A = grant, award, other
        M = option exercise
        F = payment of exercise price / tax with shares
        D = sale to issuer
        G = bona fide gift
        I = discretionary transaction
    """
    if form4_df.empty:
        return pd.DataFrame()
    df = form4_df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["quarter"] = df["filed_at"].dt.to_period("Q").dt.start_time
    df["dollars"] = df["transaction_shares"].fillna(0).astype(float) * \
                    df["transaction_price_per_share"].fillna(0).astype(float)

    agg = df.groupby(["ticker", "quarter"]).agg(
        n_transactions=("transaction_code", "size"),
        n_open_market_buys=("transaction_code", lambda s: int((s == "P").sum())),
        n_open_market_sells=("transaction_code", lambda s: int((s == "S").sum())),
        n_option_exercises=("transaction_code", lambda s: int((s == "M").sum())),
        total_buy_usd=("dollars", lambda s: float(s[df.loc[s.index, "transaction_code"] == "P"].sum())),
        total_sell_usd=("dollars", lambda s: float(s[df.loc[s.index, "transaction_code"] == "S"].sum())),
        n_unique_insiders=("insider_name", lambda s: s.nunique()),
    ).reset_index()
    return agg


def aggregate_13dg_to_quarterly(disclosures_df: pd.DataFrame) -> pd.DataFrame:
    """Roll up Schedule 13D/G filings to per-(ticker, quarter) presence + activist flag.

    Expected input columns:
        ticker, filed_at, schedule_type, filer_cik, filer_name,
        pct_owned, purpose_text
    """
    if disclosures_df.empty:
        return pd.DataFrame()
    df = disclosures_df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["quarter"] = df["filed_at"].dt.to_period("Q").dt.start_time
    df["is_activist"] = df["schedule_type"].str.startswith("13D")
    agg = df.groupby(["ticker", "quarter"]).agg(
        n_activist_filings=("is_activist", "sum"),
        n_passive_filings=("is_activist", lambda s: int((~s).sum())),
        n_unique_filers=("filer_cik", "nunique"),
        max_pct_owned=("pct_owned", "max"),
    ).reset_index()
    return agg
