"""SEC Form 4 XML body parser + bulk fetcher.

Form 4 = insider trading report. Officers, directors, and 10% owners must
file within 2 business days of any insider trade. Per-filing XML contains:

  issuerCik / issuerTradingSymbol            — the company being traded
  reportingOwner / rptOwnerName               — the insider
  nonDerivativeTable.transaction.code         — P (open-market buy), S (sale),
    M (option exercise + sale), F (tax withhold), A (grant), D, G, I
  transactionAmounts.transactionShares.value  — share count
  transactionAmounts.transactionPricePerShare.value  — $ per share
  transactionDate.value                       — when the trade happened

This is the canonical insider feed; Finnhub's /stock/insider-transactions
endpoint wraps it. We parse direct from EDGAR for two reasons:

  1. Free, unlimited rate (10 req/sec polite cap).
  2. Covers EVERY US-listed company; Finnhub Starter doesn't gate by ticker
     but does gate news-sentiment etc., and we want consistent per-(ticker,
     quarter) aggregation across our entire 13F universe.

Output schema:
  Form4Transaction: ticker, issuer_cik, insider_cik, insider_name,
                    transaction_date, code, shares, price_per_share,
                    direct_or_indirect, accession
"""
from __future__ import annotations

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger("super_investor.adapters.form4")


_MIN_REQUEST_GAP_S = 0.11  # ~9 req/sec; polite limit is 10/sec


_USER_AGENT_DEFAULT = "super-investor-mirror researcher@example.com"


@dataclass(frozen=True)
class Form4Transaction:
    ticker: str
    issuer_cik: str
    insider_cik: str
    insider_name: str
    transaction_date: pd.Timestamp
    code: str
    shares: float
    price_per_share: float
    direct_or_indirect: str  # D direct / I indirect
    accession: str


def _txt(element: ET.Element, tag: str, default: str = "") -> str:
    if element is None:
        return default
    found = element.find(tag)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def parse_form4_xml(xml_bytes: bytes, accession: str = "") -> list[Form4Transaction]:
    """Parse a single Form 4 XML body into a list of insider transactions.

    Form 4 may have entries under <nonDerivativeTable> (common stock trades)
    and/or <derivativeTable> (options, RSUs, etc.). We parse non-derivative
    only because that's what drives the price-relevant signal for our use case.
    Derivative grants matter for compensation but rarely move the model.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.warning("parse_form4_xml(%s): %s: %s", accession, type(e).__name__, e)
        return []

    issuer = root.find("issuer")
    issuer_cik = _txt(issuer, "issuerCik")
    ticker = _txt(issuer, "issuerTradingSymbol")
    if not ticker:
        return []

    insider = root.find("reportingOwner")
    insider_cik = ""
    insider_name = ""
    if insider is not None:
        rid = insider.find("reportingOwnerId")
        if rid is not None:
            insider_cik = _txt(rid, "rptOwnerCik")
            insider_name = _txt(rid, "rptOwnerName")

    transactions: list[Form4Transaction] = []
    non_deriv = root.find("nonDerivativeTable")
    if non_deriv is None:
        return []
    for tx in non_deriv.findall("nonDerivativeTransaction"):
        coding = tx.find("transactionCoding")
        code = _txt(coding, "transactionCode")
        date_el = tx.find("transactionDate")
        date_str = _txt(date_el, "value")
        try:
            date = pd.Timestamp(date_str)
        except Exception:
            continue
        amounts = tx.find("transactionAmounts")
        shares_el = None
        price_el = None
        if amounts is not None:
            shares_el = amounts.find("transactionShares")
            price_el = amounts.find("transactionPricePerShare")
        shares = 0.0
        if shares_el is not None:
            try:
                shares = float(_txt(shares_el, "value", "0"))
            except ValueError:
                shares = 0.0
        price = 0.0
        if price_el is not None:
            try:
                price = float(_txt(price_el, "value", "0"))
            except ValueError:
                price = 0.0
        ownership_nature = tx.find("ownershipNature")
        direct_or_indirect = _txt(ownership_nature, "directOrIndirectOwnership/value", "D")
        # ElementTree's xpath is limited; fall back to manual traversal
        if direct_or_indirect == "D" and ownership_nature is not None:
            doi = ownership_nature.find("directOrIndirectOwnership")
            if doi is not None and doi.text is not None:
                direct_or_indirect = doi.text.strip()
        transactions.append(Form4Transaction(
            ticker=ticker, issuer_cik=issuer_cik,
            insider_cik=insider_cik, insider_name=insider_name,
            transaction_date=date, code=code, shares=shares,
            price_per_share=price, direct_or_indirect=direct_or_indirect,
            accession=accession,
        ))
    return transactions


class Form4Client:
    """Polite EDGAR Form 4 fetcher: list filings per CIK, download + parse each."""

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
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)),
    )
    def _get(self, url: str) -> bytes:
        self._throttle()
        with httpx.Client(timeout=20.0, headers=self._headers, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.content

    def list_filings(self, issuer_cik: str, forms: tuple[str, ...] = ("4",)) -> list[dict]:
        """Return all filings for the issuer matching `forms` (default just Form 4)."""
        cik = str(issuer_cik).lstrip("0").zfill(10)
        cache_path = self._cache_dir / "submissions" / f"CIK{cik}.json" if self._cache_dir else None
        if cache_path and cache_path.exists():
            raw = cache_path.read_bytes()
        else:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                raw = self._get(url)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return []
                raise
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(raw)
        payload = json.loads(raw)
        recent = payload.get("filings", {}).get("recent", {})
        rows = []
        for i in range(len(recent.get("form", []))):
            if recent["form"][i] in forms:
                accession = recent["accessionNumber"][i]
                primary_doc = recent["primaryDocument"][i]
                rows.append({
                    "form": recent["form"][i],
                    "accession": accession,
                    "primary_document": primary_doc,
                    "filed_date": recent["filingDate"][i],
                    "period_of_report": recent.get("reportDate", [None] * len(recent["form"]))[i],
                })
        return rows

    def fetch_form4_xml(self, issuer_cik: str, accession: str, primary_doc: str) -> bytes:
        """Fetch the machine-readable Form 4 XML.

        Submissions JSON often advertises an XSLT-styled variant under
        `xslF345X06/form4.xml` which is HTML, not the raw schema. Strip the
        prefix and request the bare `form4.xml` instead (canonical location
        in every Form 4 archive directory).
        """
        cik = str(issuer_cik).lstrip("0")
        accession_nodash = accession.replace("-", "")
        # Strip XSLT prefix and any subdirectory; the raw XML lives at the root.
        raw_doc = primary_doc.split("/")[-1]
        cache_path = (self._cache_dir / "form4" / cik / accession_nodash / raw_doc) if self._cache_dir else None
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{raw_doc}"
        try:
            raw = self._get(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return b""
            raise
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw)
        return raw

    def pull_ticker(self, issuer_cik: str, since: pd.Timestamp | None = None,
                    max_filings: int = 500) -> list[Form4Transaction]:
        """Pull every Form 4 transaction for `issuer_cik` filed at or after `since`.

        max_filings caps per-ticker runtime: top mega-caps have 5k+ Form 4 filings
        over a decade; we sort newest-first and stop at the cap so a single
        ticker can't dominate the run.
        """
        filings = self.list_filings(issuer_cik, forms=("4",))
        # Filter by since first, then sort newest-first, then cap
        after_since: list[dict] = []
        for f in filings:
            try:
                filed = pd.Timestamp(f["filed_date"])
            except Exception:
                continue
            if since is not None and filed < since:
                continue
            after_since.append(f)
        after_since.sort(key=lambda x: x.get("filed_date", ""), reverse=True)
        after_since = after_since[:max_filings]

        out: list[Form4Transaction] = []
        for f in after_since:
            xml = self.fetch_form4_xml(issuer_cik, f["accession"], f["primary_document"])
            if not xml:
                continue
            out.extend(parse_form4_xml(xml, accession=f["accession"]))
        return out


def aggregate_to_quarter(txs: list[Form4Transaction]) -> pd.DataFrame:
    """Roll up a list of Form 4 transactions to per-(ticker, quarter) aggregates."""
    if not txs:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ticker": t.ticker,
        "transaction_date": t.transaction_date,
        "code": t.code,
        "shares": t.shares,
        "price_per_share": t.price_per_share,
        "insider_name": t.insider_name,
    } for t in txs])
    df["quarter_end"] = df["transaction_date"].dt.to_period("Q").dt.end_time.dt.normalize()
    df["dollars"] = df["shares"] * df["price_per_share"]
    agg = df.groupby(["ticker", "quarter_end"]).agg(
        n_transactions=("code", "size"),
        n_open_market_buys=("code", lambda s: int((s == "P").sum())),
        n_open_market_sells=("code", lambda s: int((s == "S").sum())),
        n_option_exercises=("code", lambda s: int((s == "M").sum())),
        n_grants=("code", lambda s: int((s == "A").sum())),
        total_buy_usd=("dollars", lambda s: float(s[df.loc[s.index, "code"] == "P"].sum())),
        total_sell_usd=("dollars", lambda s: float(s[df.loc[s.index, "code"] == "S"].sum())),
        max_buy_usd=("dollars", lambda s: float(s[df.loc[s.index, "code"] == "P"].max() if (df.loc[s.index, "code"] == "P").any() else 0.0)),
        n_unique_insiders=("insider_name", "nunique"),
    ).reset_index()
    return agg
