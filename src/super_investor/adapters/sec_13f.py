"""SEC EDGAR 13F-HR adapter.

Fetches 13F-HR (and 13F-HR/A amended) filings for a given institutional manager
CIK, parses the INFORMATION TABLE XML, and returns a DataFrame of holdings.

SEC EDGAR usage rules (terms of service):
  - Identify yourself in every request via a `User-Agent` header containing
    your name and email. Anonymous requests get rate-limited or blocked.
  - Maximum 10 requests/second. We use a 0.15s sleep between calls to stay
    well under.

Endpoints used:
  - https://data.sec.gov/submissions/CIK{cik}.json
      Lists every filing for a CIK; we filter to form types 13F-HR and 13F-HR/A.
  - https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{file}
      Individual filing artefacts. We download the info table XML.

XML structure (FORM 13F-HR INFORMATION TABLE, schema 13fInfoTable.xsd):
  <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
    <infoTable>
      <nameOfIssuer>APPLE INC</nameOfIssuer>
      <titleOfClass>COM</titleOfClass>
      <cusip>037833100</cusip>
      <value>178391000</value>             <!-- thousands of USD -->
      <shrsOrPrnAmt>
        <sshPrnamt>905560000</sshPrnamt>
        <sshPrnamtType>SH</sshPrnamtType>
      </shrsOrPrnAmt>
      <investmentDiscretion>DFND</investmentDiscretion>
      <votingAuthority>
        <Sole>905560000</Sole><Shared>0</Shared><None>0</None>
      </votingAuthority>
      <putCall>Put</putCall>               <!-- optional; only on derivatives -->
    </infoTable>
    ...
  </informationTable>

Note: 13F reports VALUES in $thousands; we multiply by 1000 on parse.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from lxml import etree
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("super_investor.adapters.sec_13f")


_USER_AGENT_DEFAULT = "super-investor-mirror dev@example.com"
_MIN_REQUEST_GAP_S = 0.15  # ~6 req/s, well under SEC's 10/s limit


@dataclass(frozen=True)
class FilingRef:
    cik: str
    accession: str        # e.g. "0001067983-24-000005"
    accession_nodash: str # e.g. "000106798324000005"
    form: str             # "13F-HR" or "13F-HR/A"
    filed_at: pd.Timestamp
    period_of_report: pd.Timestamp


class Edgar13FClient:
    """Polite, retrying HTTP client for SEC EDGAR 13F endpoints."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None):
        ua = user_agent or os.environ.get("SEC_USER_AGENT") or _USER_AGENT_DEFAULT
        if "@" not in ua:
            raise ValueError(
                "SEC_USER_AGENT must include a contact email (SEC EDGAR terms). "
                f"Got {ua!r}. Set SEC_USER_AGENT=\"Name email@example.com\" in .env."
            )
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._last_request_t: float = 0.0
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # One persistent client per Edgar13FClient: reuses the TCP/TLS
        # connection across the hundreds of small requests a full-history
        # scrape makes (the old per-request Client paid a handshake every
        # call). Throttling is unchanged — _MIN_REQUEST_GAP_S stays 0.15s,
        # well under SEC's 10 req/s ceiling.
        self._client = httpx.Client(timeout=30.0, headers=self._headers)

    def _throttle(self) -> None:
        now = time.monotonic()
        delta = now - self._last_request_t
        if delta < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - delta)
        self._last_request_t = time.monotonic()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _get(self, url: str) -> bytes:
        self._throttle()
        r = self._client.get(url)
        r.raise_for_status()
        return r.content

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Edgar13FClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get_cached(self, url: str, cache_subpath: str) -> bytes:
        if self._cache_dir:
            p = self._cache_dir / cache_subpath
            if p.exists():
                return p.read_bytes()
            content = self._get(url)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            return content
        return self._get(url)

    def list_filings(self, cik: str) -> list[FilingRef]:
        """Return every 13F-HR (+amended) filing for `cik`, oldest first."""
        cik = cik.zfill(10)
        # Recent filings live in the top-level submissions JSON; older filings are
        # linked via `files` -> a separate JSON per page.
        recent_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        recent_raw = self._get_cached(recent_url, f"submissions/CIK{cik}.json")
        import json
        recent = json.loads(recent_raw)

        all_rows: list[dict] = []
        recent_block = recent.get("filings", {}).get("recent", {})
        all_rows.extend(_block_to_rows(recent_block))
        for older in recent.get("filings", {}).get("files", []) or []:
            older_url = f"https://data.sec.gov/submissions/{older['name']}"
            older_raw = self._get_cached(older_url, f"submissions/{older['name']}")
            older_block = json.loads(older_raw)
            all_rows.extend(_block_to_rows(older_block))

        out: list[FilingRef] = []
        for r in all_rows:
            form = r.get("form", "")
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            accession = r.get("accessionNumber", "")
            if not accession:
                continue
            out.append(FilingRef(
                cik=cik,
                accession=accession,
                accession_nodash=accession.replace("-", ""),
                form=form,
                filed_at=pd.Timestamp(r.get("filingDate", "")),
                period_of_report=pd.Timestamp(r.get("reportDate", "")),
            ))
        out.sort(key=lambda f: f.period_of_report)
        return out

    def fetch_holdings(self, ref: FilingRef) -> pd.DataFrame:
        """Download + parse the INFORMATION TABLE XML for a single 13F filing.

        Filings vary in artefact naming: newer filings have `infotable.xml`,
        Berkshire filings use a numeric id like `32398.xml`, older filings use
        `informationtable.xml` or even plain `form13fInfoTable.xml`. We list
        the filing directory, then for each `.xml` file (skipping the known
        header file `primary_doc.xml`) we open it and accept the first one
        whose root tag is `informationTable`.
        """
        cik_int = int(ref.cik)
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{ref.accession_nodash}/"
            f"index.json"
        )
        idx_raw = self._get_cached(index_url, f"filings/{ref.cik}/{ref.accession}-index.json")
        import json
        idx = json.loads(idx_raw)
        items = idx.get("directory", {}).get("item", [])
        xml_names = [
            item["name"] for item in items
            if item.get("name", "").lower().endswith(".xml")
            and item.get("name", "").lower() != "primary_doc.xml"
        ]
        # Heuristically try most-likely names first (`infotable*`, `informationtable*`),
        # then fall through to anything else so Berkshire-style numeric names still work.
        def _priority(n: str) -> int:
            ln = n.lower()
            if "infotable" in ln or "informationtable" in ln:
                return 0
            if ln.startswith("form13f"):
                return 1
            return 2
        xml_names.sort(key=_priority)

        for name in xml_names:
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{ref.accession_nodash}/"
                f"{name}"
            )
            raw = self._get_cached(url, f"filings/{ref.cik}/{ref.accession}/{name}")
            head = raw[:200].lower()
            if b"informationtable" in head:
                return _parse_info_table_xml(raw, ref)
        raise RuntimeError(
            f"no information-table xml found in filing {ref.cik}/{ref.accession} "
            f"(tried {len(xml_names)} files)"
        )

    def iter_holdings(self, cik: str) -> Iterator[tuple[FilingRef, pd.DataFrame]]:
        """Yield (filing, holdings_df) for every 13F filing in oldest-first order."""
        for ref in self.list_filings(cik):
            try:
                df = self.fetch_holdings(ref)
            except Exception as e:
                # Don't fail the whole run on one bad filing; surface and skip.
                log.warning("iter_holdings(%s/%s): %s: %s", ref.cik, ref.accession, type(e).__name__, e)
                continue
            yield ref, df


def _block_to_rows(block: dict) -> list[dict]:
    """Convert one parallel-array submissions block to row dicts."""
    keys = list(block.keys())
    n = len(block.get("accessionNumber", []))
    out = []
    for i in range(n):
        out.append({k: block[k][i] for k in keys if i < len(block[k])})
    return out


_DEFAULT_INFO_NS = "http://www.sec.gov/edgar/document/thirteenf/informationtable"


def _parse_info_table_xml(raw: bytes, ref: FilingRef) -> pd.DataFrame:
    """Parse a 13F-HR information table XML.

    Robust to three namespace variations:
      1. Default namespace = informationtable schema (Berkshire-era filings).
      2. No namespace at all (very old filings).
      3. Default namespace + xsi binding (some EDGAR filers).

    Returns rows = one infoTable element each. value_usd is converted from
    filing thousands to plain USD.
    """
    root = etree.fromstring(raw)

    # Match by LOCAL tag name, ignoring namespace entirely. Handles every
    # variant: default xmlns, no namespace, xsi binding, AND prefixed
    # namespaces (e.g. <ns1:informationTable>, used by Himalaya/Millennium/
    # Icahn and many others) — the old default-ns-only logic returned 0 rows
    # for prefixed filings, silently dropping them.
    rows = root.xpath(".//*[local-name()='infoTable']")

    def _find(node, tag):
        hits = node.xpath(f"./*[local-name()={tag!r}]")
        return hits[0] if hits else None

    def _txt(node, tag):
        el = _find(node, tag)
        return el.text if el is not None else None

    records: list[dict] = []
    for tbl in rows:
        shares_node = _find(tbl, "shrsOrPrnAmt")
        shares = sh_type = None
        if shares_node is not None:
            shares = _txt(shares_node, "sshPrnamt")
            sh_type = _txt(shares_node, "sshPrnamtType")
        records.append({
            "cik": ref.cik,
            "accession": ref.accession,
            "filed_at": ref.filed_at,
            "period_of_report": ref.period_of_report,
            "name_of_issuer": _txt(tbl, "nameOfIssuer"),
            "title_of_class": _txt(tbl, "titleOfClass"),
            "cusip": (_txt(tbl, "cusip") or "").strip().upper(),
            "value_usd": int(_txt(tbl, "value") or 0) * 1000,
            "shares": int(shares or 0),
            "shares_type": sh_type or "",
            "put_call": _txt(tbl, "putCall") or "",
            "investment_discretion": _txt(tbl, "investmentDiscretion") or "",
        })
    return pd.DataFrame(records)
