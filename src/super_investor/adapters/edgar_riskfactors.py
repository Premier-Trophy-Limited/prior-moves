"""EDGAR 10-K / 10-Q Risk Factors section count + textual richness.

Reuses the existing ``edgar_8k.py`` index walking pattern. For each top-N
ticker, walk recent 10-K/10-Q filings (5yr window). Extract risk-factor
section length + count of distinct risk headings. Channel prefix ``rf_*``.

Lightweight signal: filings with growing risk-factor sections / unusual
new wording often precede sentiment shifts.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

UA = "super-investor-mirror research@example.com"


@dataclass
class RiskFactorFiling:
    ticker: str
    cik: str
    form: str
    filing_date: pd.Timestamp
    risk_section_len: int
    n_risk_headings: int


def _get(url: str, retries: int = 3) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=30)
            if r.status_code == 200:
                return r.text
            time.sleep(2)
        except Exception as e:
            last = e
            time.sleep(2)
    if last:
        raise RuntimeError(f"GET {url} failed: {last}")
    return ""


_CIK_LOOKUP = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=10-&dateb=&owner=include&count=40"
_RISK_HEADING_RE = re.compile(r"item\s*1a[\s\.\-]*risk\s*factors", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_risk_filings(
    ticker: str,
    cik: str,
    after: pd.Timestamp,
    cache_dir: Path | None = None,
    max_filings: int = 20,
) -> list[RiskFactorFiling]:
    """Pull recent 10-K / 10-Q filings and extract risk-factor stats."""
    safe = ticker.upper().replace("/", "-")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    idx_url = (
        f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    )
    cp_idx = cache_dir / f"{safe}_index.json" if cache_dir else None
    try:
        if cp_idx and cp_idx.exists() and cp_idx.stat().st_size > 1000:
            import json
            data = json.loads(cp_idx.read_text())
        else:
            txt = _get(idx_url)
            if not txt:
                return []
            if cp_idx:
                cp_idx.write_text(txt)
            import json
            data = json.loads(txt)
    except Exception:
        return []
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", [])
    primary_doc = recent.get("primaryDocument", [])
    accession = recent.get("accessionNumber", [])
    fdates = recent.get("filingDate", [])
    out: list[RiskFactorFiling] = []
    for i, frm in enumerate(forms):
        if frm not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
            continue
        try:
            fd = pd.Timestamp(fdates[i])
        except Exception:
            continue
        if fd < after:
            continue
        acc = accession[i].replace("-", "")
        doc = primary_doc[i]
        if not doc:
            continue
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        )
        cp_doc = cache_dir / f"{safe}_{acc}.txt" if cache_dir else None
        if cp_doc and cp_doc.exists():
            try:
                body = cp_doc.read_text(errors="replace")
            except Exception:
                body = ""
        else:
            try:
                body = _get(doc_url)
                if cp_doc and body:
                    cp_doc.write_text(body[:2_000_000], errors="replace")
            except Exception:
                body = ""
        if not body:
            continue
        # rough risk-section extraction: find 'item 1a' through 'item 1b' or 'item 2'
        m = _RISK_HEADING_RE.search(body)
        if not m:
            continue
        start = m.end()
        end_m = re.search(r"item\s*[12]b|item\s*2[\s\.\-]", body[start:], re.IGNORECASE)
        section = body[start : start + (end_m.start() if end_m else 200_000)]
        section_text = _TAG_RE.sub(" ", section)
        section_text = re.sub(r"\s+", " ", section_text)
        out.append(RiskFactorFiling(
            ticker=safe,
            cik=str(cik),
            form=frm,
            filing_date=fd,
            risk_section_len=len(section_text),
            n_risk_headings=len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+", section_text)),
        ))
        if len(out) >= max_filings:
            break
        time.sleep(0.2)
    return out


def aggregate_to_ticker_quarter(
    filings: Iterable[RiskFactorFiling],
) -> pd.DataFrame:
    rows = []
    for f in filings:
        q_end = f.filing_date.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        rows.append({
            "ticker": f.ticker,
            "quarter_end": q_end,
            "len": f.risk_section_len,
            "n_head": f.n_risk_headings,
            "form": f.form,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "rf_n_filings", "rf_section_len_mean", "rf_section_len_max",
            "rf_n_headings_mean", "rf_has_10k",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        rf_n_filings=("len", "size"),
        rf_section_len_mean=("len", "mean"),
        rf_section_len_max=("len", "max"),
        rf_n_headings_mean=("n_head", "mean"),
        rf_has_10k=("form", lambda x: int(any("10-K" in v for v in x))),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
