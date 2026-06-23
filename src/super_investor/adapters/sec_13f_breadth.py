"""SEC Form 13F structured datasets — total institutional breadth per ticker.

Beyond the 12 tracked super-investors: how MANY of the ~5,000 13F filers hold
each name, and how that count + aggregate value moves quarter over quarter.
This is the "smart-money breadth / institutional flow" signal.

Source (free, no key): the SEC Form 13F data sets, one zip per filing window:
  https://www.sec.gov/files/structureddata/data/form-13f-data-sets/<window>_form13f.zip

Each zip:
  INFOTABLE.tsv  — holdings: ACCESSION_NUMBER, CUSIP, VALUE, SSHPRNAMT, PUTCALL
  SUBMISSION.tsv — ACCESSION_NUMBER → PERIODOFREPORT

Aggregate per (CUSIP, quarter_end): n_filers, total_value, total_shares.
Channel prefix ``inst_`` with QoQ deltas.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("super_investor.adapters.sec_13f_breadth")

UA = "Mozilla/5.0 super-investor-mirror research@example.com"
INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
BASE = "https://www.sec.gov"


def list_dataset_urls() -> list[str]:
    r = requests.get(INDEX, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    rels = re.findall(r'href="(/files/structureddata/data/form-13f-data-sets/[^"]+\.zip)"', r.text)
    seen, out = set(), []
    for rel in rels:
        if rel in seen:
            continue
        seen.add(rel)
        out.append(BASE + rel)
    return out


def fetch_window(url: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download one 13F window zip and aggregate to (cusip, quarter_end)."""
    fname = url.rsplit("/", 1)[-1]
    cp = cache_dir / fname if cache_dir else None
    raw: bytes | None = None
    if cp and cp.exists() and cp.stat().st_size > 10_000:
        raw = cp.read_bytes()
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
            if r.status_code == 200:
                raw = r.content
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cp.write_bytes(raw)
        except Exception as e:
            log.warning("fetch_window(%s): %s: %s", url, type(e).__name__, e)
            return pd.DataFrame()
    if not raw:
        return pd.DataFrame()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        with zf.open("INFOTABLE.tsv") as fh:
            info = pd.read_csv(fh, sep="\t", usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "PUTCALL"],
                               dtype={"CUSIP": str}, low_memory=False)
        with zf.open("SUBMISSION.tsv") as fh:
            sub = pd.read_csv(fh, sep="\t", usecols=["ACCESSION_NUMBER", "PERIODOFREPORT"], low_memory=False)
    except Exception as e:
        log.warning("fetch_window(%s): %s: %s", url, type(e).__name__, e)
        return pd.DataFrame()

    # exclude option positions (PUTCALL set); keep plain share holdings
    info = info[info["PUTCALL"].isna() | (info["PUTCALL"].astype(str).str.strip() == "")]
    info = info.merge(sub, on="ACCESSION_NUMBER", how="left")
    info = info.dropna(subset=["PERIODOFREPORT", "CUSIP"])
    info["quarter_end"] = pd.to_datetime(info["PERIODOFREPORT"], errors="coerce")
    info = info.dropna(subset=["quarter_end"])
    # normalize CUSIP (9-char, upper)
    info["cusip"] = info["CUSIP"].astype(str).str.strip().str.upper().str[:9]

    g = info.groupby(["cusip", "quarter_end"], as_index=False).agg(
        inst_n_filers=("ACCESSION_NUMBER", "nunique"),
        inst_total_value=("VALUE", "sum"),
        inst_total_shares=("SSHPRNAMT", "sum"),
    )
    return g


def aggregate(frames: Iterable[pd.DataFrame], cusip_map: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # multiple windows can report the same quarter (amendments / late filings) →
    # take the max n_filers / value per (cusip, quarter)
    df = df.groupby(["cusip", "quarter_end"], as_index=False).agg(
        inst_n_filers=("inst_n_filers", "max"),
        inst_total_value=("inst_total_value", "max"),
        inst_total_shares=("inst_total_shares", "max"),
    )
    # QoQ deltas per cusip
    df = df.sort_values(["cusip", "quarter_end"])
    df["inst_n_filers_chg"] = df.groupby("cusip")["inst_n_filers"].diff()
    df["inst_value_chg_pct"] = df.groupby("cusip")["inst_total_value"].pct_change()
    # map cusip -> ticker
    cm = cusip_map[["cusip", "ticker"]].dropna().copy()
    cm["cusip"] = cm["cusip"].astype(str).str.upper().str[:9]
    df = df.merge(cm, on="cusip", how="left").dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    # one row per (ticker, quarter)
    df = df.sort_values("inst_total_value", ascending=False).drop_duplicates(["ticker", "quarter_end"])
    return df
