"""Sector classification from SEC SIC codes (keyless, deterministic, free).

GICS sectors are proprietary. yfinance exposes a GICS-ish ``sector`` but is
rate-limited and flaky for hundreds of tickers. The SEC submissions API
(``https://data.sec.gov/submissions/CIK##########.json``) returns an ``sic``
code for every filer, keyless and fast. We map SIC ranges to an 11-bucket
GICS-style taxonomy with the tech/semis special-cases that matter for the
concentration view (the Top Picks skew semiconductor-heavy).

SIC is imperfect (it predates the modern tech economy) so the mapping
special-cases the codes that would otherwise land in "Manufacturing":
  3674 semiconductors, 357x computers, 737x software/IT → Information Technology
  3711 autos, 5961 catalog retail → Consumer Discretionary, etc.

Cache: data/tickers/sectors.parquet  (ticker, sic, sic_desc, sector)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TICKERS_DIR = REPO / "data" / "tickers"
CACHE = TICKERS_DIR / "sectors.parquet"
UA = "super-investor-mirror sector-map chakhanghowardchan2008@gmail.com"

GICS_ORDER = [
    "Information Technology", "Health Care", "Financials",
    "Consumer Discretionary", "Communication Services", "Industrials",
    "Consumer Staples", "Energy", "Materials", "Real Estate", "Utilities",
    "Unknown",
]


def sic_to_sector(sic: int | None) -> str:
    """Map a numeric SIC code to an 11-bucket GICS-style sector."""
    if sic is None or not isinstance(sic, (int, float)) or sic != sic:
        return "Unknown"
    s = int(sic)

    # --- Information Technology (special-cased out of Manufacturing) ---
    if (3570 <= s <= 3579 or 3670 <= s <= 3679 or 3661 <= s <= 3669
            or 7370 <= s <= 7379 or s in (3559, 3674)):
        return "Information Technology"
    # --- Communication Services ---
    if (4800 <= s <= 4899 or 2700 <= s <= 2799 or 7800 <= s <= 7841
            or 4830 <= s <= 4841 or s in (7310, 7311, 7312)):
        return "Communication Services"
    # --- Health Care (incl. health insurers, which SIC files under 632x
    #     "Accident & Health Insurance / Hospital & Medical Service Plans" —
    #     intuitively Health Care, not Financials: UNH, HUM, ELV, CI, CNC) ---
    if (2830 <= s <= 2836 or 3840 <= s <= 3851 or 8000 <= s <= 8099
            or 6320 <= s <= 6324
            or s in (2833, 2834, 2835, 2836, 5912, 3826)):
        return "Health Care"
    # --- Real Estate (before Financials so REITs split out) ---
    if (6500 <= s <= 6599 or s == 6798):
        return "Real Estate"
    # --- Financials ---
    if (6000 <= s <= 6499 or 6700 <= s <= 6799):
        return "Financials"
    # --- Energy ---
    if (1300 <= s <= 1399 or 2900 <= s <= 2999 or 1000 <= s <= 1049
            or s in (5171, 1311)):
        return "Energy"
    # --- Consumer Staples ---
    if (2000 <= s <= 2199 or 2080 <= s <= 2099 or 5400 <= s <= 5499
            or 2840 <= s <= 2844 or s in (2100, 5140, 5141)):
        return "Consumer Staples"
    # --- Consumer Discretionary ---
    if (5200 <= s <= 5999 or 2300 <= s <= 2399 or 3700 <= s <= 3716
            or 7000 <= s <= 7299 or 5700 <= s <= 5736 or 3630 <= s <= 3639
            or s in (3711, 3714, 3751, 2510, 2511)):
        return "Consumer Discretionary"
    # --- Utilities ---
    if 4900 <= s <= 4999:
        return "Utilities"
    # --- Materials ---
    if (1400 <= s <= 1499 or 2600 <= s <= 2699 or 2800 <= s <= 2829
            or 3300 <= s <= 3399 or 3200 <= s <= 3299 or 1040 <= s <= 1099
            or 1200 <= s <= 1299):
        return "Materials"
    # --- Industrials (broad manufacturing + transport + construction) ---
    if (3400 <= s <= 3569 or 3580 <= s <= 3669 or 3680 <= s <= 3699
            or 3720 <= s <= 3799 or 4000 <= s <= 4799 or 1500 <= s <= 1799
            or 8700 <= s <= 8744):
        return "Industrials"
    # default bucket for the remaining 2xxx/3xxx manufacturers
    if 2000 <= s <= 3999:
        return "Industrials"
    return "Unknown"


def _ticker_to_cik() -> dict[str, str]:
    p = TICKERS_DIR / "sec_company_tickers.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    rows = raw.values() if isinstance(raw, dict) else raw
    out: dict[str, str] = {}
    for r in rows:
        t = str(r.get("ticker", "")).upper()
        cik = r.get("cik_str") or r.get("cik")
        if t and cik is not None:
            out[t] = str(int(cik)).zfill(10)
    return out


def fetch_sectors(tickers: list[str], sleep: float = 0.12) -> pd.DataFrame:
    """Fetch SIC + sector for each ticker via SEC submissions; cache & reuse.

    Incremental: tickers already in the cache are skipped, so re-runs are cheap.
    """
    import requests

    TICKERS_DIR.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(
        columns=["ticker", "sic", "sic_desc", "sector"]
    )
    have_set = set(have["ticker"].astype(str).str.upper()) if not have.empty else set()
    want = [str(t).upper() for t in tickers if str(t).upper() not in have_set]
    want = sorted(set(want))
    if not want:
        return have

    t2c = _ticker_to_cik()
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    new_rows = []
    for i, tk in enumerate(want):
        cik = t2c.get(tk)
        if not cik:
            new_rows.append({"ticker": tk, "sic": None, "sic_desc": "", "sector": "Unknown"})
            continue
        try:
            r = sess.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=20)
            if r.status_code == 200:
                j = r.json()
                sic_raw = j.get("sic")
                sic = int(sic_raw) if str(sic_raw).isdigit() else None
                new_rows.append({
                    "ticker": tk, "sic": sic,
                    "sic_desc": j.get("sicDescription", "") or "",
                    "sector": sic_to_sector(sic),
                })
            else:
                new_rows.append({"ticker": tk, "sic": None, "sic_desc": "", "sector": "Unknown"})
        except Exception:
            new_rows.append({"ticker": tk, "sic": None, "sic_desc": "", "sector": "Unknown"})
        if (i + 1) % 50 == 0:
            print(f"  sectors {i + 1}/{len(want)}", flush=True)
            # checkpoint
            pd.concat([have, pd.DataFrame(new_rows)], ignore_index=True).to_parquet(CACHE, index=False)
        time.sleep(sleep)

    out = pd.concat([have, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.drop_duplicates("ticker", keep="last").reset_index(drop=True)
    out.to_parquet(CACHE, index=False)
    return out


def load_sector_map() -> dict[str, str]:
    """ticker -> sector dict from the cache (empty if not built yet)."""
    if not CACHE.exists():
        return {}
    d = pd.read_parquet(CACHE)
    return dict(zip(d["ticker"].astype(str).str.upper(), d["sector"]))
