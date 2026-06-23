"""S&P 500 constituent history and quarter-end fundamentals.

Two-stage build:

1. Constituents: scrape the current list from the Wikipedia page and merge with
   the historical changes table that lives on the same page. The result is a
   per-(ticker, date) presence flag spanning ~1996-present. Free, no API key.

2. Per (ticker, quarter): yfinance Ticker.history + Ticker.quarterly_financials
   gives close, volume, PE, P/B, ROE, debt-to-equity, profit margin. Free, but
   yfinance rate-limits roughly 2000 requests/hour anonymously — chunk requests
   and cache.

Output:
    data/tickers/sp500_history.parquet  (ticker, added_at, removed_at)
    data/features/sp500_quarters.parquet  (ticker, quarter_end, close, ret_1q,
                                           ret_4q, vol_1y, pe, pb, roe, d2e,
                                           margin, market_cap)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class ConstituentRow:
    ticker: str
    added_at: pd.Timestamp | None
    removed_at: pd.Timestamp | None
    name: str = ""
    sector: str = ""


def fetch_sp500_history() -> pd.DataFrame:
    """Scrape Wikipedia for current constituents + change history.

    Returns a DataFrame keyed by (ticker, added_at). A ticker removed and later
    re-added produces two rows. Tickers with no removal date are still in the
    index as of scrape time.
    """
    import httpx
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "super-investor-mirror sp500-scraper/0.1"}
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        r = client.get(WIKI_URL)
        r.raise_for_status()
        html = r.text

    tables = pd.read_html(html)
    # Two tables: current constituents (idx 0), historical changes (idx 1).
    current = tables[0].copy()
    changes = tables[1].copy() if len(tables) > 1 else pd.DataFrame()

    current.columns = [str(c).strip().lower().replace(" ", "_") for c in current.columns]
    if "symbol" in current.columns:
        current = current.rename(columns={"symbol": "ticker"})
    if "gics_sector" in current.columns:
        current = current.rename(columns={"gics_sector": "sector"})
    if "security" in current.columns:
        current = current.rename(columns={"security": "name"})
    current["added_at"] = pd.to_datetime(current.get("date_added", pd.NaT), errors="coerce")
    current["removed_at"] = pd.NaT
    current_keep = current[["ticker", "name", "sector", "added_at", "removed_at"]].copy()

    # Parse changes table: flatten MultiIndex columns if present
    if not changes.empty:
        if isinstance(changes.columns, pd.MultiIndex):
            changes.columns = ["_".join(str(c) for c in col).strip().lower().replace(" ", "_")
                                for col in changes.columns]
        else:
            changes.columns = [str(c).strip().lower().replace(" ", "_") for c in changes.columns]
        rename_map = {}
        for c in changes.columns:
            if "added" in c and "ticker" in c:
                rename_map[c] = "added_ticker"
            if "added" in c and "security" in c:
                rename_map[c] = "added_name"
            if "removed" in c and "ticker" in c:
                rename_map[c] = "removed_ticker"
            if "removed" in c and "security" in c:
                rename_map[c] = "removed_name"
            if c == "date" or c.endswith("_date"):
                rename_map[c] = "change_date"
        changes = changes.rename(columns=rename_map)
        change_rows = []
        for _, row in changes.iterrows():
            d = pd.to_datetime(row.get("change_date", pd.NaT), errors="coerce")
            if row.get("removed_ticker"):
                change_rows.append({
                    "ticker": str(row["removed_ticker"]).strip(),
                    "name": str(row.get("removed_name", "")).strip(),
                    "removed_at": d,
                })
        removed_df = pd.DataFrame(change_rows)
        if not removed_df.empty:
            # Attach removal date to whichever current_keep row matches; current set
            # are not removed (since they're in current). So removed_df rows are
            # tickers that exist only historically.
            removed_df["added_at"] = pd.NaT
            removed_df["sector"] = ""
            current_keep = pd.concat([current_keep, removed_df], ignore_index=True)

    current_keep["ticker"] = current_keep["ticker"].astype(str).str.strip()
    return current_keep[["ticker", "name", "sector", "added_at", "removed_at"]]


def is_in_sp500(history: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> bool:
    """Was `ticker` an S&P 500 member on `as_of`?"""
    rows = history[history["ticker"] == ticker]
    if rows.empty:
        return False
    for _, r in rows.iterrows():
        added = r["added_at"]
        removed = r["removed_at"]
        before_added = pd.notna(added) and as_of < added
        after_removed = pd.notna(removed) and as_of >= removed
        if before_added or after_removed:
            continue
        return True
    return False
