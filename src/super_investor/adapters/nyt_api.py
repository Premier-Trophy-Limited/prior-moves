"""NYT Developer API — Article Search.

Requires free dev key (https://developer.nytimes.com/). Store in macOS
Keychain (preferred) or as fallback env var:

  security add-generic-password -s "nyt-api-key" -a "$USER" -w "<key>"
  # or, fallback:
  export NYT_API_KEY=...

Endpoint:
  https://api.nytimes.com/svc/search/v2/articlesearch.json
    ?q=$TICKER&fq=section_name:("Business"|"Markets")&begin_date=YYYYMMDD&end_date=...

Returns title + abstract + lead_paragraph + URL + pub_date. NOT full body
(paywalled). Sufficient signal for ticker mention + quarterly count.

Channel prefix ``ny_*``. Archive goes back to 1851.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from super_investor.secrets import get_secret

UA = "super-investor-mirror research@example.com"
BASE = "https://api.nytimes.com/svc/search/v2/articlesearch.json"


@dataclass
class NYTArticle:
    ticker: str
    title: str
    abstract: str
    url: str
    published_at: pd.Timestamp


def _key() -> str | None:
    return get_secret("nyt-api-key", env_var="NYT_API_KEY")


def search_ticker(
    ticker: str,
    begin_date: str = "20210601",
    end_date: str = "20260601",
    cache_dir: Path | None = None,
    max_pages: int = 10,
    company_name: str | None = None,
) -> list[NYTArticle]:
    """Search NYT Article Search.

    NYT indexes company names, not tickers — pass ``company_name`` (the human
    name like "Apple Inc.") to actually get hits. If omitted, falls back to
    the ticker symbol (rarely productive).
    """
    api_key = _key()
    if not api_key:
        return []
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    out: list[NYTArticle] = []
    query = company_name or safe
    for page in range(max_pages):
        url = BASE
        params = {
            "q": query,
            "begin_date": begin_date,
            "end_date": end_date,
            "page": page,
            "sort": "newest",
            "api-key": api_key,
        }
        cp = cache_dir / f"{safe}_p{page}.json" if cache_dir else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        data = None
        if cp and cp.exists() and cp.stat().st_size > 200:
            import json
            data = json.loads(cp.read_text())
        else:
            try:
                r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=20)
                if r.status_code == 200:
                    if cp:
                        cp.write_text(r.text)
                    data = r.json()
                elif r.status_code == 429:
                    time.sleep(12)
                    continue
            except Exception:
                pass
        if not data:
            break
        docs = (data.get("response") or {}).get("docs") or []
        if not docs:
            break
        for doc in docs:
            pub_raw = doc.get("pub_date", "")
            try:
                pub = pd.Timestamp(pub_raw)
                if pub.tzinfo is None:
                    pub = pub.tz_localize("UTC")
            except Exception:
                continue
            out.append(NYTArticle(
                ticker=safe,
                title=str((doc.get("headline") or {}).get("main", ""))[:300],
                abstract=str(doc.get("abstract") or "")[:500],
                url=str(doc.get("web_url") or ""),
                published_at=pub,
            ))
        if len(docs) < 10:
            break
        # NYT free tier: ~5/min, 4000/day. Throttle hard.
        time.sleep(12.5)
    return out


def aggregate_to_ticker_quarter(
    rows: Iterable[NYTArticle],
) -> pd.DataFrame:
    out_rows = []
    for r in rows:
        q_end = r.published_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        out_rows.append({
            "ticker": r.ticker,
            "quarter_end": q_end,
            "len_abstract": len(r.abstract or ""),
        })
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        ny_n_articles=("len_abstract", "size"),
        ny_mean_abstract_len=("len_abstract", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
