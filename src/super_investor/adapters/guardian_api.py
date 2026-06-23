"""Guardian Open Platform — Content Search API.

Requires free dev key (https://open-platform.theguardian.com/access/).
Store in macOS Keychain (preferred) or env-var fallback:

  security add-generic-password -s "guardian-api-key" -a "$USER" -w "<key>"
  # or:
  export GUARDIAN_API_KEY=...

Default tier: 12 req/sec, 5000/day.

Endpoint:
  https://content.guardianapis.com/search
    ?q=$TICKER&section=business|business/markets&from-date=...&page-size=50

Returns webTitle + webPublicationDate + sectionName + URL + fields.bodyText.
Channel prefix ``gd_*``. Archive goes back to 1999.
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
BASE = "https://content.guardianapis.com/search"


@dataclass
class GuardianArticle:
    ticker: str
    title: str
    body: str
    url: str
    section: str
    published_at: pd.Timestamp


def _key() -> str | None:
    return get_secret("guardian-api-key", env_var="GUARDIAN_API_KEY")


def search_ticker(
    ticker: str,
    from_date: str = "2021-06-01",
    to_date: str = "2026-06-01",
    cache_dir: Path | None = None,
    max_pages: int = 10,
) -> list[GuardianArticle]:
    api_key = _key()
    if not api_key:
        return []
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    out: list[GuardianArticle] = []
    for page in range(1, max_pages + 1):
        params = {
            "q": f'"{safe}"',
            "section": "business",
            "from-date": from_date,
            "to-date": to_date,
            "page": page,
            "page-size": 50,
            "order-by": "newest",
            "show-fields": "bodyText,trailText",
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
                r = requests.get(BASE, params=params, headers={"User-Agent": UA}, timeout=20)
                if r.status_code == 200:
                    if cp:
                        cp.write_text(r.text)
                    data = r.json()
                elif r.status_code == 429:
                    time.sleep(10)
                    continue
            except Exception:
                pass
        if not data:
            break
        results = (data.get("response") or {}).get("results") or []
        if not results:
            break
        for art in results:
            pub_raw = art.get("webPublicationDate", "")
            try:
                pub = pd.Timestamp(pub_raw)
                if pub.tzinfo is None:
                    pub = pub.tz_localize("UTC")
            except Exception:
                continue
            fields = art.get("fields") or {}
            out.append(GuardianArticle(
                ticker=safe,
                title=str(art.get("webTitle") or "")[:300],
                body=str(fields.get("bodyText") or fields.get("trailText") or "")[:5000],
                url=str(art.get("webUrl") or ""),
                section=str(art.get("sectionName") or ""),
                published_at=pub,
            ))
        pages = (data.get("response") or {}).get("pages", 0)
        if page >= pages:
            break
        time.sleep(0.2)
    return out


def aggregate_to_ticker_quarter(rows: Iterable[GuardianArticle]) -> pd.DataFrame:
    out_rows = []
    for r in rows:
        q_end = r.published_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        out_rows.append({
            "ticker": r.ticker,
            "quarter_end": q_end,
            "body_len": len(r.body or ""),
        })
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        gd_n_articles=("body_len", "size"),
        gd_mean_body_len=("body_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
