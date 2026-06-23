"""Hacker News mentions adapter via the Algolia HN search API.

HN is a useful early-signal source for tech-heavy investors (Druckenmiller,
Tepper, two_sigma) because story selection skews toward fundamentals-relevant
tech / earnings / regulatory news. Algolia exposes the entire HN corpus via:

  https://hn.algolia.com/api/v1/search?query=NVDA&tags=story
       &numericFilters=created_at_i>=<unix>,created_at_i<<unix>
       &hitsPerPage=100&page=<n>

Free, no auth, ~10k req/hour soft limit. Per (ticker, quarter_end) we extract:

  hn_n_stories, hn_mean_points, hn_mean_num_comments,
  hn_joined_titles (for Gemma embed)

Cashtag query (``$NVDA``) is the only reliable signal — bare company names
("Nvidia") match too many off-topic stories on HN. So this is a focused
signal: hits when the cashtag is explicitly mentioned in title.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_BASE_URL = "https://hn.algolia.com/api/v1/search"
_USER_AGENT = "super-investor-mirror/0.1 research educational"


@dataclass(frozen=True)
class HNStory:
    object_id: str
    ticker: str
    title: str
    points: int
    num_comments: int
    created_at: pd.Timestamp
    url: str


class HackerNewsClient:
    def __init__(self, cache_dir: Path | None = None, throttle_sec: float = 0.4):
        self._headers = {"User-Agent": _USER_AGENT}
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._throttle = throttle_sec
        self._last_t = 0.0

    def _gate(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < self._throttle:
            time.sleep(self._throttle - gap)
        self._last_t = time.monotonic()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)
        ),
    )
    def _get(self, params: dict) -> dict:
        self._gate()
        with httpx.Client(timeout=20.0, headers=self._headers) as c:
            r = c.get(_BASE_URL, params=params)
            r.raise_for_status()
            return r.json()

    def search_ticker(
        self,
        ticker: str,
        since: datetime,
        until: datetime,
        max_pages: int = 5,
        per_page: int = 100,
    ) -> list[HNStory]:
        """Return all HN stories mentioning the cashtag for ticker in window."""
        # Sanitize ticker for filesystem (TECK/B etc. — see yfinance adapter)
        safe = ticker.upper().replace("/", "-").replace(" ", "_")
        cache_path = (
            self._cache_dir
            / f"{safe}_{since.date()}_{until.date()}.json"
            if self._cache_dir
            else None
        )
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text())
        else:
            stories: list[dict] = []
            since_i = int(since.replace(tzinfo=timezone.utc).timestamp())
            until_i = int(until.replace(tzinfo=timezone.utc).timestamp())
            for page in range(max_pages):
                # Algolia search supports both bare-string and cashtag forms;
                # cashtag avoids name collisions ($AAPL not "Apple"/"apple")
                params = {
                    "query": f"${ticker.upper()}",
                    "tags": "story",
                    "numericFilters": f"created_at_i>={since_i},created_at_i<{until_i}",
                    "hitsPerPage": per_page,
                    "page": page,
                }
                resp = self._get(params)
                hits = resp.get("hits", [])
                stories.extend(hits)
                if len(hits) < per_page:
                    break
                if page + 1 >= resp.get("nbPages", 0):
                    break
            payload = {"stories": stories}
            if cache_path:
                cache_path.write_text(json.dumps(payload))
        out: list[HNStory] = []
        for h in payload.get("stories", []):
            try:
                created = pd.Timestamp(h.get("created_at"))
            except Exception:
                continue
            out.append(HNStory(
                object_id=str(h.get("objectID", "")),
                ticker=ticker.upper(),
                title=str(h.get("title") or h.get("story_title") or ""),
                points=int(h.get("points") or 0),
                num_comments=int(h.get("num_comments") or 0),
                created_at=created,
                url=str(h.get("url") or ""),
            ))
        return out


def aggregate_to_quarters(stories: list[HNStory]) -> pd.DataFrame:
    """Roll per-story rows up to per-(ticker, quarter_end)."""
    if not stories:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ticker": s.ticker,
        "title": s.title,
        "points": s.points,
        "num_comments": s.num_comments,
        "created_at": s.created_at,
    } for s in stories])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    if df.empty:
        return pd.DataFrame()
    df["quarter_end"] = (
        df["created_at"].dt.tz_convert(None).dt.to_period("Q").dt.end_time.dt.normalize()
    )
    agg = (
        df.groupby(["ticker", "quarter_end"])
        .agg(
            hn_n_stories=("title", "size"),
            hn_mean_points=("points", "mean"),
            hn_mean_num_comments=("num_comments", "mean"),
            hn_joined_titles=("title", lambda s: " || ".join(s.astype(str))[:4000]),
        )
        .reset_index()
    )
    return agg
