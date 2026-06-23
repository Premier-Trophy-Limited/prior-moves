"""Reddit DD scraper using public no-auth endpoints.

Reddit's app-registration flow is gated / unreliable in 2024-2026. This
adapter sidesteps the app creation entirely by hitting two endpoints that
don't require OAuth:

1. **old.reddit.com .json** — append `.json` to any Reddit URL → JSON.
   Public, no auth, ~60 req/min anonymous.
   Used for "live" recent listings (top/new/hot).

2. **PullPush API** — community-run Pushshift successor at
   https://api.pullpush.io. Free, no auth, full historical archive,
   accepts arbitrary time-range filters via `after` / `before`.
   Used for historical backfills (years of DD posts).

Both return JSON with the same per-post fields we care about: id, title,
selftext, score, upvote_ratio, num_comments, created_utc, author.

Subreddits scraped (high DD signal-to-noise):
  r/SecurityAnalysis     long-form value DD
  r/valueinvesting       Buffett/Klarman style
  r/wallstreetbets       momentum + options unusual activity
  r/investing            broad
  r/stocks               broad

Per post we extract:
  cashtag tickers ($AAPL) with 2-5 letter regex
  bullish/bearish lexicon score (count of words from each side)
  is_dd_post flag (title regex)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


SUBS_DEFAULT = (
    "SecurityAnalysis",
    "valueinvesting",
    "wallstreetbets",
    "investing",
    "stocks",
)

BULLISH_TERMS = re.compile(
    r"\b(?:long|buying|added|bought|moon|bullish|undervalued|deep[- ]value|"
    r"breakout|squeeze|conviction|loaded|accumulate)\b",
    re.IGNORECASE,
)
BEARISH_TERMS = re.compile(
    r"\b(?:short|shorting|put|puts|sold|exit|bearish|overvalued|crash|bubble|"
    r"avoid|underperform|distressed)\b",
    re.IGNORECASE,
)
CASHTAG_RE = re.compile(r"\$([A-Z]{2,5})\b")
DD_TITLE_RE = re.compile(r"\bDD\b|due diligence|analysis|thesis|deep dive", re.IGNORECASE)


@dataclass(frozen=True)
class RedditPost:
    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    author: str
    url: str


class RedditPublicClient:
    """No-auth Reddit scraper. Combines old.reddit.com .json + PullPush."""

    def __init__(self, cache_dir: Path | None = None,
                 user_agent: str = "super-investor-mirror/0.1 (educational research)"):
        self._headers = {"User-Agent": user_agent}
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_t = 0.0
        # Anonymous old.reddit cap is ~60 rpm; back off well under that.
        self._min_gap = 60.0 / 30  # 30 rpm safety margin

    def _throttle(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last_t = time.monotonic()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)),
    )
    def _get_json(self, url: str, params: dict | None = None) -> dict:
        self._throttle()
        with httpx.Client(timeout=20.0, headers=self._headers, follow_redirects=True) as c:
            r = c.get(url, params=params)
            if r.status_code == 429:
                time.sleep(10)
                r.raise_for_status()
            r.raise_for_status()
            return r.json()

    def fetch_listing(self, subreddit: str, *, listing: str = "top",
                      time_filter: str = "year", limit: int = 100) -> list[RedditPost]:
        """Public old.reddit.com .json endpoint. Limit max 100 per call."""
        url = f"https://old.reddit.com/r/{subreddit}/{listing}.json"
        params = {"t": time_filter, "limit": limit}
        cache_key = self._cache_dir / "old_reddit" / f"{subreddit}_{listing}_{time_filter}_{limit}.json" \
            if self._cache_dir else None
        data: dict
        if cache_key and cache_key.exists():
            data = json.loads(cache_key.read_text())
        else:
            data = self._get_json(url, params=params)
            if cache_key:
                cache_key.parent.mkdir(parents=True, exist_ok=True)
                cache_key.write_text(json.dumps(data))
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            try:
                created = datetime.utcfromtimestamp(d.get("created_utc", 0))
            except Exception:
                continue
            posts.append(RedditPost(
                id=d.get("id", ""),
                subreddit=subreddit,
                title=d.get("title", ""),
                selftext=(d.get("selftext", "") or "")[:8000],
                score=int(d.get("score", 0) or 0),
                upvote_ratio=float(d.get("upvote_ratio", 0) or 0),
                num_comments=int(d.get("num_comments", 0) or 0),
                created_utc=created,
                author=d.get("author", ""),
                url=f"https://reddit.com{d.get('permalink', '')}",
            ))
        return posts

    def fetch_pullpush_range(self, subreddit: str, since: datetime, until: datetime,
                             size: int = 500) -> list[RedditPost]:
        """PullPush historical archive. Filter by subreddit + after + before."""
        url = "https://api.pullpush.io/reddit/search/submission/"
        params = {
            "subreddit": subreddit,
            "sort": "score",
            "sort_type": "score",
            "size": min(size, 500),  # PullPush hard cap is 500/call
            "after": int(since.timestamp()),
            "before": int(until.timestamp()),
        }
        cache_key = (self._cache_dir / "pullpush" /
                     f"{subreddit}_{since.date()}_{until.date()}.json") if self._cache_dir else None
        data: dict
        if cache_key and cache_key.exists():
            data = json.loads(cache_key.read_text())
        else:
            data = self._get_json(url, params=params)
            if cache_key:
                cache_key.parent.mkdir(parents=True, exist_ok=True)
                cache_key.write_text(json.dumps(data))
        posts = []
        for d in data.get("data", []):
            try:
                created = datetime.utcfromtimestamp(d.get("created_utc", 0))
            except Exception:
                continue
            posts.append(RedditPost(
                id=d.get("id", ""),
                subreddit=subreddit,
                title=d.get("title", ""),
                selftext=(d.get("selftext", "") or "")[:8000],
                score=int(d.get("score", 0) or 0),
                upvote_ratio=float(d.get("upvote_ratio", 0) or 0),
                num_comments=int(d.get("num_comments", 0) or 0),
                created_utc=created,
                author=d.get("author", ""),
                url=f"https://reddit.com/r/{subreddit}/comments/{d.get('id', '')}",
            ))
        return posts

    def pull_all(self, subreddits: tuple[str, ...] = SUBS_DEFAULT,
                 since: datetime | None = None, until: datetime | None = None,
                 listing_time_filter: str = "year",
                 limit_per_listing: int = 100) -> list[RedditPost]:
        """Combined: PullPush for historical (if since provided) + old.reddit for recent top."""
        all_posts: list[RedditPost] = []
        for sub in subreddits:
            try:
                live = self.fetch_listing(sub, listing="top",
                                          time_filter=listing_time_filter,
                                          limit=limit_per_listing)
                all_posts.extend(live)
            except Exception as e:
                print(f"  WARN old.reddit {sub}: {e}")
            if since is not None:
                u = until or datetime.utcnow()
                try:
                    hist = self.fetch_pullpush_range(sub, since, u, size=500)
                    all_posts.extend(hist)
                except Exception as e:
                    print(f"  WARN pullpush {sub}: {e}")
        # Dedupe by id
        by_id = {p.id: p for p in all_posts if p.id}
        return list(by_id.values())


def aggregate_posts_to_ticker_quarter(posts: list[RedditPost],
                                      valid_tickers: set[str]) -> pd.DataFrame:
    """Roll posts up to per-(ticker, subreddit, quarter) aggregates."""
    if not posts:
        return pd.DataFrame()
    rows = []
    for p in posts:
        text = f"{p.title} {p.selftext[:2000]}"
        bull = len(BULLISH_TERMS.findall(text))
        bear = len(BEARISH_TERMS.findall(text))
        is_dd = bool(DD_TITLE_RE.search(p.title))
        for raw in CASHTAG_RE.findall(text):
            ticker = raw.upper()
            if ticker not in valid_tickers:
                continue
            rows.append({
                "ticker": ticker, "subreddit": p.subreddit,
                "created": p.created_utc,
                "score": p.score, "upvote_ratio": p.upvote_ratio,
                "num_comments": p.num_comments,
                "is_dd": is_dd, "bull": bull, "bear": bear,
                "text": p.title,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df["created"])
    df["quarter_end"] = df["created"].dt.to_period("Q").dt.end_time.dt.normalize()
    agg = df.groupby(["ticker", "subreddit", "quarter_end"]).agg(
        n_mentions=("ticker", "size"),
        n_dd_posts=("is_dd", "sum"),
        mean_score=("score", "mean"),
        mean_upvote_ratio=("upvote_ratio", "mean"),
        mean_comments=("num_comments", "mean"),
        bullish_count=("bull", "sum"),
        bearish_count=("bear", "sum"),
        joined_text=("text", lambda s: " || ".join(s)[:4000]),
    ).reset_index()
    return agg
