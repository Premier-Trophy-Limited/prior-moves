"""Reddit DD post scraper — long-form due-diligence threads from investing subs.

Pulls top-scored posts from:
  r/SecurityAnalysis     — long-form value DD
  r/valueinvesting       — Klarman / Buffett style discussion
  r/wallstreetbets       — momentum + meme + options unusual activity
  r/investing            — broad
  r/stocks               — broad

Per (subreddit, ticker, quarter) we emit:
  n_mentions, n_dd_posts, mean_score, mean_upvote_ratio,
  bullish_count, bearish_count, joined_text (for downstream Gemma embedding)

Uses PRAW (Reddit's official Python wrapper). Requires:
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT   (e.g. "super-investor-mirror by /u/yourhandle")

Without creds, the adapter degrades to a no-op that returns an empty
DataFrame — safe to leave wired into the feature pipeline before signup.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


SUBS_DEFAULT = (
    "SecurityAnalysis",
    "valueinvesting",
    "wallstreetbets",
    "investing",
    "stocks",
)

# Bullish / bearish keyword lexicon for cheap polarity scoring (Reddit titles
# are noisy; this is no replacement for a Gemma embedding, but it gives a
# numeric signal that survives without API rate limits).
BULLISH_TERMS = re.compile(
    r"\b(?:long|buying|added|bought|moon|bullish|undervalued|deep value|"
    r"breakout|squeeze|conviction|loaded|accumulate)\b",
    re.IGNORECASE,
)
BEARISH_TERMS = re.compile(
    r"\b(?:short|shorting|put|puts|sold|exit|bearish|overvalued|crash|bubble|"
    r"avoid|underperform|distressed)\b",
    re.IGNORECASE,
)

# Cashtag $XYZ pattern. We accept 2-5 letter symbols (excludes things like $1B)
CASHTAG_RE = re.compile(r"\$([A-Z]{2,5})\b")


@dataclass(frozen=True)
class RedditTickerSnapshot:
    subreddit: str
    ticker: str
    quarter_end: pd.Timestamp
    n_mentions: int
    n_dd_posts: int
    mean_score: float
    mean_upvote_ratio: float
    bullish_count: int
    bearish_count: int
    joined_text: str


class RedditDDClient:
    """Lightweight PRAW wrapper. Falls back to no-op if creds missing."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
    ):
        self._client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self._client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self._user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT",
                                                         "super-investor-mirror/0.1")
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._reddit = None  # lazy-init

    def _ensure_reddit(self):
        if self._reddit is not None:
            return self._reddit
        if not (self._client_id and self._client_secret):
            return None
        import praw  # noqa
        self._reddit = praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent,
            check_for_async=False,
        )
        self._reddit.read_only = True
        return self._reddit

    def pull_subreddit(self, sub: str, since: datetime, until: datetime,
                       limit: int = 500) -> pd.DataFrame:
        reddit = self._ensure_reddit()
        if reddit is None:
            return pd.DataFrame()
        rows: list[dict] = []
        # Reddit search returns max ~1000 results; we use top + new to cover both
        for listing in ("top", "new"):
            sr = reddit.subreddit(sub)
            iter_ = sr.top(time_filter="all", limit=limit) if listing == "top" \
                else sr.new(limit=limit)
            for post in iter_:
                created = datetime.utcfromtimestamp(post.created_utc)
                if created < since or created > until:
                    continue
                rows.append({
                    "subreddit": sub,
                    "post_id": post.id,
                    "created": created,
                    "title": post.title or "",
                    "selftext": (post.selftext or "")[:6000],
                    "score": int(post.score or 0),
                    "upvote_ratio": float(post.upvote_ratio or 0),
                    "num_comments": int(post.num_comments or 0),
                    "url": f"https://reddit.com{post.permalink}",
                })
        df = pd.DataFrame(rows).drop_duplicates("post_id") if rows else pd.DataFrame()
        return df

    def aggregate_to_ticker_quarter(self, posts: pd.DataFrame,
                                    valid_tickers: set[str]) -> pd.DataFrame:
        if posts.empty:
            return pd.DataFrame()
        df = posts.copy()
        df["created"] = pd.to_datetime(df["created"])
        df["quarter_end"] = df["created"].dt.to_period("Q").dt.end_time.dt.normalize()
        df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
        df["bull_score"] = df["full_text"].str.count(BULLISH_TERMS)
        df["bear_score"] = df["full_text"].str.count(BEARISH_TERMS)
        df["is_dd_post"] = df["title"].str.contains(r"DD|due diligence|analysis",
                                                    case=False, regex=True, na=False)
        # Explode by ticker mentioned in title or first 2k chars
        rows: list[dict] = []
        for _, p in df.iterrows():
            for raw in CASHTAG_RE.findall(p["full_text"][:2000]):
                ticker = raw.upper()
                if ticker not in valid_tickers:
                    continue
                rows.append({
                    "ticker": ticker, "subreddit": p["subreddit"],
                    "quarter_end": p["quarter_end"],
                    "score": p["score"], "upvote_ratio": p["upvote_ratio"],
                    "is_dd": bool(p["is_dd_post"]),
                    "bull": int(p["bull_score"]), "bear": int(p["bear_score"]),
                    "text": p["title"],
                })
        if not rows:
            return pd.DataFrame()
        mentions = pd.DataFrame(rows)
        agg = mentions.groupby(["ticker", "subreddit", "quarter_end"]).agg(
            n_mentions=("ticker", "size"),
            n_dd_posts=("is_dd", "sum"),
            mean_score=("score", "mean"),
            mean_upvote_ratio=("upvote_ratio", "mean"),
            bullish_count=("bull", "sum"),
            bearish_count=("bear", "sum"),
            joined_text=("text", lambda s: " || ".join(s)[:4000]),
        ).reset_index()
        return agg
