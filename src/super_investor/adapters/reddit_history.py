"""Reddit historical archive via arctic-shift (pushshift mirror).

PRAW only returns ~1000 most-recent posts per subreddit. arctic-shift exposes
the full 2005-present Reddit archive via a paginated JSON API:

  https://arctic-shift.photon-reddit.com/api/posts/search
  https://arctic-shift.photon-reddit.com/api/comments/search

Usage:
    from super_investor.adapters.reddit_history import (
        ArcticShiftClient,
        aggregate_posts_to_ticker_quarter,
    )

    cli = ArcticShiftClient()
    posts = cli.fetch_subreddit_posts(
        subreddit="SecurityAnalysis",
        after="2021-06-01",
        before="2026-06-01",
        cache_dir=Path("data/reddit_history_cache"),
    )

Output schema mirrors `reddit_dd.py` so the same downstream aggregator works.
Channel prefix: `rh_*` (reddit history) to keep it distinct from recent
`rd_*` PRAW features.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"


@dataclass
class RedditPost:
    post_id: str
    subreddit: str
    title: str
    selftext: str
    created_utc: int
    score: int
    num_comments: int
    url: str
    permalink: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.selftext or ''}"


@dataclass
class ArcticShiftClient:
    base_url: str = ARCTIC_BASE
    timeout: int = 30
    sleep_between_calls: float = 1.0
    user_agent: str = "super-investor-mirror/1.0 (research; +contact@example.com)"

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        after: str,
        before: str,
        cache_dir: Path,
        limit_per_call: int = 100,
        max_total: int = 50_000,
    ) -> list[RedditPost]:
        """Walk arctic-shift posts/search for subreddit between (after, before).

        Persists JSONL to ``cache_dir/<subreddit>.jsonl`` (idempotent — appends).
        Returns the parsed RedditPost objects (in-memory at end).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{subreddit}.jsonl"
        seen_ids = self._load_seen_ids(out_path)
        posts: list[RedditPost] = []
        after_ts = int(pd.Timestamp(after, tz="UTC").timestamp())
        before_ts = int(pd.Timestamp(before, tz="UTC").timestamp())
        cursor = after_ts
        total_new = 0
        with open(out_path, "a", encoding="utf-8") as fh:
            while True:
                params = {
                    "subreddit": subreddit,
                    "after": cursor,
                    "before": before_ts,
                    "limit": limit_per_call,
                    "sort": "asc",
                    "sort_type": "created_utc",
                }
                try:
                    payload = self._get("/posts/search", params)
                except Exception as e:  # pragma: no cover
                    print(f"  arctic-shift error: {e}; sleeping 30s")
                    time.sleep(30)
                    continue
                data = payload.get("data", [])
                if not data:
                    break
                fresh = 0
                for row in data:
                    pid = row.get("id")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    p = RedditPost(
                        post_id=pid,
                        subreddit=row.get("subreddit", subreddit),
                        title=row.get("title", ""),
                        selftext=row.get("selftext", ""),
                        created_utc=int(row.get("created_utc", 0)),
                        score=int(row.get("score", 0) or 0),
                        num_comments=int(row.get("num_comments", 0) or 0),
                        url=row.get("url", ""),
                        permalink=row.get("permalink", ""),
                    )
                    fh.write(json.dumps(row) + "\n")
                    posts.append(p)
                    fresh += 1
                fh.flush()
                total_new += fresh
                last_ts = max(int(r.get("created_utc", cursor)) for r in data)
                print(
                    f"  {subreddit}: cursor={pd.Timestamp(cursor, unit='s', tz='UTC').date()} "
                    f"+{fresh} (total_new={total_new}, total_seen={len(seen_ids)})"
                )
                if last_ts <= cursor:
                    # avoid infinite loop if API returns same window
                    cursor += 1
                else:
                    cursor = last_ts + 1
                if total_new >= max_total:
                    break
                if len(data) < limit_per_call:
                    break
                time.sleep(self.sleep_between_calls)
        return posts

    @staticmethod
    def _load_seen_ids(path: Path) -> set[str]:
        seen: set[str] = set()
        if not path.exists():
            return seen
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line).get("id"))
                except Exception:
                    pass
        return seen


_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<![A-Z])([A-Z]{2,5})(?![A-Z])")


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for m in _TICKER_RE.finditer(text):
        sym = m.group(1) or m.group(2)
        if sym and sym in universe and sym not in hits:
            hits.append(sym)
    return hits


def load_cache_to_posts(cache_dir: Path) -> list[RedditPost]:
    posts: list[RedditPost] = []
    for p in cache_dir.glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                posts.append(
                    RedditPost(
                        post_id=d.get("id", ""),
                        subreddit=d.get("subreddit", p.stem),
                        title=d.get("title", ""),
                        selftext=d.get("selftext", ""),
                        created_utc=int(d.get("created_utc", 0) or 0),
                        score=int(d.get("score", 0) or 0),
                        num_comments=int(d.get("num_comments", 0) or 0),
                        url=d.get("url", ""),
                        permalink=d.get("permalink", ""),
                    )
                )
    return posts


def aggregate_posts_to_ticker_quarter(
    posts: Iterable[RedditPost],
    universe: set[str],
) -> pd.DataFrame:
    """Per-(ticker, quarter_end) features with prefix ``rh_``."""
    rows = []
    for p in posts:
        if not p.created_utc:
            continue
        tickers = extract_tickers(p.text, universe)
        if not tickers:
            continue
        ts = pd.Timestamp(p.created_utc, unit="s", tz="UTC")
        q_end = ts.to_period("Q").end_time.normalize()
        for t in tickers:
            rows.append({
                "ticker": t,
                "quarter_end": q_end,
                "score": p.score,
                "num_comments": p.num_comments,
                "title_len": len(p.title or ""),
                "body_len": len(p.selftext or ""),
                "subreddit": p.subreddit,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "rh_n_posts", "rh_n_unique_subs",
            "rh_score_sum", "rh_score_mean",
            "rh_comments_sum", "rh_text_len_mean",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        rh_n_posts=("score", "size"),
        rh_n_unique_subs=("subreddit", "nunique"),
        rh_score_sum=("score", "sum"),
        rh_score_mean=("score", "mean"),
        rh_comments_sum=("num_comments", "sum"),
        rh_title_len_mean=("title_len", "mean"),
        rh_body_len_mean=("body_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
