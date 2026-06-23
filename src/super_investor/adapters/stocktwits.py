"""StockTwits public API adapter — cashtag-tagged retail-sentiment stream.

StockTwits is the original cashtag social network (founded 2008, every post
has a structured ``$TICKER`` mention). Their public REST API exposes recent
messages per ticker without auth (with conservative rate limits).

Endpoint:
  https://api.stocktwits.com/api/2/streams/symbol/<TICKER>.json
  ?max=<message_id>&limit=30

Per message we capture:
  - id (for pagination cursor)
  - created_at
  - body (text)
  - user_followers (popularity weight)
  - entities.sentiment.basic = "Bullish" / "Bearish" / None (user-self-tagged)
  - symbols (cross-ticker mentions, just `symbol` strings)

Per (ticker, quarter_end) we aggregate:
  st_n_messages, st_n_bullish, st_n_bearish, st_mean_followers,
  st_joined_text (lede + body concatenation for Gemma embed)

The walk-back cursor uses `max=<min_id_of_last_page>`; absence of a `max`
hits the most-recent page. Pagination is *backwards in time* — newest first.

Rate limit: ~200 req/hour anonymous. Throttle 18s between calls for safety.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_USER_AGENT = "super-investor-mirror/0.1 research educational"


@dataclass(frozen=True)
class StocktwitsMessage:
    id: int
    ticker: str
    created_at: pd.Timestamp
    body: str
    user_followers: int
    sentiment: str  # "Bullish" / "Bearish" / "" (no tag)


class StocktwitsClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        throttle_sec: float = 18.0,
        max_pages_per_ticker: int = 10,
    ):
        self._headers = {"User-Agent": _USER_AGENT}
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._throttle = throttle_sec
        self._max_pages = max_pages_per_ticker
        self._last_t = 0.0

    def _gate(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < self._throttle:
            time.sleep(self._throttle - gap)
        self._last_t = time.monotonic()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException)
        ),
    )
    def _get_page(self, ticker: str, max_id: int | None = None) -> dict:
        url = _BASE_URL.format(ticker=ticker.upper())
        params: dict = {"limit": 30}
        if max_id is not None:
            params["max"] = max_id
        self._gate()
        with httpx.Client(timeout=15.0, headers=self._headers) as c:
            r = c.get(url, params=params)
            if r.status_code == 429:
                raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
            if r.status_code == 404:
                return {"messages": []}
            r.raise_for_status()
            return r.json()

    def _page_cache_path(self, ticker: str, page_idx: int) -> Path | None:
        if not self._cache_dir:
            return None
        # Sanitize ticker for filesystem (TECK/B etc.)
        safe = ticker.upper().replace("/", "-").replace(" ", "_")
        return self._cache_dir / f"{safe}_p{page_idx:02d}.json"

    def pull_ticker(self, ticker: str) -> list[StocktwitsMessage]:
        """Walk back N pages of messages for a ticker (paginated by max_id)."""
        messages: list[StocktwitsMessage] = []
        max_id: int | None = None
        for page_idx in range(self._max_pages):
            cache_path = self._page_cache_path(ticker, page_idx)
            if cache_path and cache_path.exists():
                page = json.loads(cache_path.read_text())
            else:
                try:
                    page = self._get_page(ticker, max_id=max_id)
                except Exception as e:
                    print(f"  st {ticker} p{page_idx} fail: {e}")
                    break
                if cache_path:
                    cache_path.write_text(json.dumps(page))
            msgs = page.get("messages", []) or []
            if not msgs:
                break
            for m in msgs:
                try:
                    created = pd.Timestamp(m.get("created_at"))
                except Exception:
                    continue
                sentiment = ""
                ent = m.get("entities") or {}
                if isinstance(ent, dict):
                    s = ent.get("sentiment")
                    if isinstance(s, dict):
                        sentiment = s.get("basic") or ""
                user = m.get("user") or {}
                messages.append(StocktwitsMessage(
                    id=int(m.get("id", 0)),
                    ticker=ticker.upper(),
                    created_at=created,
                    body=str(m.get("body", "")),
                    user_followers=int(user.get("followers", 0) or 0),
                    sentiment=str(sentiment),
                ))
            # Walk back via max_id = smallest id this page - 1
            ids = [int(m.get("id", 0)) for m in msgs if m.get("id")]
            if not ids:
                break
            max_id = min(ids) - 1
        return messages


def aggregate_to_quarters(messages: list[StocktwitsMessage]) -> pd.DataFrame:
    """Roll per-message rows up to per-(ticker, quarter_end)."""
    if not messages:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ticker": m.ticker,
        "created_at": m.created_at,
        "body": m.body,
        "user_followers": m.user_followers,
        "sentiment": m.sentiment,
    } for m in messages])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    if df.empty:
        return pd.DataFrame()
    df["quarter_end"] = (
        df["created_at"].dt.tz_convert(None).dt.to_period("Q").dt.end_time.dt.normalize()
    )
    df["is_bull"] = (df["sentiment"] == "Bullish").astype(int)
    df["is_bear"] = (df["sentiment"] == "Bearish").astype(int)
    agg = (
        df.groupby(["ticker", "quarter_end"])
        .agg(
            st_n_messages=("body", "size"),
            st_n_bullish=("is_bull", "sum"),
            st_n_bearish=("is_bear", "sum"),
            st_mean_followers=("user_followers", "mean"),
            st_joined_text=("body", lambda s: " || ".join(s.astype(str))[:4000]),
        )
        .reset_index()
    )
    return agg
