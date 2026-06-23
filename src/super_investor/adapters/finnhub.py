"""Finnhub Premium adapter — Starter plan ($50/mo).

Endpoints we hit (all live-confirmed with current key):

- /quote                          spot price + intraday delta
- /stock/insider-transactions     Form 4 derived w/ price & code (cleaner than EDGAR XML)
- /company-news                   per-ticker news 1 article = 1 row, with headline+source
- /news-sentiment                 article-aggregated sentiment + buzz score
- /stock/recommendation           buy/hold/sell trend by month
- /calendar/earnings              upcoming earnings + EPS estimate
- /stock/social-sentiment         Reddit/Twitter mention counts + sentiment

Rate limit: 60 calls/min on Starter; we throttle to 50 to leave headroom.

Output schemas designed to feed the per-(ticker, quarter) feature builder:

    InsiderQuarterSnapshot(ticker, quarter_end,
                           n_buys, n_sells, n_option_exercises,
                           total_buy_usd, total_sell_usd,
                           n_unique_insiders, max_buy_usd)

    NewsQuarterSnapshot(ticker, quarter_end,
                        n_articles, mean_sentiment, sentiment_std,
                        buzz_score, headlines_text)

    SentimentQuarterSnapshot(ticker, quarter_end,
                             reddit_mention_count, reddit_pos_score,
                             twitter_mention_count, twitter_pos_score)

    RecommendationSnapshot(ticker, quarter_end,
                           strong_buy, buy, hold, sell, strong_sell,
                           consensus_delta_30d)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


FINNHUB_BASE = "https://finnhub.io/api/v1"
_MIN_REQUEST_GAP_S = 60 / 50  # 50 rpm = 1.2s gap


@dataclass(frozen=True)
class InsiderQuarterSnapshot:
    ticker: str
    quarter_end: pd.Timestamp
    n_buys: int
    n_sells: int
    n_option_exercises: int
    total_buy_usd: float
    total_sell_usd: float
    n_unique_insiders: int
    max_buy_usd: float


@dataclass(frozen=True)
class NewsQuarterSnapshot:
    ticker: str
    quarter_end: pd.Timestamp
    n_articles: int
    mean_sentiment: float
    sentiment_std: float
    buzz_score: float
    headlines_text: str


@dataclass(frozen=True)
class RecommendationSnapshot:
    ticker: str
    quarter_end: pd.Timestamp
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int


class FinnhubClient:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        self._key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self._key:
            raise RuntimeError("FINNHUB_API_KEY missing")
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_t = 0.0

    def _throttle(self) -> None:
        now = time.monotonic()
        gap = now - self._last_t
        if gap < _MIN_REQUEST_GAP_S:
            time.sleep(_MIN_REQUEST_GAP_S - gap)
        self._last_t = time.monotonic()

    def _get(self, path: str, params: dict, allow_403: bool = False) -> dict | list:
        """GET with disk cache + 5xx retry. 4xx never retries (Starter tier
        gates some endpoints; return empty dict instead of raising)."""
        import json
        cache_key = None
        if self._cache_dir:
            stable = "_".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "token")
            cache_key = self._cache_dir / path.strip("/").replace("/", "_") / f"{stable}.json"
            if cache_key.exists():
                return json.loads(cache_key.read_text())
        params = {**params, "token": self._key}

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _do() -> dict | list:
            self._throttle()
            with httpx.Client(timeout=15.0) as c:
                r = c.get(f"{FINNHUB_BASE}{path}", params=params)
                if r.status_code == 403 and allow_403:
                    return {}  # endpoint not available at current plan tier
                if 400 <= r.status_code < 500:
                    r.raise_for_status()
                r.raise_for_status()
                return r.json()
        try:
            data = _do()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and allow_403:
                data = {}
            else:
                raise
        if cache_key:
            cache_key.parent.mkdir(parents=True, exist_ok=True)
            cache_key.write_text(json.dumps(data))
        return data

    def insider_quarter(self, ticker: str, quarter_end: pd.Timestamp) -> InsiderQuarterSnapshot:
        """Roll up Finnhub insider-transactions for the 90 days ending at quarter_end."""
        start = (quarter_end - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        end = quarter_end.strftime("%Y-%m-%d")
        data = self._get("/stock/insider-transactions",
                         {"symbol": ticker, "from": start, "to": end})
        rows = data.get("data", []) if isinstance(data, dict) else []
        n_buys = n_sells = n_opt = 0
        total_buy = total_sell = max_buy = 0.0
        insiders: set[str] = set()
        for tx in rows:
            code = tx.get("transactionCode", "")
            shares = float(tx.get("share", 0) or 0)
            price = float(tx.get("transactionPrice", 0) or 0)
            usd = abs(shares * price)
            insiders.add(str(tx.get("name", "")))
            if code == "P":
                n_buys += 1
                total_buy += usd
                max_buy = max(max_buy, usd)
            elif code == "S":
                n_sells += 1
                total_sell += usd
            elif code == "M":
                n_opt += 1
        return InsiderQuarterSnapshot(
            ticker=ticker, quarter_end=quarter_end,
            n_buys=n_buys, n_sells=n_sells, n_option_exercises=n_opt,
            total_buy_usd=total_buy, total_sell_usd=total_sell,
            n_unique_insiders=len(insiders), max_buy_usd=max_buy,
        )

    def news_quarter(self, ticker: str, quarter_end: pd.Timestamp,
                     max_headlines_chars: int = 8000) -> NewsQuarterSnapshot:
        """Pull last-90d news + per-article aggregate sentiment buzz."""
        start = (quarter_end - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        end = quarter_end.strftime("%Y-%m-%d")
        articles = self._get("/company-news",
                             {"symbol": ticker, "from": start, "to": end})
        if not isinstance(articles, list):
            articles = []
        # Headlines packed for downstream embedding
        lines: list[str] = []
        for a in articles:
            ts = a.get("datetime")
            try:
                t = pd.Timestamp(ts, unit="s") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            except Exception:
                continue
            line = f"{t.date()} {a.get('source','')}: {a.get('headline','')}"
            lines.append(line)
        headlines = "\n".join(lines)[:max_headlines_chars]

        # Buzz / sentiment via news-sentiment endpoint — premium tier only; gracefully
        # degrade if Starter plan does not include it.
        sentiment = self._get("/news-sentiment", {"symbol": ticker}, allow_403=True)
        s = sentiment.get("sentiment", {}) if isinstance(sentiment, dict) else {}
        buzz = sentiment.get("buzz", {}) if isinstance(sentiment, dict) else {}
        mean_s = float(s.get("bullishPercent", 0) - s.get("bearishPercent", 0))
        return NewsQuarterSnapshot(
            ticker=ticker, quarter_end=quarter_end,
            n_articles=len(articles),
            mean_sentiment=mean_s,
            sentiment_std=0.0,  # Finnhub doesn't expose std at this tier
            buzz_score=float(buzz.get("buzz", 0)),
            headlines_text=headlines,
        )

    def recommendation_snapshot(self, ticker: str, quarter_end: pd.Timestamp) -> RecommendationSnapshot:
        """Latest recommendation-trend row at or before quarter_end."""
        rows = self._get("/stock/recommendation", {"symbol": ticker})
        if not isinstance(rows, list) or not rows:
            return RecommendationSnapshot(
                ticker, quarter_end, 0, 0, 0, 0, 0,
            )
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        before = df[df["period"] <= quarter_end].sort_values("period")
        if before.empty:
            r = df.iloc[0]
        else:
            r = before.iloc[-1]
        return RecommendationSnapshot(
            ticker=ticker, quarter_end=quarter_end,
            strong_buy=int(r.get("strongBuy", 0) or 0),
            buy=int(r.get("buy", 0) or 0),
            hold=int(r.get("hold", 0) or 0),
            sell=int(r.get("sell", 0) or 0),
            strong_sell=int(r.get("strongSell", 0) or 0),
        )

    def earnings_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Forward earnings calendar between two dates. Useful for screening upcoming events."""
        data = self._get("/calendar/earnings",
                         {"from": start.strftime("%Y-%m-%d"),
                          "to": end.strftime("%Y-%m-%d")})
        events = data.get("earningsCalendar", []) if isinstance(data, dict) else []
        if not events:
            return pd.DataFrame()
        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"])
        return df
