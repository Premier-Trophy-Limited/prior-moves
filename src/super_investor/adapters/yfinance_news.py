"""yfinance news + recommendations adapter.

yfinance exposes (free):
- Ticker.news: list of recent headlines with provider + publish time
- Ticker.recommendations: historical analyst rating changes
- Ticker.upgrades_downgrades: discrete upgrade/downgrade events
- Ticker.analyst_price_targets: current targets (mean/high/low/median)
- Ticker.calendar: next earnings date

Per (ticker, quarter) we emit numeric counts + a packed text blob for embedding:

    n_news_30d, n_upgrades_30d, n_downgrades_30d,
    price_target_dispersion, mean_price_target_premium,
    headlines_concatenated (text for Gemma embed)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TickerNewsSnapshot:
    ticker: str
    quarter_end: pd.Timestamp
    n_news_30d: int
    n_upgrades_30d: int
    n_downgrades_30d: int
    mean_price_target_premium_pct: float
    price_target_dispersion: float
    headlines_text: str  # joined "title - publisher" lines


class YfinanceNewsClient:
    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, ticker: str, quarter_end: pd.Timestamp) -> TickerNewsSnapshot:
        import yfinance as yf
        t = yf.Ticker(ticker)
        window_start = quarter_end - pd.Timedelta(days=30)

        # News
        try:
            news = t.news or []
        except Exception:
            news = []
        recent_news = []
        for item in news:
            content = item.get("content") if isinstance(item.get("content"), dict) else item
            ts = content.get("pubDate") or content.get("providerPublishTime")
            if ts is None:
                continue
            try:
                t_news = pd.Timestamp(ts, unit="s") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            except Exception:
                continue
            if t_news.tzinfo is not None:
                t_news = t_news.tz_localize(None)
            if window_start <= t_news <= quarter_end:
                title = content.get("title", "")
                publisher = content.get("provider", {}).get("displayName", "") \
                    if isinstance(content.get("provider"), dict) else content.get("publisher", "")
                recent_news.append((t_news, title, publisher))

        headlines_text = "\n".join(
            f"{ts.date()} {pub}: {title}" for ts, title, pub in sorted(recent_news)
        )

        # Recommendations: count upgrades/downgrades in window
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                ud.index = pd.to_datetime(ud.index)
                if ud.index.tz is not None:
                    ud.index = ud.index.tz_localize(None)
                window = ud[(ud.index >= window_start) & (ud.index <= quarter_end)]
                n_up = int((window.get("Action", pd.Series()) == "up").sum())
                n_down = int((window.get("Action", pd.Series()) == "down").sum())
            else:
                n_up = n_down = 0
        except Exception:
            n_up = n_down = 0

        # Price targets
        try:
            pt = t.analyst_price_targets or {}
            mean = float(pt.get("mean") or 0)
            high = float(pt.get("high") or 0)
            low = float(pt.get("low") or 0)
            current = float(pt.get("current") or 0)
            mean_premium_pct = ((mean - current) / current * 100) if current > 0 else 0.0
            dispersion = ((high - low) / mean) if mean > 0 else 0.0
        except Exception:
            mean_premium_pct = 0.0
            dispersion = 0.0

        return TickerNewsSnapshot(
            ticker=ticker,
            quarter_end=quarter_end,
            n_news_30d=len(recent_news),
            n_upgrades_30d=n_up,
            n_downgrades_30d=n_down,
            mean_price_target_premium_pct=mean_premium_pct,
            price_target_dispersion=dispersion,
            headlines_text=headlines_text,
        )
