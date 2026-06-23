"""Multi-source financial news RSS aggregator.

Feeds (all free, no auth):
  - WSJ Markets:           https://feeds.a.dj.com/rss/RSSMarketsMain.xml
  - WSJ Business:          https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml
  - MarketWatch top:       https://feeds.content.dowjones.io/public/rss/mw_topstories
  - MarketWatch markets:   https://feeds.content.dowjones.io/public/rss/mw_marketpulse
  - Reuters business:      https://feeds.reuters.com/reuters/businessNews
  - Reuters markets:       https://feeds.reuters.com/reuters/USMarketsNews
  - CNBC top news:         https://www.cnbc.com/id/100003114/device/rss/rss.html
  - Investing.com:         https://www.investing.com/rss/news.rss
  - Yahoo Finance topstories: https://finance.yahoo.com/news/rssindex
  - Bloomberg Matt Levine: https://www.bloomberg.com/opinion/authors/ARbTQlRLRjE/matt-levine.rss

Each feed: pull RSS → extract title + body snippet + published_at → match
tickers from $CASHTAG + company-name matcher → per-(ticker, quarter)
aggregate.

Channel prefix: ``nw_*`` (news).
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("super_investor.adapters.news_rss")


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

FEEDS: list[tuple[str, str]] = [
    ("wsj_markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("wsj_business", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"),
    ("mw_top", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("mw_markets", "https://feeds.content.dowjones.io/public/rss/mw_marketpulse"),
    ("reuters_biz", "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
    ("cnbc_top", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("yf_top", "https://finance.yahoo.com/news/rssindex"),
    ("seekingalpha_news", "https://seekingalpha.com/feed.xml"),
    ("matt_levine", "https://www.bloomberg.com/opinion/authors/ARbTQlRLRjE/matt-levine.rss"),
    ("kiplinger", "https://www.kiplinger.com/feed/all"),
    ("zerohedge", "https://feeds.feedburner.com/zerohedge/feed"),
    ("benzinga", "https://www.benzinga.com/feed"),
]


@dataclass
class NewsItem:
    source: str
    title: str
    link: str
    published_at: pd.Timestamp
    body: str


_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<![A-Z])([A-Z]{2,5})(?![A-Z])")


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "")


def fetch_feed(source: str, url: str, cache_dir: Path | None = None) -> list[NewsItem]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"{source}.xml"
    else:
        cp = None
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code == 200:
            xml = r.content
            if cp:
                cp.write_bytes(xml)
        else:
            xml = cp.read_bytes() if cp and cp.exists() else b""
    except Exception as e:
        log.warning("fetch_feed(%s): %s: %s", url, type(e).__name__, e)
        xml = cp.read_bytes() if cp and cp.exists() else b""
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    posts: list[NewsItem] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        body = ""
        for tag in ("description", "{http://purl.org/rss/1.0/modules/content/}encoded"):
            t = item.find(tag)
            if t is not None and t.text:
                body = t.text
                break
        try:
            pub = pd.Timestamp(pd.to_datetime(pub_raw, utc=True))
        except Exception:
            pub = pd.NaT
        posts.append(NewsItem(source=source, title=title, link=link, published_at=pub, body=body))
    return posts


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    out: list[str] = []
    for m in _TICKER_RE.finditer(text or ""):
        sym = m.group(1) or m.group(2)
        if sym and sym in universe and sym not in out:
            out.append(sym)
    return out


def fetch_all_feeds(cache_dir: Path | None = None) -> list[NewsItem]:
    all_posts: list[NewsItem] = []
    for src, url in FEEDS:
        try:
            posts = fetch_feed(src, url, cache_dir=cache_dir)
            all_posts.extend(posts)
            print(f"  {src:18s} +{len(posts)}", flush=True)
        except Exception as e:
            log.warning("fetch_all_feeds(%s): %s: %s", src, type(e).__name__, e)
        time.sleep(0.3)
    return all_posts


def aggregate_to_ticker_quarter(
    items: Iterable[NewsItem],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for it in items:
        if pd.isna(it.published_at):
            continue
        text = f"{it.title}\n{_clean(it.body)}"
        tickers = extract_tickers(text, universe)
        if not tickers:
            continue
        q_end = it.published_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        for t in tickers:
            rows.append({
                "ticker": t,
                "quarter_end": q_end,
                "source": it.source,
                "body_len": len(text),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "nw_n_mentions", "nw_n_unique_sources", "nw_mean_body_len",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        nw_n_mentions=("source", "size"),
        nw_n_unique_sources=("source", "nunique"),
        nw_mean_body_len=("body_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
