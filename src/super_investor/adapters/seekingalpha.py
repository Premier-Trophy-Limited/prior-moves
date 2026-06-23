"""Seeking Alpha DD article + analysis archive via scrapling StealthyFetcher.

Per-ticker analysis pages:
  https://seekingalpha.com/symbol/<sym>/analysis

Each article page contains author byline, publish date, body. Bot-protected
(Cloudflare); scrapling.StealthyFetcher with Camoufox handles it.

Channel prefix ``sa2_*`` (sa already used by stockanalysis).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class SeekingAlphaPost:
    ticker: str
    url: str
    title: str
    published_at: pd.Timestamp
    body_text: str = ""
    word_count: int = 0


def _fetcher():
    """Lazy-import scrapling StealthyFetcher so unrelated paths don't pay cost."""
    from scrapling.fetchers import StealthyFetcher
    return StealthyFetcher


_TAG_RE = re.compile(r"<[^>]+>")


def _clean(html: str) -> str:
    txt = _TAG_RE.sub(" ", html or "")
    return re.sub(r"\s+", " ", txt).strip()


_ARTICLE_LINK_RE = re.compile(r'href="(/article/\d+[^"]*)"', re.IGNORECASE)
_DATE_RE = re.compile(r'<time[^>]+datetime="([^"]+)"', re.IGNORECASE)


def list_articles_for_ticker(
    ticker: str,
    cache_dir: Path | None = None,
    max_pages: int = 3,
) -> list[str]:
    SF = _fetcher()
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        url = f"https://seekingalpha.com/symbol/{safe}/analysis?page={page}"
        cp = cache_dir / f"_list_{safe}_p{page}.html" if cache_dir else None
        html = None
        if cp and cp.exists():
            html = cp.read_text()
        else:
            try:
                resp = SF.fetch(url, headless=True, network_idle=True, timeout=60_000)
                html = resp.html_content if hasattr(resp, "html_content") else str(resp)
                if cp and html:
                    cp.write_text(html)
            except Exception:
                html = ""
        if not html:
            break
        page_links = _ARTICLE_LINK_RE.findall(html)
        if not page_links:
            break
        for h in page_links:
            full = "https://seekingalpha.com" + h
            if full not in urls:
                urls.append(full)
        time.sleep(0.5)
    return urls


def fetch_article(url: str, ticker: str, cache_dir: Path | None = None) -> SeekingAlphaPost | None:
    SF = _fetcher()
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    aid = url.rstrip("/").split("/")[-1]
    cp = cache_dir / f"{safe}__{aid}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    html = None
    if cp and cp.exists():
        html = cp.read_text()
    else:
        try:
            resp = SF.fetch(url, headless=True, network_idle=True, timeout=60_000)
            html = resp.html_content if hasattr(resp, "html_content") else str(resp)
            if cp and html:
                cp.write_text(html)
        except Exception:
            return None
    if not html:
        return None
    dm = _DATE_RE.search(html)
    try:
        pub = pd.Timestamp(dm.group(1)) if dm else pd.NaT
        if not pd.isna(pub) and pub.tzinfo is None:
            pub = pub.tz_localize("UTC")
    except Exception:
        pub = pd.NaT
    body = _clean(html)[:50_000]
    return SeekingAlphaPost(
        ticker=ticker.upper(),
        url=url,
        title="",
        published_at=pub,
        body_text=body,
        word_count=len(body.split()),
    )


def aggregate_to_ticker_quarter(
    posts: Iterable[SeekingAlphaPost],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for p in posts:
        if p.ticker not in universe:
            continue
        if pd.isna(p.published_at):
            continue
        q_end = p.published_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        rows.append({
            "ticker": p.ticker,
            "quarter_end": q_end,
            "wc": p.word_count,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "sa2_n_articles", "sa2_total_words", "sa2_mean_words",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        sa2_n_articles=("wc", "size"),
        sa2_total_words=("wc", "sum"),
        sa2_mean_words=("wc", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
