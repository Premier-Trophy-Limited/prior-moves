"""Motley Fool earnings-call transcript archive.

Per-ticker history pages:
  https://www.fool.com/earnings/call-transcripts/?ticker=<sym>

Each transcript page:
  https://www.fool.com/earnings/<year>/<month>/<day>/<sym>-q<n>-<yr>-earnings-call-transcript/

Free, static HTML, no paywall. Channel prefix ``mf_*``.

Yields per (ticker, quarter_end) features: transcript_count, total_word_count,
avg_sentiment_proxy, has_transcript flag.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE = "https://www.fool.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class FoolTranscript:
    url: str
    ticker: str
    title: str
    published_at: pd.Timestamp
    body_text: str = ""
    word_count: int = 0
    quarter_end: pd.Timestamp = field(default=pd.NaT)


def _get(url: str, retries: int = 4, sleep: float = 2.0) -> str:
    last: Exception | None = None
    backoff = sleep
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code in {404, 410}:
                return ""
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(backoff)
    if last is None:
        return ""
    raise RuntimeError(f"GET {url} failed: {last}")


_TRANSCRIPT_LINK_RE = re.compile(
    r'<a[^>]+href="(/earnings/\d{4}/\d{2}/\d{2}/[^"]+-earnings-call-transcript/?)"',
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"/earnings/(\d{4})/(\d{2})/(\d{2})/", re.IGNORECASE)
_QUARTER_RE = re.compile(
    r"-q([1-4])-?(\d{4})-earnings|fiscal[\-\s]?(q[1-4])[\-\s]?(\d{4})",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "").strip()


def list_transcripts_for_ticker(
    ticker: str,
    cache_dir: Path | None = None,
) -> list[str]:
    """Return list of relative URLs for transcripts linked from ticker page."""
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"{BASE}/earnings/call-transcripts/?ticker={safe}"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"_list_{safe}.html"
        if cp.exists():
            html = cp.read_text()
        else:
            html = _get(url)
            if html:
                cp.write_text(html)
    else:
        html = _get(url)
    if not html:
        return []
    hrefs = _TRANSCRIPT_LINK_RE.findall(html)
    seen = set()
    out = []
    for h in hrefs:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def fetch_transcript(rel_url: str, ticker: str, cache_dir: Path | None = None) -> FoolTranscript | None:
    full = f"{BASE}{rel_url}" if rel_url.startswith("/") else rel_url
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = rel_url.rstrip("/").split("/")[-1] + ".html"
        cp = cache_dir / f"{safe}__{fname}"
        if cp.exists():
            html = cp.read_text()
        else:
            html = _get(full)
            if html:
                cp.write_text(html)
    else:
        html = _get(full)
    if not html:
        return None
    dm = _DATE_RE.search(rel_url)
    if not dm:
        return None
    yr, mo, day = dm.groups()
    pub = pd.Timestamp(f"{yr}-{mo}-{day}", tz="UTC")
    # body: strip tags after first <article> if present
    body_html = html
    a_start = html.find("<article")
    if a_start > 0:
        a_end = html.find("</article>", a_start)
        if a_end > 0:
            body_html = html[a_start:a_end]
    body = _clean(body_html)
    body_clean = re.sub(r"\s+", " ", body)
    # quarter
    quarter = pd.NaT
    qm = _QUARTER_RE.search(rel_url + " " + body[:1000])
    if qm:
        groups = [g for g in qm.groups() if g]
        if len(groups) >= 2:
            q = groups[0].lower().lstrip("q")
            y = groups[1]
            md = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}.get(q)
            if md:
                try:
                    quarter = pd.Timestamp(f"{y}-{md}", tz="UTC")
                except Exception:
                    quarter = pd.NaT
    if pd.isna(quarter):
        quarter = pub.to_period("Q").end_time.tz_localize("UTC")
    return FoolTranscript(
        url=full,
        ticker=ticker.upper(),
        title="",
        published_at=pub,
        body_text=body_clean[:50_000],
        word_count=len(body_clean.split()),
        quarter_end=quarter,
    )


def aggregate_transcripts_to_ticker_quarter(
    transcripts: Iterable[FoolTranscript],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for t in transcripts:
        if t.ticker not in universe:
            continue
        if pd.isna(t.quarter_end):
            continue
        rows.append({
            "ticker": t.ticker,
            "quarter_end": t.quarter_end,
            "wc": t.word_count,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "mf_n_transcripts", "mf_total_words", "mf_mean_words",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        mf_n_transcripts=("wc", "size"),
        mf_total_words=("wc", "sum"),
        mf_mean_words=("wc", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
