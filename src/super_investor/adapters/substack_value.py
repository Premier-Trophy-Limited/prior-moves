"""Substack RSS pull for value-investor / financial substacks.

Each substack has an RSS feed at ``<sub>.substack.com/feed``. Titles + summaries
mention tickers. Channel prefix ``sb_*``.

Curated list (5+ year publishers; expand at will):
  - doomberg          (geopolitics/energy macro)
  - netinterest       (Marc Rubinstein, financials)
  - thebearcave       (short ideas)
  - notboring         (Packy McCormick, big-picture tech)
  - moneystuff        (Matt Levine via Bloomberg — RSS via fool-bberg? deferred)
  - lillianli         (value picks)
  - kyla              (macro)
  - stratechery       (mirror — paid, skip for now)
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

DEFAULT_SUBS = [
    "doomberg",
    "netinterest",
    "thebearcave",
    "notboring",
    "lillianli",
    "kylascanlon",
    "valuesteward",
    "tsoh",  # The Science of Hitting
    "growthandvalue",
    "moontower",
]


@dataclass
class SubstackPost:
    sub: str
    title: str
    link: str
    published_at: pd.Timestamp
    body: str


def fetch_feed(sub: str, cache_dir: Path | None = None) -> list[SubstackPost]:
    url = f"https://{sub}.substack.com/feed"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"{sub}.xml"
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code in {200, 304}:
                cp.write_bytes(r.content)
                xml = r.content
            else:
                xml = cp.read_bytes() if cp.exists() else b""
        except Exception:
            xml = cp.read_bytes() if cp.exists() else b""
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            xml = r.content
        except Exception:
            xml = b""
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    posts: list[SubstackPost] = []
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
        posts.append(SubstackPost(sub=sub, title=title, link=link, published_at=pub, body=body))
    return posts


_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<![A-Z])([A-Z]{2,5})(?![A-Z])")


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "")


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    out: list[str] = []
    for m in _TICKER_RE.finditer(text or ""):
        sym = m.group(1) or m.group(2)
        if sym and sym in universe and sym not in out:
            out.append(sym)
    return out


def aggregate_posts_to_ticker_quarter(
    posts: Iterable[SubstackPost],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for p in posts:
        if pd.isna(p.published_at):
            continue
        text = f"{p.title}\n{_clean(p.body)}"
        tickers = extract_tickers(text, universe)
        if not tickers:
            continue
        q_end = p.published_at.to_period("Q").end_time.tz_localize("UTC") if p.published_at.tzinfo is None else p.published_at.to_period("Q").end_time
        for t in tickers:
            rows.append({
                "ticker": t,
                "quarter_end": q_end,
                "sub": p.sub,
                "body_len": len(text),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end",
            "sb_n_mentions", "sb_n_unique_subs", "sb_mean_body_len",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        sb_n_mentions=("sub", "size"),
        sb_n_unique_subs=("sub", "nunique"),
        sb_mean_body_len=("body_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
