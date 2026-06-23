"""Aswath Damodaran blog (musingsonmarkets.blogspot.com) — valuation commentary.

Blogspot exposes a full Atom feed at:
  https://aswathdamodaran.blogspot.com/feeds/posts/default?max-results=500

500-post window covers ~5-7 years given his cadence. Channel prefix ``dm_*``.
"""
from __future__ import annotations

import re
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
FEED = "https://aswathdamodaran.blogspot.com/feeds/posts/default?max-results=500"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<![A-Z])([A-Z]{2,5})(?![A-Z])")


@dataclass
class DamodaranPost:
    title: str
    link: str
    published_at: pd.Timestamp
    body: str


def _clean(s: str) -> str:
    return _TAG_RE.sub(" ", s or "")


def fetch_feed(cache_dir: Path | None = None) -> list[DamodaranPost]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / "damodaran.xml"
    else:
        cp = None
    try:
        r = requests.get(FEED, headers={"User-Agent": UA}, timeout=30)
        if r.status_code == 200:
            xml = r.content
            if cp:
                cp.write_bytes(xml)
        else:
            xml = cp.read_bytes() if cp and cp.exists() else b""
    except Exception:
        xml = cp.read_bytes() if cp and cp.exists() else b""
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    posts: list[DamodaranPost] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        link_el = entry.find('atom:link[@rel="alternate"]', ATOM_NS)
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        pub_raw = entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
        body = entry.findtext("atom:content", default="", namespaces=ATOM_NS) or ""
        try:
            pub = pd.Timestamp(pd.to_datetime(pub_raw, utc=True))
        except Exception:
            pub = pd.NaT
        posts.append(DamodaranPost(title=title, link=link, published_at=pub, body=_clean(body)[:50_000]))
    return posts


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    out: list[str] = []
    for m in _TICKER_RE.finditer(text or ""):
        sym = m.group(1) or m.group(2)
        if sym and sym in universe and sym not in out:
            out.append(sym)
    return out


def aggregate_to_ticker_quarter(
    posts: Iterable[DamodaranPost], universe: set[str]
) -> pd.DataFrame:
    rows = []
    for p in posts:
        if pd.isna(p.published_at):
            continue
        text = f"{p.title}\n{p.body}"
        tickers = extract_tickers(text, universe)
        if not tickers:
            continue
        q_end = p.published_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        for t in tickers:
            rows.append({
                "ticker": t,
                "quarter_end": q_end,
                "body_len": len(text),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "quarter_end", "dm_n_mentions", "dm_mean_body_len",
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        dm_n_mentions=("body_len", "size"),
        dm_mean_body_len=("body_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
