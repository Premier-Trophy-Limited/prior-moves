"""Pocket / Instapaper read-it-later export ingestor.

Pocket export URL: https://getpocket.com/export  (downloads a single HTML
file with `<dl><dt><a href="..." time_added="...">Title</a></dt>`).

Instapaper export URL: https://www.instapaper.com/user  (CSV or HTML).

Treat each saved article as a personal-signal data point: when the user
saved an article matching a ticker, that ticker had user attention that
quarter. Channel prefix ``pk_*``.

Usage:
    from super_investor.adapters.pocket_export import (
        parse_pocket_html, aggregate_to_ticker_quarter,
    )
    items = parse_pocket_html(Path("~/Downloads/ril_export.html"))
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SavedItem:
    url: str
    title: str
    tags: list[str]
    saved_at: pd.Timestamp


_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<![A-Z])([A-Z]{2,5})(?![A-Z])")
_DL_RE = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*time_added="(\d+)"[^>]*(?:tags="([^"]*)")?[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)


def parse_pocket_html(path: Path) -> list[SavedItem]:
    """Pocket HTML export — flat <dl>/<dt>/<a> structure."""
    if not path.exists():
        return []
    txt = path.read_text(errors="replace")
    out: list[SavedItem] = []
    for m in _DL_RE.finditer(txt):
        url, ts_str, tags_str, title = m.groups()
        try:
            ts = pd.Timestamp(int(ts_str), unit="s", tz="UTC")
        except Exception:
            continue
        tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]
        out.append(SavedItem(url=url, title=title.strip(), tags=tags, saved_at=ts))
    return out


def parse_instapaper_csv(path: Path) -> list[SavedItem]:
    """Instapaper CSV columns: URL,Title,Selection,Folder,Timestamp."""
    if not path.exists():
        return []
    out: list[SavedItem] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_raw = row.get("Timestamp") or row.get("time_added")
            try:
                ts = pd.Timestamp(int(ts_raw), unit="s", tz="UTC") if ts_raw and ts_raw.isdigit() else pd.Timestamp(ts_raw, tz="UTC")
            except Exception:
                continue
            tags = []
            folder = (row.get("Folder") or "").strip()
            if folder:
                tags.append(folder)
            out.append(SavedItem(
                url=row.get("URL") or row.get("url") or "",
                title=(row.get("Title") or row.get("title") or "").strip(),
                tags=tags,
                saved_at=ts,
            ))
    return out


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    out: list[str] = []
    for m in _TICKER_RE.finditer(text or ""):
        sym = m.group(1) or m.group(2)
        if sym and sym in universe and sym not in out:
            out.append(sym)
    return out


def aggregate_to_ticker_quarter(
    items: Iterable[SavedItem], universe: set[str]
) -> pd.DataFrame:
    rows = []
    for it in items:
        text = it.title + " " + " ".join(it.tags) + " " + it.url
        tickers = extract_tickers(text, universe)
        if not tickers:
            continue
        q_end = it.saved_at.to_period("Q").end_time
        if q_end.tzinfo is None:
            q_end = q_end.tz_localize("UTC")
        for t in tickers:
            rows.append({
                "ticker": t,
                "quarter_end": q_end,
                "title_len": len(it.title),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["ticker", "quarter_end"], as_index=False).agg(
        pk_n_saves=("title_len", "size"),
        pk_mean_title_len=("title_len", "mean"),
    )
    g["quarter_end"] = pd.to_datetime(g["quarter_end"], utc=True)
    return g
