"""finviz.com — static fundamental snapshots + recent news per ticker.

Per-ticker page: https://finviz.com/quote.ashx?t=<sym>
Snapshot table has 72 fundamental keys (P/E, ROE, Inst Own %, Short Float %,
Avg Vol, Beta, RSI, SMA20/50/200 relative, etc.).

Channel prefix ``fv_*``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


def _get(url: str, retries: int = 2) -> str:
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(0.8)
    raise RuntimeError(f"GET {url} failed: {last}")


_PAIR_RE = re.compile(
    r'<td[^>]*class="[^"]*snapshot[^"]*"[^>]*>([^<]+)</td>\s*'
    r'<td[^>]*class="[^"]*snapshot[^"]*"[^>]*>(.*?)</td>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return _TAG_RE.sub("", s or "").replace("&nbsp;", " ").strip()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_num(s: str) -> float | None:
    if not s or s in {"-", "—"}:
        return None
    s = s.replace("%", "").replace(",", "").replace("$", "")
    if s.endswith("B"):
        try:
            return float(s[:-1]) * 1e9
        except Exception:
            return None
    if s.endswith("M"):
        try:
            return float(s[:-1]) * 1e6
        except Exception:
            return None
    if s.endswith("K"):
        try:
            return float(s[:-1]) * 1e3
        except Exception:
            return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


KEEP_KEYS = {
    "P/E": "pe",
    "Forward P/E": "fwd_pe",
    "PEG": "peg",
    "P/S": "ps",
    "P/B": "pb",
    "P/C": "pc",
    "P/FCF": "pfcf",
    "EPS (ttm)": "eps_ttm",
    "EPS this Y": "eps_this_y",
    "EPS next Y": "eps_next_y",
    "EPS Q/Q": "eps_qq",
    "Sales Q/Q": "sales_qq",
    "Inst Own": "inst_own",
    "Inst Trans": "inst_trans",
    "Short Float": "short_float",
    "Short Ratio": "short_ratio",
    "Profit Margin": "prof_margin",
    "Oper Margin": "op_margin",
    "Gross Margin": "gross_margin",
    "ROE": "roe",
    "ROA": "roa",
    "ROI": "roi",
    "Beta": "beta",
    "RSI (14)": "rsi14",
    "Recom": "recom",
    "Target Price": "target_price",
    "Perf Week": "perf_w",
    "Perf Month": "perf_m",
    "Perf Quarter": "perf_q",
    "Perf Half Y": "perf_hy",
    "Perf Year": "perf_y",
    "Volatility": "vol",
    "Avg Volume": "avg_vol",
}


def fetch_snapshot(ticker: str, cache_dir: Path | None = None) -> dict | None:
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    url = f"https://finviz.com/quote.ashx?t={safe}&p=d"
    cp = cache_dir / f"{safe}.html" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists():
        html = cp.read_text()
    else:
        html = _get(url)
        if cp and html:
            cp.write_text(html)
    if not html:
        return None
    snap: dict[str, float | None] = {"ticker": safe}
    keys_iter = iter(_PAIR_RE.findall(html))
    for k_raw, v_raw in keys_iter:
        k = _strip(k_raw)
        if k in KEEP_KEYS:
            col = KEEP_KEYS[k]
            snap[f"fv_{col}"] = _to_num(_strip(v_raw))
    return snap


def fetch_many(tickers: Iterable[str], cache_dir: Path | None = None, sleep: float = 0.5) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            r = fetch_snapshot(t, cache_dir=cache_dir)
            if r:
                rows.append(r)
        except Exception:
            pass
        time.sleep(sleep)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # finviz is a snapshot — stamp current quarter
    q_end = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
    df["quarter_end"] = q_end
    return df


def aggregate_to_ticker_quarter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
