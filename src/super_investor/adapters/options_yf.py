"""Per-ticker options sentiment via yfinance — put/call + IV skew.

Free (yfinance). Current snapshot only (no history), so this is a
forward-looking signal for the current quarter. Channel prefix ``opt_``.

Per ticker, across the nearest 1-3 expiries:
  opt_pc_oi_ratio   put open-interest / call open-interest  (>1 = bearish skew)
  opt_pc_vol_ratio  put volume / call volume
  opt_iv_skew       avg put IV - avg call IV (positive = downside fear)
  opt_total_oi      total open interest (options-market interest)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger("super_investor.adapters.options_yf")


@dataclass
class OptSnapshot:
    ticker: str
    pc_oi_ratio: float
    pc_vol_ratio: float
    iv_skew: float
    total_oi: float
    quarter_end: pd.Timestamp


def fetch_ticker(ticker: str, max_expiries: int = 3, cache_dir: Path | None = None) -> OptSnapshot | None:
    import json
    safe = ticker.upper().replace("/", "-").replace(".", "-")
    cp = cache_dir / f"{safe}.json" if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists():
        try:
            d = json.loads(cp.read_text())
            return OptSnapshot(
                ticker=safe, pc_oi_ratio=d["pc_oi_ratio"], pc_vol_ratio=d["pc_vol_ratio"],
                iv_skew=d["iv_skew"], total_oi=d["total_oi"],
                quarter_end=pd.Timestamp(d["quarter_end"]),
            )
        except Exception:
            pass
    try:
        import yfinance as yf
        t = yf.Ticker(safe)
        expiries = list(t.options or [])[:max_expiries]
        if not expiries:
            return None
        call_oi = put_oi = call_vol = put_vol = 0.0
        call_iv: list[float] = []
        put_iv: list[float] = []
        for e in expiries:
            try:
                ch = t.option_chain(e)
            except Exception:
                continue
            c, p = ch.calls, ch.puts
            call_oi += float(c["openInterest"].fillna(0).sum())
            put_oi += float(p["openInterest"].fillna(0).sum())
            call_vol += float(c["volume"].fillna(0).sum())
            put_vol += float(p["volume"].fillna(0).sum())
            call_iv += c["impliedVolatility"].dropna().tolist()
            put_iv += p["impliedVolatility"].dropna().tolist()
        if call_oi + put_oi == 0:
            return None
        pc_oi = put_oi / call_oi if call_oi > 0 else float("nan")
        pc_vol = put_vol / call_vol if call_vol > 0 else float("nan")
        skew = (
            (sum(put_iv) / len(put_iv)) - (sum(call_iv) / len(call_iv))
            if put_iv and call_iv else float("nan")
        )
        q = pd.Timestamp.utcnow().to_period("Q").end_time.tz_localize("UTC")
        snap = OptSnapshot(
            ticker=safe, pc_oi_ratio=pc_oi, pc_vol_ratio=pc_vol,
            iv_skew=skew, total_oi=call_oi + put_oi, quarter_end=q,
        )
        if cp:
            cp.write_text(json.dumps({
                "pc_oi_ratio": pc_oi, "pc_vol_ratio": pc_vol, "iv_skew": skew,
                "total_oi": call_oi + put_oi, "quarter_end": str(q),
            }))
        return snap
    except Exception as e:
        log.warning("fetch_ticker(%s): %s: %s", ticker, type(e).__name__, e)
        return None


def aggregate(snaps: Iterable[OptSnapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        if s is None:
            continue
        rows.append({
            "ticker": s.ticker, "quarter_end": s.quarter_end,
            "opt_pc_oi_ratio": s.pc_oi_ratio, "opt_pc_vol_ratio": s.pc_vol_ratio,
            "opt_iv_skew": s.iv_skew, "opt_total_oi": s.total_oi,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True)
    return df.drop_duplicates(["ticker", "quarter_end"])
