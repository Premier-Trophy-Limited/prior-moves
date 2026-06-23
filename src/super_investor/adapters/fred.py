"""FRED macro adapter — quarter-end snapshots of regime indicators.

FRED (Federal Reserve Economic Data) exposes free CSVs via:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}

No API key needed for CSV downloads. Daily updates.

Series we pull (high-yield for Druckenmiller/Soros macro-tilt picks):

| series | id | what |
|---|---|---|
| VIX | VIXCLS | implied vol, fear gauge |
| HY-OAS | BAMLH0A0HYM2 | high-yield bond spread vs Treasuries |
| IG-OAS | BAMLC0A0CM | investment-grade spread |
| 10y-3m | T10Y3M | yield curve slope |
| 10y-2y | T10Y2Y | classic recession indicator |
| Real 10y | DFII10 | TIPS yield, real rate |
| USD | DTWEXBGS | broad dollar index |
| Breakeven 10y | T10YIE | inflation expectations |
| Fed funds upper | DFEDTARU | policy rate |
| M2 yoy | M2SL (transformed) | money supply growth |
| Consumer sentiment | UMCSENT | sentiment |
| Industrial production | INDPRO | activity |

Each is fetched and rolled to quarter-end (`.resample('QE').last()`).
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("super_investor.adapters.fred")


FRED_SERIES = {
    # --- regime (original set) ---
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "term_10y_3m": "T10Y3M",
    "term_10y_2y": "T10Y2Y",
    "real_10y": "DFII10",
    "usd_broad": "DTWEXBGS",
    "breakeven_10y": "T10YIE",
    "fed_funds_upper": "DFEDTARU",
    "consumer_sentiment": "UMCSENT",
    "industrial_production": "INDPRO",
    "m2": "M2SL",
    # --- D2: rates / curve ---
    "dgs10": "DGS10",
    "dgs2": "DGS2",
    "dgs3mo": "DGS3MO",
    "sofr": "SOFR",
    # --- D2: credit / financial conditions ---
    "nfci": "NFCI",
    "stlfsi": "STLFSI4",
    "lending_standards": "DRTSCILM",
    # --- D2: FX ---
    "jpy_usd": "DEXJPUS",
    "cny_usd": "DEXCHUS",
    "eur_usd": "DEXUSEU",
    # --- D2: inflation (actuals) ---
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "core_pce": "PCEPILFE",
    "ppi": "PPIACO",
    # --- D2: labor ---
    "unemployment": "UNRATE",
    "payrolls": "PAYEMS",
    "claims": "ICSA",
    "avg_hourly_earnings": "CES0500000003",
    # --- D2: growth / activity ---
    "gdp": "GDPC1",
    "retail_sales": "RSXFS",
    # --- D2: commodity prices (not positioning) ---
    "wti": "DCOILWTICO",
    "brent": "DCOILBRENTEU",
    "natgas": "DHHNGSP",
}

# Level series whose informative form is a 4-quarter (1yr) % change. Each gets a
# derived "<col>_yoy_pct" column. Leak-safe: computed on the quarter-end snapshot,
# never peeking inside the quarter.
YOY_DERIVED = (
    "cpi", "core_cpi", "core_pce", "ppi", "retail_sales", "gdp",
    "payrolls", "avg_hourly_earnings",
)


class FredClient:
    def __init__(self, cache_dir: Path | None = None):
        # Use the realistic browser UA — FRED's CDN sometimes throttles the
        # generic httpx default UA, and our tenacity retry then hangs for
        # 60+s per attempt on TLS connect rather than failing fast.
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                          "super-investor-mirror/0.1",
            "Accept": "text/csv,application/csv,*/*",
        }
        self._last_t: float = 0.0
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        now = time.monotonic()
        if now - self._last_t < 0.1:
            time.sleep(0.1 - (now - self._last_t))
        self._last_t = time.monotonic()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=15))
    def _get_series(self, series_id: str) -> pd.Series:
        cache = (self._cache_dir / f"{series_id}.csv") if self._cache_dir else None
        if cache and cache.exists():
            return _parse_fred_csv(cache.read_text(), series_id)
        self._throttle()
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        # Short connect timeout (FRED CDN sometimes 503-stalls); long read — the
        # full-history daily series (DGS*, DEX*, oil) are large CSVs the CDN can
        # take >20s to stream, which previously tripped ReadTimeout.
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
        with httpx.Client(timeout=timeout, headers=self._headers, follow_redirects=True, http2=False) as c:
            r = c.get(url, params={"id": series_id})
            r.raise_for_status()
            text = r.text
        if cache:
            cache.write_text(text)
        return _parse_fred_csv(text, series_id)

    def quarterly_snapshot(self, start: pd.Timestamp = pd.Timestamp("2014-01-01")) -> pd.DataFrame:
        """One row per quarter-end with all series as columns (leak-safe anchor
        for the backtest). YoY derived over 4 quarters."""
        return self._snapshot(start, freq="QE", index_name="quarter_end", yoy_periods=4)

    def weekly_snapshot(self, start: pd.Timestamp = pd.Timestamp("2018-01-01")) -> pd.DataFrame:
        """One row per week (W, Sun-anchored) with all series as columns — the
        LIVE-cadence frame for the Event Lens. NEVER used in any backtested
        number; it exists so the live lens reads current regime and the
        intra-quarter detector can fire. YoY derived over 52 weeks."""
        return self._snapshot(start, freq="W", index_name="week_end", yoy_periods=52)

    def _snapshot(self, start: pd.Timestamp, freq: str, index_name: str,
                  yoy_periods: int) -> pd.DataFrame:
        frames: list[pd.Series] = []
        for col, sid in FRED_SERIES.items():
            try:
                s = self._get_series(sid)
                s = s.loc[s.index >= start]
                # Last value in the period; forward-fill the weekly frame so it
                # isn't riddled with NaN for monthly/quarterly series between
                # releases (the live lens wants the latest known print).
                q = s.resample(freq).last()
                if freq == "W":
                    q = q.ffill()
                q.name = col
                frames.append(q)
            except Exception as e:
                log.warning("_snapshot(%s/%s, %s): %s: %s", sid, col, freq, type(e).__name__, e)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
        # Derived series (original)
        if "m2" in out.columns:
            out["m2_yoy_pct"] = out["m2"].pct_change(yoy_periods) * 100
        if "industrial_production" in out.columns:
            out["industrial_production_yoy_pct"] = out["industrial_production"].pct_change(yoy_periods) * 100
        # D2 derived: 1yr % change for level series where the change is what matters.
        for col in YOY_DERIVED:
            if col in out.columns:
                out[f"{col}_yoy_pct"] = out[col].pct_change(yoy_periods) * 100
        out.index.name = index_name
        return out.reset_index()


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    # CSV column names vary by version: "DATE,VALUE" or "DATE,VIXCLS" or "observation_date,..."
    date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date")), df.columns[0])
    val_col = next((c for c in df.columns if c != date_col), df.columns[-1])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df[val_col]
