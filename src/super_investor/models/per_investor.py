"""Per-investor LightGBM models — separate fit per investor.

Hypothesis: a single shared model averages across very different investor
styles. Per-investor models let value-tier features dominate where they
matter (Buffett) while quant signals dominate where they matter (Renaissance).

Each investor gets its own train / val / holdout split (temporal). Features
join the same macro + Finnhub backfill (where the per-(ticker, quarter) data
exists) and fall back to macro-only when Finnhub is missing.

Output (per investor):
    runs/per_investor/<slug>/metrics.json
    runs/per_investor/<slug>/feature_importance.json
    runs/per_investor/<slug>/holdout_predictions.parquet

Top-N picks emission (the iterative-compare loop hook):
    runs/per_investor/<slug>/picks_<quarter>.parquet
    one row per (ticker, p_new_entry), sorted descending
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


REPO = Path(__file__).resolve().parents[3]
HOLDOUT_START = pd.Timestamp("2017-09-30")  # ~35 holdout quarters, all 25 investors (post-backfill)
VAL_QUARTERS = 4

# "Prioritize 2026" — recency-weight training rows so the current regime
# dominates. Each row's weight decays by half every RECENCY_HALF_LIFE_YEARS of
# age relative to the prediction date. A 2-year half-life means a filing from
# 2024 counts ~half as much as one from 2026, and a 2018 filing ~1/16. Set to
# a large number (e.g. 1e9) to disable (uniform weights).
RECENCY_HALF_LIFE_YEARS = 4.0
RECENCY_MIN_WEIGHT = 0.25  # floor so ancient rows still contribute meaningfully


def recency_weights(
    periods: pd.Series,
    ref_date: pd.Timestamp,
    half_life_years: float = RECENCY_HALF_LIFE_YEARS,
    min_weight: float = RECENCY_MIN_WEIGHT,
) -> np.ndarray:
    """Exponential-decay sample weights by row age relative to ``ref_date``.

    weight = max(min_weight, 0.5 ** (age_years / half_life_years)),
    clipped to [min_weight, 1.0]. Future-dated rows (shouldn't happen in a
    walk-forward window) get weight 1.0.
    """
    p = pd.to_datetime(periods)
    age_days = (pd.Timestamp(ref_date) - p).dt.days.clip(lower=0).to_numpy(dtype=float)
    age_years = age_days / 365.25
    w = np.power(0.5, age_years / float(half_life_years))
    return np.clip(w, min_weight, 1.0)


# ---------------------------------------------------------------------------
# Per-investor channel selection
# ---------------------------------------------------------------------------
# Adding 218 features helps some investors and hurts others — FINRA short
# interest is gold for short-bias funds but noise for Buffett. Each investor
# style gets a curated channel allowlist. Channels not on the allowlist are
# dropped from feature_cols (their coverage flag is also dropped).
#
# An entry of "ALL" means use every channel. Default for unknown investors
# is "ALL" (back-compat).

CHANNEL_ALL = "ALL"

# Style archetypes — channels each style cares about
STYLE_CHANNELS: dict[str, list[str]] = {
    # Value: fundamentals + super-investor mirror + insider + activist.
    # PRUNED 2026-06-03: dropped the pure-social channels (rh/hn/wp/sa) — they
    # added columns but hurt AUC for fundamentals-driven value investors
    # (greenblatt/buffett dipped at the full 257-feature space). Keep
    # structured/numeric + reputable-press channels only.
    # ny/gd/avg = NYT/Guardian/AlphaVantage news — reputable, kept across styles.
    "value": [
        "fh", "f4", "av", "yh", "e8k", "cg",
        "dr", "rf", "oi", "ad", "dm", "fv",
        "ny", "gd", "avg",
        "xf",
        "inst",
        "gl",
    ],
    # Activist / event-driven — flow events dominate
    "activist": [
        "f4", "e8k", "ad", "dr", "oi", "rf", "fh", "cg", "av",
        "ny", "gd", "avg",
        "xf",
        "inst",
        "gl",
    ],
    # Macro — top-down with sentiment context
    "macro": [
        "yh", "e8k", "cg", "nw", "dm", "aa", "sh", "fv",
        "ny", "gd", "avg",
        "xf",
        "inst",
        "gl",
        "opt",
    ],
    # Growth / momentum — sentiment + DD heavy. Social channels legitimately
    # carry signal for momentum names, but the long tail (wp/sb) is noise;
    # keep the higher-signal social (hn/rh/st/tr) + numeric.
    "growth": [
        "hn", "rh", "st", "av", "dm", "nw", "tr", "fv", "f4",
        "ny", "gd", "avg",
        "xf",
        "inst",
        "gl",
        "opt",
    ],
    # Quant — PRUNED 2026-06-03 from CHANNEL_ALL. "Everything goes" overfit:
    # renaissance/two_sigma AUC dipped at the full 257-feature space. A
    # gradient-boosted model with a few hundred rows per investor cannot use
    # 257 columns without memorizing. Curated to the dense, structured,
    # numeric channels (fundamentals + flow + breadth + options + reputable
    # news tone); the pure-chatter social channels (st/hn/rh/wp/pk/sb/sa/nw)
    # are dropped.
    "quant": [
        "fh", "f4", "av", "yh", "e8k", "cg",
        "dr", "fv", "dm", "aa", "rf", "oi", "ad", "sh", "tr",
        "ny", "gd", "avg",
        "xf", "inst", "opt", "gl",
    ],
    # Catalyst / short-bias — risk-factor + short interest + activist
    "catalyst": [
        "sh", "rf", "ad", "f4", "e8k", "oi", "fh", "hn",
        "ny", "gd", "avg",
        "xf",
        "inst",
        "gl",
        "opt",
    ],
}

# Per-investor style assignment
INVESTOR_STYLE: dict[str, str] = {
    "buffett": "value",
    "klarman": "value",
    "greenblatt": "value",
    "einhorn": "value",
    "tepper": "value",
    "ackman": "activist",
    "loeb": "activist",
    "icahn": "activist",
    "burry": "catalyst",
    "soros": "macro",
    "druckenmiller": "macro",
    "dalio": "macro",
    "renaissance": "quant",
    "two_sigma": "quant",
    "des": "quant",
    "wood": "growth",
    "cathie_wood": "growth",
}


def channels_for_investor(slug: str) -> list[str]:
    """Return list of channel prefixes (without trailing '_') for one investor.

    Returns ``[CHANNEL_ALL]`` for any unknown slug — back-compatible with the
    original "use everything" behavior.
    """
    style = INVESTOR_STYLE.get(slug.lower())
    if not style:
        return [CHANNEL_ALL]
    return STYLE_CHANNELS.get(style, [CHANNEL_ALL])


# Base channels that ALWAYS stay regardless of style (macro + label scaffolding)
ALWAYS_KEEP_PREFIXES: tuple[str, ...] = ("emb_", "macro_")


def filter_feature_cols(
    feature_cols: list[str],
    cat_cols: list[str],
    slug: str,
) -> tuple[list[str], list[str]]:
    """Drop columns whose channel prefix isn't on the investor's allowlist.

    Always keeps: ``base_cols`` (cusip_id, year, q_index, all macro_*),
    Gemma text embedding (``emb_*``), and the matching ``has_<x>_coverage``
    flags.
    """
    allowed = channels_for_investor(slug)
    if CHANNEL_ALL in allowed:
        return feature_cols, cat_cols
    allowed_set = set(allowed)
    base = {"cusip_id", "year", "q_index"}

    def _keep(col: str) -> bool:
        if col in base:
            return True
        if col.startswith(ALWAYS_KEEP_PREFIXES):
            return True
        if not any(col.startswith(f"{p}_") for p in allowed_set | {"has"}):
            # not on allowlist and not a coverage flag — drop
            return False
        if col.startswith("has_"):
            chan = col[4:].split("_coverage")[0]
            return chan in allowed_set
        chan = col.split("_", 1)[0]
        return chan in allowed_set

    kept_features = [c for c in feature_cols if _keep(c)]
    kept_cats = [c for c in cat_cols if c in kept_features]
    return kept_features, kept_cats


def _load_inputs(repo: Path = REPO) -> dict[str, pd.DataFrame]:
    labels = pd.read_parquet(repo / "data" / "13f" / "labels.parquet")
    labels["period_of_report"] = pd.to_datetime(labels["period_of_report"])

    macro = pd.read_parquet(repo / "data" / "features" / "macro_quarterly.parquet")
    macro["quarter_end"] = pd.to_datetime(macro["quarter_end"])

    cusip_map_path = repo / "data" / "tickers" / "cusip_to_ticker.parquet"
    cusip_map = pd.read_parquet(cusip_map_path)[["cusip", "ticker"]].dropna() \
        if cusip_map_path.exists() else pd.DataFrame(columns=["cusip", "ticker"])

    fh_insider_path = repo / "data" / "features" / "finnhub_insider.parquet"
    fh_news_path = repo / "data" / "features" / "finnhub_news.parquet"
    fh_rec_path = repo / "data" / "features" / "finnhub_recommendations.parquet"
    fh_insider = pd.read_parquet(fh_insider_path) if fh_insider_path.exists() else pd.DataFrame()
    fh_news = pd.read_parquet(fh_news_path) if fh_news_path.exists() else pd.DataFrame()
    fh_rec = pd.read_parquet(fh_rec_path) if fh_rec_path.exists() else pd.DataFrame()

    # PCA-reduced Gemma text embeddings (one row per ticker × quarter)
    text_path = repo / "data" / "features" / "text_embeddings_pca.parquet"
    text_emb = pd.read_parquet(text_path) if text_path.exists() else pd.DataFrame()

    # FT Alphaville per-(ticker, quarter) aggregates + Gemma-PCA embedding
    av_quarterly_path = repo / "data" / "features" / "alphaville_quarterly.parquet"
    av_quarterly = pd.read_parquet(av_quarterly_path) if av_quarterly_path.exists() else pd.DataFrame()
    av_emb_path = repo / "data" / "features" / "alphaville_embeddings_pca.parquet"
    av_emb = pd.read_parquet(av_emb_path) if av_emb_path.exists() else pd.DataFrame()

    # yfinance historical (quarterly fundamentals + analyst rating events + dividends)
    yh_path = repo / "data" / "features" / "yfinance_quarterly.parquet"
    yh = pd.read_parquet(yh_path) if yh_path.exists() else pd.DataFrame()

    # SEC 8-K item-code counts per quarter
    e8k_path = repo / "data" / "features" / "edgar_8k_quarterly.parquet"
    e8k = pd.read_parquet(e8k_path) if e8k_path.exists() else pd.DataFrame()

    # Congressional PTR aggregates per ticker × quarter
    cg_path = repo / "data" / "features" / "congress_quarterly.parquet"
    cg = pd.read_parquet(cg_path) if cg_path.exists() else pd.DataFrame()

    # Hacker News cashtag mentions per quarter
    hn_path = repo / "data" / "features" / "hackernews_quarterly.parquet"
    hn = pd.read_parquet(hn_path) if hn_path.exists() else pd.DataFrame()

    # StockTwits sentiment per quarter
    st_path = repo / "data" / "features" / "stocktwits_quarterly.parquet"
    st = pd.read_parquet(st_path) if st_path.exists() else pd.DataFrame()

    # Form 4 insider aggregates (EDGAR direct; free, broader coverage than Finnhub)
    form4_path = repo / "data" / "features" / "form4_quarterly.parquet"
    form4 = pd.read_parquet(form4_path) if form4_path.exists() else pd.DataFrame()

    # DataRoma super-investor portfolio aggregates (dr_* channel)
    dr_path = repo / "data" / "features" / "dataroma_quarterly.parquet"
    dr = pd.read_parquet(dr_path) if dr_path.exists() else pd.DataFrame()

    # stockanalysis.com quarterly fundamentals (sa_* channel)
    sa_path = repo / "data" / "features" / "stockanalysis_quarterly.parquet"
    sa = pd.read_parquet(sa_path) if sa_path.exists() else pd.DataFrame()

    # Substack value-investor mentions (sb_* channel)
    sb_path = repo / "data" / "features" / "substack_quarterly.parquet"
    sb = pd.read_parquet(sb_path) if sb_path.exists() else pd.DataFrame()

    # Reddit historical (arctic-shift) per-ticker quarterly aggregates (rh_*)
    rh_path = repo / "data" / "features" / "reddit_history_quarterly.parquet"
    rh = pd.read_parquet(rh_path) if rh_path.exists() else pd.DataFrame()

    # finviz snapshot (fv_*)
    fv_path = repo / "data" / "features" / "finviz_quarterly.parquet"
    fv = pd.read_parquet(fv_path) if fv_path.exists() else pd.DataFrame()

    # News RSS aggregator (nw_* — WSJ/MarketWatch/CNBC/YF/etc)
    nw_path = repo / "data" / "features" / "news_rss_quarterly.parquet"
    nw = pd.read_parquet(nw_path) if nw_path.exists() else pd.DataFrame()

    # Damodaran blog feed (dm_*)
    dm_path = repo / "data" / "features" / "damodaran_quarterly.parquet"
    dm = pd.read_parquet(dm_path) if dm_path.exists() else pd.DataFrame()

    # AAII sentiment macro-level (aa_*)
    aa_path = repo / "data" / "features" / "aaii_quarterly.parquet"
    aa = pd.read_parquet(aa_path) if aa_path.exists() else pd.DataFrame()

    # EDGAR 10-K/10-Q risk-factors aggregates (rf_*)
    rf_path = repo / "data" / "features" / "edgar_riskfactors_quarterly.parquet"
    rf = pd.read_parquet(rf_path) if rf_path.exists() else pd.DataFrame()

    # TipRanks analyst consensus (tr_*)
    tr_path = repo / "data" / "features" / "tipranks_quarterly.parquet"
    tr = pd.read_parquet(tr_path) if tr_path.exists() else pd.DataFrame()

    # OpenInsider insider trades (oi_*) — alt to Finnhub/Form 4
    oi_path = repo / "data" / "features" / "openinsider_quarterly.parquet"
    oi = pd.read_parquet(oi_path) if oi_path.exists() else pd.DataFrame()

    # SEC SC 13D/13G activist disclosures (ad_*)
    ad_path = repo / "data" / "features" / "sec_13d_quarterly.parquet"
    ad = pd.read_parquet(ad_path) if ad_path.exists() else pd.DataFrame()

    # FINRA short-interest daily aggregated (sh_*)
    sh_path = repo / "data" / "features" / "finra_quarterly.parquet"
    sh = pd.read_parquet(sh_path) if sh_path.exists() else pd.DataFrame()

    # Wikipedia per-article monthly pageviews (wp_*)
    wp_path = repo / "data" / "features" / "wikipedia_quarterly.parquet"
    wp = pd.read_parquet(wp_path) if wp_path.exists() else pd.DataFrame()

    # NYT Article Search (ny_*) — requires NYT_API_KEY
    ny_path = repo / "data" / "features" / "nyt_quarterly.parquet"
    ny = pd.read_parquet(ny_path) if ny_path.exists() else pd.DataFrame()

    # Guardian Open Platform (gd_*) — requires GUARDIAN_API_KEY
    gd_path = repo / "data" / "features" / "guardian_quarterly.parquet"
    gd = pd.read_parquet(gd_path) if gd_path.exists() else pd.DataFrame()

    # Alpha Vantage news + earnings (avg_*) — requires ALPHAVANTAGE_API_KEY
    avg_path = repo / "data" / "features" / "alpha_vantage_quarterly.parquet"
    avg = pd.read_parquet(avg_path) if avg_path.exists() else pd.DataFrame()

    # Pocket/Instapaper personal read-it-later (pk_*)
    pk_path = repo / "data" / "features" / "pocket_quarterly.parquet"
    pk = pd.read_parquet(pk_path) if pk_path.exists() else pd.DataFrame()

    # SEC XBRL companyfacts — real quarterly fundamentals, whole universe (xf_*)
    xf_path = repo / "data" / "features" / "sec_xbrl_quarterly.parquet"
    xf = pd.read_parquet(xf_path) if xf_path.exists() else pd.DataFrame()

    # SEC 13F total institutional breadth — n_filers + flow per ticker (inst_*)
    inst_path = repo / "data" / "features" / "sec_13f_breadth_quarterly.parquet"
    inst = pd.read_parquet(inst_path) if inst_path.exists() else pd.DataFrame()

    # Options sentiment — put/call + IV skew via yfinance (opt_*)
    opt_path = repo / "data" / "features" / "options_quarterly.parquet"
    opt = pd.read_parquet(opt_path) if opt_path.exists() else pd.DataFrame()

    # GDELT news-tone per company (gl_*)
    gl_path = repo / "data" / "features" / "gdelt_quarterly.parquet"
    gl = pd.read_parquet(gl_path) if gl_path.exists() else pd.DataFrame()

    # Reddit DD aggregates (per ticker × subreddit × quarter). Pivot to wide
    # subreddit columns so each ticker × quarter has columns like
    # rd_wallstreetbets_score, rd_securityanalysis_n_mentions, etc.
    reddit_path = repo / "data" / "features" / "reddit_quarterly.parquet"
    reddit = pd.read_parquet(reddit_path) if reddit_path.exists() else pd.DataFrame()
    if len(reddit):
        # Sum across subreddits to give one row per (ticker, quarter)
        reddit_agg = reddit.groupby(["ticker", "quarter_end"]).agg(
            rd_n_mentions=("n_mentions", "sum"),
            rd_n_dd=("n_dd_posts", "sum"),
            rd_mean_score=("mean_score", "mean"),
            rd_bull=("bullish_count", "sum"),
            rd_bear=("bearish_count", "sum"),
        ).reset_index()
    else:
        reddit_agg = pd.DataFrame()

    return {
        "labels": labels,
        "macro": macro,
        "cusip_map": cusip_map,
        "fh_insider": fh_insider,
        "fh_news": fh_news,
        "fh_rec": fh_rec,
        "text_emb": text_emb,
        "form4": form4,
        "reddit": reddit_agg,
        "av_quarterly": av_quarterly,
        "av_emb": av_emb,
        "yh": yh,
        "e8k": e8k,
        "hn": hn,
        "st": st,
        "cg": cg,
        "dr": dr,
        "sa_fin": sa,
        "sb": sb,
        "rh": rh,
        "fv": fv,
        "nw": nw,
        "dm": dm,
        "aa": aa,
        "rf": rf,
        "tr": tr,
        "oi": oi,
        "ad": ad,
        "sh": sh,
        "wp": wp,
        "ny": ny,
        "gd": gd,
        "avg": avg,
        "pk": pk,
        "xf": xf,
        "inst": inst,
        "opt": opt,
        "gl": gl,
    }


def _attach_features(labels: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], list[str]]:
    macro = inputs["macro"].set_index("quarter_end")
    cusip_map = inputs["cusip_map"]
    fh_insider = inputs["fh_insider"]
    fh_news = inputs["fh_news"]
    fh_rec = inputs["fh_rec"]

    df = labels.copy()
    df["y"] = (df["label"] == "new_entry").astype(np.int8)
    df["year"] = df["period_of_report"].dt.year
    df["q_index"] = df["period_of_report"].dt.quarter
    df["cusip_id"] = df["cusip"].astype("category").cat.codes

    # CUSIP → ticker
    if len(cusip_map):
        df = df.merge(cusip_map, on="cusip", how="left")
    else:
        df["ticker"] = None

    # Macro join
    df = df.join(macro, on="period_of_report", how="left")

    # Finnhub joins (left, so most rows have NaN for backfilled subset)
    if len(fh_insider):
        fhi = fh_insider.rename(columns={"quarter_end": "period_of_report"})
        fhi = fhi.add_prefix("fh_ins_").rename(columns={"fh_ins_ticker": "ticker", "fh_ins_period_of_report": "period_of_report"})
        df = df.merge(fhi, on=["ticker", "period_of_report"], how="left")
    if len(fh_news):
        fhn = fh_news.rename(columns={"quarter_end": "period_of_report"})
        fhn = fhn.drop(columns=["headlines_text"], errors="ignore")
        fhn = fhn.add_prefix("fh_news_").rename(columns={"fh_news_ticker": "ticker", "fh_news_period_of_report": "period_of_report"})
        df = df.merge(fhn, on=["ticker", "period_of_report"], how="left")
    if len(fh_rec):
        fhr = fh_rec.rename(columns={"quarter_end": "period_of_report"})
        fhr = fhr.add_prefix("fh_rec_").rename(columns={"fh_rec_ticker": "ticker", "fh_rec_period_of_report": "period_of_report"})
        df = df.merge(fhr, on=["ticker", "period_of_report"], how="left")

    # Gemma text-embedding PCA features
    text_emb = inputs.get("text_emb", pd.DataFrame())
    if len(text_emb):
        te = text_emb.rename(columns={"quarter_end": "period_of_report"})
        emb_cols = [c for c in te.columns if c.startswith("emb_")]
        te = te[["ticker", "period_of_report"] + emb_cols]
        df = df.merge(te, on=["ticker", "period_of_report"], how="left")

    # Form 4 EDGAR aggregates
    form4 = inputs.get("form4", pd.DataFrame())
    if len(form4):
        f4 = form4.rename(columns={"quarter_end": "period_of_report"})
        f4 = f4.add_prefix("f4_").rename(columns={"f4_ticker": "ticker", "f4_period_of_report": "period_of_report"})
        df = df.merge(f4, on=["ticker", "period_of_report"], how="left")

    # Reddit DD aggregates
    reddit = inputs.get("reddit", pd.DataFrame())
    if len(reddit):
        rd = reddit.rename(columns={"quarter_end": "period_of_report"})
        df = df.merge(rd, on=["ticker", "period_of_report"], how="left")

    # FT Alphaville per-(ticker, quarter) aggregates
    av_quarterly = inputs.get("av_quarterly", pd.DataFrame())
    if len(av_quarterly):
        av = av_quarterly.rename(columns={"quarter_end": "period_of_report"})
        # Drop the raw blob — model only consumes numeric + the PCA embedding
        av = av.drop(columns=["joined_text"], errors="ignore")
        av = av.add_prefix("av_").rename(
            columns={"av_ticker": "ticker", "av_period_of_report": "period_of_report"}
        )
        df = df.merge(av, on=["ticker", "period_of_report"], how="left")

    # FT Alphaville PCA embeddings — prefix with av_emb_ to avoid colliding
    # with the Finnhub-news PCA columns (emb_0..emb_N).
    av_emb = inputs.get("av_emb", pd.DataFrame())
    if len(av_emb):
        ae = av_emb.rename(columns={"quarter_end": "period_of_report"})
        ae_emb_cols = [c for c in ae.columns if c.startswith("emb_")]
        ae = ae[["ticker", "period_of_report"] + ae_emb_cols]
        ae = ae.rename(columns={c: f"av_{c}" for c in ae_emb_cols})
        df = df.merge(ae, on=["ticker", "period_of_report"], how="left")

    # yfinance historical fundamentals + analyst events (yh_ prefix already on cols)
    yh = inputs.get("yh", pd.DataFrame())
    if len(yh):
        yhd = yh.rename(columns={"quarter_end": "period_of_report"})
        df = df.merge(yhd, on=["ticker", "period_of_report"], how="left")

    # SEC 8-K item-code counts (e8k_ prefix already on cols)
    e8k = inputs.get("e8k", pd.DataFrame())
    if len(e8k):
        e8kd = e8k.rename(columns={"quarter_end": "period_of_report"})
        df = df.merge(e8kd, on=["ticker", "period_of_report"], how="left")

    # Hacker News cashtag aggregates (hn_ prefix on cols, drop the text blob)
    hn = inputs.get("hn", pd.DataFrame())
    if len(hn):
        hnd = hn.drop(columns=["hn_joined_titles"], errors="ignore").rename(
            columns={"quarter_end": "period_of_report"}
        )
        df = df.merge(hnd, on=["ticker", "period_of_report"], how="left")

    # StockTwits sentiment counts (st_ prefix on cols, drop the text blob)
    st = inputs.get("st", pd.DataFrame())
    if len(st):
        std = st.drop(columns=["st_joined_text"], errors="ignore").rename(
            columns={"quarter_end": "period_of_report"}
        )
        df = df.merge(std, on=["ticker", "period_of_report"], how="left")

    # Congressional PTR aggregates (cg_ prefix already on cols)
    cg = inputs.get("cg", pd.DataFrame())
    if len(cg):
        cgd = cg.rename(columns={"quarter_end": "period_of_report"})
        df = df.merge(cgd, on=["ticker", "period_of_report"], how="left")

    def _strip_tz(d: pd.DataFrame) -> pd.DataFrame:
        """Normalize quarter_end timestamps to naive midnight for merge."""
        out = d.copy()
        ts = pd.to_datetime(out["period_of_report"], utc=True, errors="coerce")
        # snap to the date (drop sub-day precision) so to_period(Q).end_time's
        # 23:59:59.999999 markers align with labels' midnight values
        out["period_of_report"] = ts.dt.tz_convert(None).dt.normalize()
        return out

    # DataRoma per-(ticker, quarter) aggregates (dr_ prefix already on cols)
    dr = inputs.get("dr", pd.DataFrame())
    if len(dr):
        drd = _strip_tz(dr.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(drd, on=["ticker", "period_of_report"], how="left")

    # stockanalysis.com quarterly fundamentals (sa_ prefix already on cols)
    sa_fin = inputs.get("sa_fin", pd.DataFrame())
    if len(sa_fin):
        sad = _strip_tz(sa_fin.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(sad, on=["ticker", "period_of_report"], how="left")

    # Substack value-investor mentions (sb_ prefix already on cols)
    sb = inputs.get("sb", pd.DataFrame())
    if len(sb):
        sbd = _strip_tz(sb.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(sbd, on=["ticker", "period_of_report"], how="left")

    # Reddit historical (arctic-shift, rh_ prefix already on cols)
    rh = inputs.get("rh", pd.DataFrame())
    if len(rh):
        rhd = _strip_tz(rh.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(rhd, on=["ticker", "period_of_report"], how="left")

    # finviz snapshot fundamentals (fv_ prefix already on cols)
    fv = inputs.get("fv", pd.DataFrame())
    if len(fv):
        fvd = _strip_tz(fv.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(fvd, on=["ticker", "period_of_report"], how="left")

    # News RSS aggregator (nw_* — WSJ/MarketWatch/etc.)
    nw = inputs.get("nw", pd.DataFrame())
    if len(nw):
        nwd = _strip_tz(nw.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(nwd, on=["ticker", "period_of_report"], how="left")

    # Damodaran blog feed (dm_*)
    dm = inputs.get("dm", pd.DataFrame())
    if len(dm):
        dmd = _strip_tz(dm.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(dmd, on=["ticker", "period_of_report"], how="left")

    # AAII macro-level sentiment (aa_*) — same row per ticker per quarter
    aa = inputs.get("aa", pd.DataFrame())
    if len(aa):
        aad = aa.rename(columns={"quarter_end": "period_of_report"}).copy()
        aad["period_of_report"] = pd.to_datetime(
            aad["period_of_report"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
        df = df.merge(aad, on="period_of_report", how="left")

    # EDGAR 10-K/10-Q risk-factor stats (rf_*)
    rf = inputs.get("rf", pd.DataFrame())
    if len(rf):
        rfd = _strip_tz(rf.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(rfd, on=["ticker", "period_of_report"], how="left")

    # TipRanks analyst consensus (tr_*)
    tr = inputs.get("tr", pd.DataFrame())
    if len(tr):
        trd = _strip_tz(tr.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(trd, on=["ticker", "period_of_report"], how="left")

    # OpenInsider insider trades (oi_*)
    oi = inputs.get("oi", pd.DataFrame())
    if len(oi):
        oid = _strip_tz(oi.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(oid, on=["ticker", "period_of_report"], how="left")

    # SEC SC 13D/13G activist disclosures (ad_*)
    ad = inputs.get("ad", pd.DataFrame())
    if len(ad):
        add = _strip_tz(ad.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(add, on=["ticker", "period_of_report"], how="left")

    # FINRA short interest (sh_*)
    sh = inputs.get("sh", pd.DataFrame())
    if len(sh):
        shd = _strip_tz(sh.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(shd, on=["ticker", "period_of_report"], how="left")

    # Wikipedia pageviews (wp_*)
    wp = inputs.get("wp", pd.DataFrame())
    if len(wp):
        wpd = _strip_tz(wp.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(wpd, on=["ticker", "period_of_report"], how="left")

    # NYT Article Search (ny_*)
    ny = inputs.get("ny", pd.DataFrame())
    if len(ny):
        nyd = _strip_tz(ny.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(nyd, on=["ticker", "period_of_report"], how="left")

    # Guardian Open Platform (gd_*)
    gd = inputs.get("gd", pd.DataFrame())
    if len(gd):
        gdd = _strip_tz(gd.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(gdd, on=["ticker", "period_of_report"], how="left")

    # Alpha Vantage news + earnings (avg_*)
    avg = inputs.get("avg", pd.DataFrame())
    if len(avg):
        avgd = _strip_tz(avg.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(avgd, on=["ticker", "period_of_report"], how="left")

    # Pocket/Instapaper saved items (pk_*)
    pk = inputs.get("pk", pd.DataFrame())
    if len(pk):
        pkd = _strip_tz(pk.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(pkd, on=["ticker", "period_of_report"], how="left")

    # SEC XBRL fundamentals (xf_*)
    xf = inputs.get("xf", pd.DataFrame())
    if len(xf):
        xfd = _strip_tz(xf.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(xfd, on=["ticker", "period_of_report"], how="left")

    # SEC 13F institutional breadth (inst_*)
    inst = inputs.get("inst", pd.DataFrame())
    if len(inst):
        instd = _strip_tz(inst.rename(columns={"quarter_end": "period_of_report"}))
        instd = instd.drop(columns=["cusip"], errors="ignore")
        df = df.merge(instd, on=["ticker", "period_of_report"], how="left")

    # Options sentiment (opt_*)
    opt = inputs.get("opt", pd.DataFrame())
    if len(opt):
        optd = _strip_tz(opt.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(optd, on=["ticker", "period_of_report"], how="left")

    # GDELT news-tone (gl_*)
    gl = inputs.get("gl", pd.DataFrame())
    if len(gl):
        gld = _strip_tz(gl.rename(columns={"quarter_end": "period_of_report"}))
        df = df.merge(gld, on=["ticker", "period_of_report"], how="left")

    macro_cols = list(macro.columns)
    base_cols = ["cusip_id", "year", "q_index"] + macro_cols
    fh_cols = [c for c in df.columns if c.startswith("fh_")]
    emb_feature_cols = [c for c in df.columns if c.startswith("emb_") and not c.startswith("av_emb_")]
    f4_cols = [c for c in df.columns if c.startswith("f4_")]
    rd_cols = [c for c in df.columns if c.startswith("rd_")]
    av_cols = [c for c in df.columns if c.startswith("av_")]
    yh_cols = [c for c in df.columns if c.startswith("yh_")]
    e8k_cols = [c for c in df.columns if c.startswith("e8k_")]
    hn_cols = [c for c in df.columns if c.startswith("hn_") and not c.startswith("hn_joined")]
    st_cols = [c for c in df.columns if c.startswith("st_") and not c.startswith("st_joined")]
    cg_cols = [c for c in df.columns if c.startswith("cg_")]
    dr_cols = [c for c in df.columns if c.startswith("dr_")]
    sa_cols = [c for c in df.columns if c.startswith("sa_")]
    sb_cols = [c for c in df.columns if c.startswith("sb_")]
    rh_cols = [c for c in df.columns if c.startswith("rh_")]
    fv_cols = [c for c in df.columns if c.startswith("fv_")]
    nw_cols = [c for c in df.columns if c.startswith("nw_")]
    dm_cols = [c for c in df.columns if c.startswith("dm_")]
    aa_cols = [c for c in df.columns if c.startswith("aa_")]
    rf_cols = [c for c in df.columns if c.startswith("rf_")]
    tr_cols = [c for c in df.columns if c.startswith("tr_")]
    oi_cols = [c for c in df.columns if c.startswith("oi_")]
    ad_cols = [c for c in df.columns if c.startswith("ad_")]
    sh_cols = [c for c in df.columns if c.startswith("sh_")]
    wp_cols = [c for c in df.columns if c.startswith("wp_")]
    ny_cols = [c for c in df.columns if c.startswith("ny_")]
    gd_cols = [c for c in df.columns if c.startswith("gd_")]
    avg_cols = [c for c in df.columns if c.startswith("avg_")]
    pk_cols = [c for c in df.columns if c.startswith("pk_")]
    xf_cols = [c for c in df.columns if c.startswith("xf_")]
    inst_cols = [c for c in df.columns if c.startswith("inst_")]
    opt_cols = [c for c in df.columns if c.startswith("opt_")]
    gl_cols = [c for c in df.columns if c.startswith("gl_")]

    # Coverage flags — binary "this row had any data from channel X". Without
    # these, sparse channels (HN, 8-K, ST) confuse the model on tickers they
    # don't cover, since NaN gets learned as a meaningful negative signal.
    # The flag lets the model condition: "if has_hn_coverage=0 ignore all
    # hn_* features for this prediction."
    def _has_any(cols: list[str]) -> pd.Series:
        if not cols:
            return pd.Series(0, index=df.index, dtype=np.int8)
        return df[cols].notna().any(axis=1).astype(np.int8)

    df["has_fh_coverage"] = _has_any(fh_cols)
    df["has_emb_coverage"] = _has_any(emb_feature_cols)
    df["has_f4_coverage"] = _has_any(f4_cols)
    df["has_rd_coverage"] = _has_any(rd_cols)
    df["has_av_coverage"] = _has_any(av_cols)
    df["has_yh_coverage"] = _has_any(yh_cols)
    df["has_e8k_coverage"] = _has_any(e8k_cols)
    df["has_hn_coverage"] = _has_any(hn_cols)
    df["has_st_coverage"] = _has_any(st_cols)
    df["has_cg_coverage"] = _has_any(cg_cols)
    df["has_dr_coverage"] = _has_any(dr_cols)
    df["has_sa_coverage"] = _has_any(sa_cols)
    df["has_sb_coverage"] = _has_any(sb_cols)
    df["has_rh_coverage"] = _has_any(rh_cols)
    df["has_fv_coverage"] = _has_any(fv_cols)
    df["has_nw_coverage"] = _has_any(nw_cols)
    df["has_dm_coverage"] = _has_any(dm_cols)
    df["has_aa_coverage"] = _has_any(aa_cols)
    df["has_rf_coverage"] = _has_any(rf_cols)
    df["has_tr_coverage"] = _has_any(tr_cols)
    df["has_oi_coverage"] = _has_any(oi_cols)
    df["has_ad_coverage"] = _has_any(ad_cols)
    df["has_sh_coverage"] = _has_any(sh_cols)
    df["has_wp_coverage"] = _has_any(wp_cols)
    df["has_ny_coverage"] = _has_any(ny_cols)
    df["has_gd_coverage"] = _has_any(gd_cols)
    df["has_avg_coverage"] = _has_any(avg_cols)
    df["has_pk_coverage"] = _has_any(pk_cols)
    df["has_xf_coverage"] = _has_any(xf_cols)
    df["has_inst_coverage"] = _has_any(inst_cols)
    df["has_opt_coverage"] = _has_any(opt_cols)
    df["has_gl_coverage"] = _has_any(gl_cols)
    cov_cols = [c for c in df.columns if c.startswith("has_")]

    feature_cols = (
        base_cols + fh_cols + emb_feature_cols + f4_cols + rd_cols
        + av_cols + yh_cols + e8k_cols + hn_cols + st_cols + cg_cols
        + dr_cols + sa_cols + sb_cols + rh_cols + fv_cols
        + nw_cols + dm_cols + aa_cols + rf_cols + tr_cols
        + oi_cols + ad_cols + sh_cols + wp_cols
        + ny_cols + gd_cols + avg_cols + pk_cols + xf_cols
        + inst_cols + opt_cols + gl_cols + cov_cols
    )
    categorical_cols = ["cusip_id", "year", "q_index"] + cov_cols

    # De-dup, order-preserving. `avg_hourly_earnings` is a FRED macro column (so it
    # is in base_cols via macro_cols) AND matches the `avg_` prefix group above, so
    # it lands in feature_cols twice. A repeated column carries no extra signal and
    # LightGBM rejects duplicate feature names ("appears more than one time"). This
    # also guards against any future prefix-group collision, not just this one.
    feature_cols = list(dict.fromkeys(feature_cols))
    categorical_cols = [c for c in dict.fromkeys(categorical_cols) if c in feature_cols]
    assert len(feature_cols) == len(set(feature_cols)), "feature_cols has duplicates after de-dup"
    return df, feature_cols, categorical_cols


def train_one(slug: str, df_inv: pd.DataFrame, feature_cols: list[str], cat_cols: list[str],
              holdout_start: pd.Timestamp = HOLDOUT_START,
              val_quarters: int = VAL_QUARTERS) -> dict:
    val_start = holdout_start - pd.DateOffset(months=3 * val_quarters)
    train = df_inv[df_inv["period_of_report"] < val_start]
    val = df_inv[(df_inv["period_of_report"] >= val_start) & (df_inv["period_of_report"] < holdout_start)]
    holdout = df_inv[df_inv["period_of_report"] >= holdout_start]

    if len(train) < 50 or train["y"].sum() < 5 or val["y"].sum() < 1:
        return {"slug": slug, "skip_reason": f"insufficient data n_train={len(train)} pos_train={int(train['y'].sum())}"}

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 32, "min_data_in_leaf": 25,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "verbose": -1,
    }
    if len(feature_cols) != len(set(feature_cols)):
        dups = sorted({c for c in feature_cols if feature_cols.count(c) > 1})
        raise ValueError(f"duplicate feature names reached train_one: {dups}")
    dtrain = lgb.Dataset(train[feature_cols], label=train["y"], categorical_feature=cat_cols)
    dval = lgb.Dataset(val[feature_cols], label=val["y"], categorical_feature=cat_cols, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=400,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    def _metrics(sub: pd.DataFrame) -> dict:
        if sub["y"].sum() < 1 or sub["y"].nunique() != 2:
            return {"n": int(len(sub)), "n_pos": int(sub["y"].sum()),
                    "auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
        p = booster.predict(sub[feature_cols])
        return {
            "n": int(len(sub)), "n_pos": int(sub["y"].sum()),
            "auc": float(roc_auc_score(sub["y"], p)),
            "pr_auc": float(average_precision_score(sub["y"], p)),
            "brier": float(brier_score_loss(sub["y"], p)),
        }

    val_m = _metrics(val)
    hol_m = _metrics(holdout)

    importance = booster.feature_importance(importance_type="gain")
    fi = dict(sorted(((f, float(v)) for f, v in zip(feature_cols, importance)), key=lambda x: -x[1]))

    return {
        "slug": slug,
        "n_train": int(len(train)),
        "n_train_positives": int(train["y"].sum()),
        "val": val_m, "holdout": hol_m,
        "top_features": dict(list(fi.items())[:20]),
        "_booster": booster,
        "_holdout_df": holdout.assign(p=booster.predict(holdout[feature_cols])) if len(holdout) else None,
    }


def train_all(repo: Path = REPO) -> dict:
    inputs = _load_inputs(repo)
    df_full, feature_cols, cat_cols = _attach_features(inputs["labels"], inputs)
    results: dict[str, dict] = {}
    out_root = repo / "runs" / "per_investor"
    out_root.mkdir(parents=True, exist_ok=True)
    for slug in sorted(df_full["investor_slug"].unique()):
        sub = df_full[df_full["investor_slug"] == slug]
        f_cols, c_cols = filter_feature_cols(feature_cols, cat_cols, slug)
        r = train_one(slug, sub, f_cols, c_cols)
        r["channels_used"] = channels_for_investor(slug)
        r["n_features_used"] = len(f_cols)
        results[slug] = r
        slug_dir = out_root / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        # Slim payload for metrics.json (no booster object)
        slim = {k: v for k, v in r.items() if not k.startswith("_")}
        (slug_dir / "metrics.json").write_text(json.dumps(slim, indent=2, default=str))
        # Holdout predictions for diff loop
        if r.get("_holdout_df") is not None:
            cols = ["period_of_report", "cusip", "ticker", "label", "y", "p"]
            hd = r["_holdout_df"]
            keep = [c for c in cols if c in hd.columns]
            hd[keep].to_parquet(slug_dir / "holdout_predictions.parquet", index=False)
            # Per-quarter picks parquet (the iterative-compare loop hook) — one
            # file per holdout quarter, sorted by p_new_entry descending.
            for quarter, qdf in hd.groupby("period_of_report"):
                qtag = pd.Timestamp(quarter).strftime("%Y-%m-%d")
                picks = qdf.sort_values("p", ascending=False)[
                    [c for c in cols if c in qdf.columns]
                ]
                picks.to_parquet(slug_dir / f"picks_{qtag}.parquet", index=False)
    return results


# ---------------------------------------------------------------------------
# Walk-forward retraining
# ---------------------------------------------------------------------------
#
# At each holdout quarter we retrain on STRICTLY-PRIOR data — train ∪ val ∪
# every prior holdout quarter — then predict only that quarter. This eliminates
# the look-ahead leakage of a single end-to-end fit on the whole training
# window. Picks come from the per-quarter model, which has seen every event up
# to (but not including) the predicted quarter.
#
# Port of quant/pipeline/backtest.py::_predict_walk_forward, simplified for
# LightGBM: no checkpoint warm-start required — every window is a fresh fit
# with a slightly larger training set.


def train_walk_forward(
    slug: str,
    df_inv: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    holdout_start: pd.Timestamp = HOLDOUT_START,
    val_quarters: int = VAL_QUARTERS,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 20,
    candidate_fn=None,
) -> dict:
    """Walk-forward train + predict for one investor.

    Returns the same shape as ``train_one`` plus ``window_metrics`` per quarter.
    The ``_holdout_df`` is the union of all per-quarter predictions, each one
    coming from a model trained on strictly-prior data.

    ``candidate_fn`` (optional) — ``fn(q_ts, train_window_df) -> DataFrame|None``.
    When provided, each quarter's freshly-fit booster ALSO scores this
    candidate frame (a broad, leak-free universe known at the rebalance date,
    not the filing's own constituents). Those predictions are returned in
    ``_broad_df``. The canonical path (``candidate_fn=None``) is byte-for-byte
    unchanged — the broad scoring is purely additive.
    """
    val_start = holdout_start - pd.DateOffset(months=3 * val_quarters)
    holdout_quarters = sorted(
        q for q in df_inv["period_of_report"].unique() if pd.Timestamp(q) >= holdout_start
    )
    if not holdout_quarters:
        return {"slug": slug, "skip_reason": "no holdout quarters"}

    train_initial = df_inv[df_inv["period_of_report"] < val_start]
    val = df_inv[
        (df_inv["period_of_report"] >= val_start)
        & (df_inv["period_of_report"] < holdout_start)
    ]
    if len(train_initial) < 50 or train_initial["y"].sum() < 5 or val["y"].sum() < 1:
        return {
            "slug": slug,
            "skip_reason": (
                f"insufficient data n_train={len(train_initial)} "
                f"pos_train={int(train_initial['y'].sum())}"
            ),
        }

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 32,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
    }

    window_metrics: list[dict] = []
    holdout_preds_frames: list[pd.DataFrame] = []
    broad_frames: list[pd.DataFrame] = []
    # Each window expands the training set to include every prior holdout quarter
    extra_holdout_history = df_inv.iloc[0:0]
    for q in holdout_quarters:
        q_ts = pd.Timestamp(q)
        train_window_df = pd.concat([train_initial, extra_holdout_history], ignore_index=True)
        if len(train_window_df) < 50 or train_window_df["y"].sum() < 5:
            window_metrics.append({
                "quarter": str(q_ts.date()),
                "skip_reason": f"train too small n={len(train_window_df)}",
            })
            continue

        dtrain = lgb.Dataset(
            train_window_df[feature_cols],
            label=train_window_df["y"],
            categorical_feature=cat_cols,
            weight=recency_weights(train_window_df["period_of_report"], q_ts),
        )
        dval = lgb.Dataset(
            val[feature_cols],
            label=val["y"],
            categorical_feature=cat_cols,
            reference=dtrain,
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(0),
            ],
        )

        # Leak-free broad-universe scoring (additive; canonical path unaffected).
        if candidate_fn is not None:
            cand = candidate_fn(q_ts, train_window_df)
            if cand is not None and len(cand):
                cand = cand.copy()
                cand["p"] = booster.predict(cand[feature_cols])
                broad_frames.append(cand)

        window_df = df_inv[df_inv["period_of_report"] == q_ts]
        if len(window_df) == 0:
            continue
        p = booster.predict(window_df[feature_cols])
        window_df = window_df.assign(p=p)
        holdout_preds_frames.append(window_df)

        # Window metrics — guard for single-class quarters
        if window_df["y"].sum() >= 1 and window_df["y"].nunique() == 2:
            window_metrics.append({
                "quarter": str(q_ts.date()),
                "n": int(len(window_df)),
                "n_pos": int(window_df["y"].sum()),
                "auc": float(roc_auc_score(window_df["y"], p)),
                "pr_auc": float(average_precision_score(window_df["y"], p)),
                "brier": float(brier_score_loss(window_df["y"], p)),
                "n_train": int(len(train_window_df)),
            })
        else:
            window_metrics.append({
                "quarter": str(q_ts.date()),
                "n": int(len(window_df)),
                "n_pos": int(window_df["y"].sum()),
                "auc": float("nan"),
                "pr_auc": float("nan"),
                "brier": float("nan"),
                "n_train": int(len(train_window_df)),
            })

        # Roll this quarter into the training history for the next window
        extra_holdout_history = pd.concat(
            [extra_holdout_history, window_df.drop(columns=["p"], errors="ignore")],
            ignore_index=True,
        )

    holdout_df = (
        pd.concat(holdout_preds_frames, ignore_index=True)
        if holdout_preds_frames
        else None
    )
    broad_df = (
        pd.concat(broad_frames, ignore_index=True) if broad_frames else None
    )

    # Aggregate metrics across windows (mean-weighted by n where applicable)
    valid_w = [w for w in window_metrics if not np.isnan(w.get("auc", float("nan")))]
    if valid_w:
        agg = {
            "n": int(sum(w["n"] for w in valid_w)),
            "auc": float(np.mean([w["auc"] for w in valid_w])),
            "pr_auc": float(np.mean([w["pr_auc"] for w in valid_w])),
            "brier": float(np.mean([w["brier"] for w in valid_w])),
        }
    else:
        agg = {"n": 0, "auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}

    return {
        "slug": slug,
        "mode": "walk_forward",
        "n_train_initial": int(len(train_initial)),
        "n_holdout_quarters": len(holdout_quarters),
        "holdout_agg": agg,
        "window_metrics": window_metrics,
        "_holdout_df": holdout_df,
        "_broad_df": broad_df,
    }


def train_all_walk_forward(repo: Path = REPO) -> dict:
    """Walk-forward variant of ``train_all`` — one fit per holdout quarter."""
    inputs = _load_inputs(repo)
    df_full, feature_cols, cat_cols = _attach_features(inputs["labels"], inputs)
    results: dict[str, dict] = {}
    out_root = repo / "runs" / "per_investor_wf"
    out_root.mkdir(parents=True, exist_ok=True)
    for slug in sorted(df_full["investor_slug"].unique()):
        sub = df_full[df_full["investor_slug"] == slug]
        f_cols, c_cols = filter_feature_cols(feature_cols, cat_cols, slug)
        r = train_walk_forward(slug, sub, f_cols, c_cols)
        r["channels_used"] = channels_for_investor(slug)
        r["n_features_used"] = len(f_cols)
        results[slug] = r
        slug_dir = out_root / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in r.items() if not k.startswith("_")}
        (slug_dir / "metrics.json").write_text(json.dumps(slim, indent=2, default=str))
        if r.get("_holdout_df") is not None:
            cols = ["period_of_report", "cusip", "ticker", "label", "y", "p"]
            hd = r["_holdout_df"]
            keep = [c for c in cols if c in hd.columns]
            hd[keep].to_parquet(slug_dir / "holdout_predictions.parquet", index=False)
            # Per-quarter picks
            for quarter, qdf in hd.groupby("period_of_report"):
                qtag = pd.Timestamp(quarter).strftime("%Y-%m-%d")
                picks = qdf.sort_values("p", ascending=False)[
                    [c for c in keep if c in qdf.columns]
                ]
                picks.to_parquet(slug_dir / f"picks_{qtag}.parquet", index=False)
    return results
