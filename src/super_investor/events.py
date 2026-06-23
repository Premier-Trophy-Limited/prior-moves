"""Event->Impact engine — shared library (Workstreams C2/C3).

Two jobs:
  * map_impact_to_holdings()  (C3) — ground an event's deterministic sector
    impact map to the greats' REAL current positioning (latest aggregated
    forward), so every claim is anchored to who actually holds the name.
  * causal-chain plumbing (C2) — build the Codex packet/prompt for an event and
    parse the structured 1st/2nd/3rd-order chain back. The Codex call itself is
    driven by scripts/build_event_impact.py (one `ai-do --lane=codex` per event,
    zero Claude usage).

Honesty: the deterministic map (impact_sectors -> held names) is what the
leak-free backtest trades on. The Codex chain is a *qualitative explanation*
layer — labeled as analyst synthesis, no price targets, no buy calls.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEATURES = REPO / "data" / "features"
RUNS = REPO / "runs" / "per_investor_wf"
EVENT_RUNS = REPO / "runs" / "event_study"

TIMELINE = FEATURES / "event_timeline.parquet"


# --- loaders ----------------------------------------------------------------
def load_timeline() -> pd.DataFrame:
    """Event timeline with list columns decoded back from JSON."""
    df = pd.read_parquet(TIMELINE).copy()
    for c in ("entities", "impact_sectors", "impact_themes"):
        df[c] = df[c].apply(lambda v: json.loads(v) if isinstance(v, str) else list(v))
    df["observed_date"] = pd.to_datetime(df["observed_date"])
    return df.sort_values("observed_date").reset_index(drop=True)


def load_universe() -> pd.DataFrame:
    """Latest aggregated forward picks (the scored, mirrored universe).

    NOTE: the `all_investors` column lists the investor *models* that scored a
    name (all of them score every candidate) — it is NOT who holds it. For real
    backing investors use load_holders()."""
    files = sorted(glob.glob(str(RUNS / "aggregated_forward_*.parquet")))
    if not files:
        raise FileNotFoundError("no aggregated_forward_*.parquet — run build_aggregated_forward.py")
    return pd.read_parquet(files[-1]).copy()


def load_holders() -> dict[str, list[str]]:
    """ticker -> sorted list of investor slugs who ACTUALLY hold it in their most
    recent 13F (any non-exit position). Multi-strats in MIRROR_EXCLUDE are
    dropped so this matches the mirrored consensus."""
    from super_investor.scoring import MIRROR_EXCLUDE

    lab = pd.read_parquet(REPO / "data" / "13f" / "labels.parquet").reset_index(drop=True)
    # latest filing per investor only
    latest = lab.groupby("investor_slug")["period_of_report"].transform("max")
    cur = lab[(lab["period_of_report"] == latest) & (lab["label"] != "exit") & (lab["shares"] > 0)]
    cur = cur[~cur["investor_slug"].isin(MIRROR_EXCLUDE)]
    cmap = pd.read_parquet(REPO / "data" / "tickers" / "cusip_to_ticker.parquet")[["cusip", "ticker"]]
    cur = cur.merge(cmap, on="cusip", how="left").dropna(subset=["ticker"])
    out: dict[str, set[str]] = {}
    for tk, slug in zip(cur["ticker"], cur["investor_slug"]):
        out.setdefault(str(tk).upper(), set()).add(str(slug))
    return {k: sorted(v) for k, v in out.items()}


def _investors(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    try:  # numpy array
        return [str(x) for x in list(v)]
    except TypeError:
        return [str(v)]


def event_by_id(event_id: str) -> pd.Series | None:
    tl = load_timeline()
    hit = tl[tl["event_id"] == event_id]
    return None if hit.empty else hit.iloc[0]


# --- C3: ground impact to real holdings -------------------------------------
def map_impact_to_holdings(
    event: pd.Series, universe: pd.DataFrame | None = None,
    holders: dict[str, list[str]] | None = None, top: int = 12
) -> dict:
    """Names the greats ALREADY hold inside the event's impact sectors, ranked by
    PriorScore. This is the grounded, deterministic basket — every row anchored
    to REAL backing investors (latest 13F). No LLM, no look-ahead beyond the
    latest 13F."""
    if universe is None:
        universe = load_universe()
    if holders is None:
        holders = load_holders()
    sectors = event["impact_sectors"] if isinstance(event["impact_sectors"], list) \
        else json.loads(event["impact_sectors"])
    sub = universe[universe["sector"].isin(sectors)].copy()
    sub = sub.sort_values("prior_score", ascending=False)
    holdings = []
    for _, r in sub.iterrows():
        tk = str(r["ticker"]).upper()
        backers = holders.get(tk, [])
        if not backers:  # only names a great actually holds
            continue
        holdings.append({
            "ticker": tk,
            "name": r.get("name", ""),
            "sector": r["sector"],
            "prior_score": round(float(r["prior_score"]), 1),
            "n_investors": len(backers),
            "backing_investors": backers,
            "status": "already-held",
        })
        if len(holdings) >= top:
            break
    return {
        "event_id": event["event_id"],
        "title": event["title"],
        "observed_date": pd.Timestamp(event["observed_date"]).strftime("%Y-%m-%d"),
        "impact_sectors": sectors,
        "impact_themes": event["impact_themes"] if isinstance(event["impact_themes"], list)
        else json.loads(event["impact_themes"]),
        "holdings": holdings,
    }


# --- C2: Codex causal-chain packet + prompt + parse -------------------------
def build_packet(event: pd.Series, grounded: dict) -> dict:
    """Compact, factual packet handed to Codex. Only as-of facts + the grounded
    holdings — Codex reasons the causal chain, it does not invent positioning."""
    macro = _macro_snapshot(pd.Timestamp(event["observed_date"]))
    return {
        "event": {
            "title": event["title"],
            "observed_date": grounded["observed_date"],
            "type": event["type"],
            "entities": event["entities"] if isinstance(event["entities"], list)
            else json.loads(event["entities"]),
            "description": event["description"],
        },
        "as_of_macro": macro,
        "impact_sectors": grounded["impact_sectors"],
        "impact_themes": grounded["impact_themes"],
        "greats_hold_in_these_sectors": [
            {"ticker": h["ticker"], "name": h["name"], "sector": h["sector"],
             "backing_investors": h["backing_investors"]}
            for h in grounded["holdings"]
        ],
    }


def _macro_snapshot(as_of: pd.Timestamp) -> dict:
    p = FEATURES / "macro_quarterly.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df = df[pd.to_datetime(df["quarter_end"]) <= as_of]
    if df.empty:
        return {}
    r = df.sort_values("quarter_end").iloc[-1]
    return {k: round(float(r[k]), 2) for k in _MACRO_KEEP if k in r and pd.notna(r[k])}


# Fields surfaced to the Codex causal-chain packet. Guarded by membership so an
# absent column (flaky-CDN miss, ^MOVE drop) is skipped, never errors.
_MACRO_KEEP = [
    # regime
    "vix", "hy_oas", "real_10y", "term_10y_2y", "breakeven_10y",
    "fed_funds_upper", "usd_broad", "nfci",
    # inflation / labor / growth (D2 — actuals, not just expectations)
    "core_cpi_yoy_pct", "core_pce_yoy_pct", "unemployment", "claims",
    "payrolls_yoy_pct", "gdp_yoy_pct",
    # commodity prices + key FX (D4 — gold/copper/btc real prices)
    "wti", "gold", "copper", "btc", "jpy_usd", "cny_usd",
]


def _macro_snapshot_live() -> dict:
    """Latest-available WEEKLY regime snapshot for the live Event Lens. Reads
    macro_weekly.parquet (the D3 live frame) — NEVER used in any backtested
    number. Returns the snapshot plus the as-of week date."""
    p = FEATURES / "macro_weekly.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if df.empty:
        return {}
    r = df.sort_values("week_end").iloc[-1]
    snap = {k: round(float(r[k]), 2) for k in _MACRO_KEEP if k in r and pd.notna(r[k])}
    snap["as_of"] = pd.Timestamp(r["week_end"]).strftime("%Y-%m-%d")
    return snap


CHAIN_SCHEMA_HINT = (
    '{"order1":[{"node":str,"theme":str,"tickers":[str],"rationale":str,'
    '"confidence":"high|medium|low","falsifier":str}],'
    '"order2":[...same shape...],"order3":[...same shape...],'
    '"summary":str,"key_uncertainty":str}'
)


def build_prompt(packet: dict) -> str:
    return (
        "You are a buy-side macro analyst writing QUALITATIVE research, not "
        "advice. Given the EVENT and the as-of macro snapshot below, trace the "
        "causal chain of consequences in THREE orders:\n"
        "  order1 = direct, first-order effects (sector / commodity / region).\n"
        "  order2 = second-order (supply chain, input costs, substitution, "
        "capex, financing).\n"
        "  order3 = third-order (capital flows, policy response, knock-on "
        "beneficiaries the market under-prices).\n\n"
        "For each node give 1-5 US-listed tickers that benefit, a one-line "
        "rationale, a confidence (high/medium/low), and a one-line FALSIFIER "
        "(what observation would prove the node wrong). Anchor on names the "
        "greats already hold (listed below), AND in every order include at least "
        "one or two beneficiaries they do NOT yet hold — genuine forward "
        "'predicted-next' ideas the market under-prices.\n\n"
        "HARD RULES: qualitative only; NO price targets; NO 'buy'/'sell'; trace "
        "causes-of-causes explicitly; surface uncertainty; this is analyst "
        "synthesis for a general audience, not individualized advice.\n\n"
        "Return ONLY valid minified JSON, no prose, matching exactly this shape:\n"
        + CHAIN_SCHEMA_HINT + "\n\nPACKET:\n" + json.dumps(packet, default=str)
    )


def parse_chain(raw: str) -> dict | None:
    """Extract the JSON object from a Codex response (tolerates code fences /
    leading prose)."""
    if not raw:
        return None
    s = raw.strip()
    if "```" in s:
        # take the largest fenced block
        parts = [p for p in s.split("```") if "{" in p and "}" in p]
        if parts:
            s = max(parts, key=len)
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None


def merge_chain_with_universe(
    chain: dict, universe: pd.DataFrame | None = None,
    holders: dict[str, list[str]] | None = None,
    event: pd.Series | None = None,
) -> dict:
    """Tag every ticker in the chain as already-held (with REAL backing
    investors) or predicted-next (not held by the greats yet) — the grounded
    watchlist. 'predicted-next' is the forward signal: an event-implied
    beneficiary the greats have not bought."""
    if universe is None:
        universe = load_universe()
    if holders is None:
        holders = load_holders()
    score = {str(t).upper(): float(s) for t, s in
             zip(universe["ticker"], universe["prior_score"])}
    sec_map = {str(t).upper(): s for t, s in
               zip(universe["ticker"], universe.get("sector", pd.Series(dtype=str)))}
    # D5: if a trained impact model + the event are available, predict a forward
    # impact weight per name to rank the watchlist; else fall back to PriorScore.
    impact = _predict_chain_impact(chain, event, score, sec_map) if event is not None else {}
    seen: dict[str, dict] = {}
    for order in ("order1", "order2", "order3"):
        for node in chain.get(order, []) or []:
            for tk in node.get("tickers", []) or []:
                u = str(tk).upper()
                if u in seen:
                    continue
                backers = holders.get(u, [])
                seen[u] = {
                    "ticker": u,
                    "order": order,
                    "theme": node.get("theme", ""),
                    "confidence": node.get("confidence", ""),
                    "status": "already-held" if backers else "predicted-next",
                    "prior_score": round(score[u], 1) if u in score else None,
                    "impact_weight": round(impact[u], 4) if u in impact else None,
                    "n_investors": len(backers),
                    "backing_investors": backers,
                }
    return {
        "summary": chain.get("summary", ""),
        "key_uncertainty": chain.get("key_uncertainty", ""),
        "watchlist": sorted(seen.values(),
                            key=lambda x: (x["status"] != "already-held",
                                           -(x["impact_weight"] if x["impact_weight"] is not None
                                             else (x["prior_score"] or 0) / 1e6))),
    }


def _predict_chain_impact(chain: dict, event: pd.Series, score: dict,
                          sec_map: dict) -> dict:
    """Per-ticker learned impact weight for the live watchlist (D5). Uses the
    saved model + the event's as-of-live macro. Returns {} on any miss (no
    model, import error) so the caller cleanly falls back to PriorScore."""
    try:
        from super_investor import impact_model as IM
    except Exception:
        return {}
    model = IM.ImpactModel.load(REPO / "runs" / "impact_model" / "model.json")
    if model is None:
        return {}
    sectors = event["impact_sectors"] if isinstance(event["impact_sectors"], list) \
        else json.loads(event["impact_sectors"])
    macro = _macro_snapshot_live() or _macro_snapshot(pd.Timestamp(event["observed_date"]))
    tickers, rows = [], []
    for order in ("order1", "order2", "order3"):
        for node in chain.get(order, []) or []:
            for tk in node.get("tickers", []) or []:
                u = str(tk).upper()
                if u in tickers:
                    continue
                in_sec = sec_map.get(u) in sectors
                tickers.append(u)
                rows.append(IM.feature_row(event["type"], event.get("magnitude", 0),
                                           in_sec, score.get(u), macro))
    return model.weights(tickers, rows)
