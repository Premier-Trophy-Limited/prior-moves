# Prior Moves

> Forward inference of what super-investors will buy next quarter — before
> the 45-day 13F filing window closes.

Per-investor walk-forward LightGBM models rank tickers by P(new_entry) for
each tracked super-investor. Cross-validated against congressional STOCK Act
disclosures (Senate + House PTRs) and historical mimic-backtest returns.

> Repo dir name is `super-investor-mirror`; the productised brand is
> `Prior Moves`. Package imports keep the legacy name so we don't break
> downstream tooling.

## Core mechanic

```
at quarter Q-end:
  features = fundamentals(Q) + price(Q) + sentiment(Q) for each S&P 500 ticker
  for each tracked investor:
    my_picks[investor] = top-K tickers by P(investor enters at Q | features)

at Q + 45 days:
  13F filings published; parse to extract:
    their_picks[investor] = set(Q holdings) - set(Q-1 holdings)
  for each investor:
    precision = |my_picks ∩ their_picks| / K
    recall    = |my_picks ∩ their_picks| / |their_picks|
    disagreement_report = my_picks - their_picks, their_picks - my_picks
  iterate: retrain on (features_Q, their_picks_Q) appended to training set
```

13F is what they reveal, not what they think. Treat it as a noisy proxy of
their investment thesis. Cash, private holdings, options, and shorts are
invisible.

## Investors tracked (13)

| Name | CIK | Style |
|---|---|---|
| Berkshire Hathaway (Buffett) | 0001067983 | Long-horizon value |
| Pershing Square (Ackman) | 0001336528 | Concentrated activist |
| Appaloosa (Tepper) | 0001656456 | Distressed / contrarian |
| Baupost (Klarman) | 0001061165 | Deep value |
| Greenlight (Einhorn) | 0001079114 | Long/short value |
| Scion (Burry) | 0001649339 | Contrarian, tail-risk |
| Third Point (Loeb) | 0001040273 | Event-driven activist |
| Pabrai Investment | 0001173334 | Concentrated value |
| Gotham (Greenblatt) | 0001403450 | Magic Formula systematic |
| Duquesne Family Office (Druckenmiller) | 0001536411 | Macro |
| Soros Fund Management | 0001029160 | Macro / family office |
| Renaissance Institutional | 0001037389 | Quant high-turnover |
| Two Sigma | 0001179392 | Quant high-turnover |

CIKs verified at first scrape; some entities have multiple filers (e.g. Berkshire
has BRK-A and BRK-B; Soros has multiple LP entities). The registry resolves to
the primary 13F filer per name.

## Universe

S&P 500 constituents at each quarter end. Historical constituent diffs scraped
from Wikipedia. Tickers that joined / left the index mid-quarter are handled by
the universe-as-of-Q-end convention.

## Data sources

- **SEC EDGAR**: 13F-HR XML, free, structured. Per filing: list of
  (ticker, CUSIP, shares, value, change vs prior).
- **yfinance**: quarterly price + basic fundamentals (PE, P/B, ROE, debt/equity,
  margins) per ticker. Free.
- **Wikipedia**: S&P 500 constituent history. Free.
- **Optional**: simfin / financialmodelingprep for cleaner fundamentals.

## Why a separate repo from AryaaSk/quant

Aryaa's quant predicts a binary outcome (earnings beat/miss) for one ticker at
one event. This project predicts a multi-investor multi-ticker portfolio at
quarter ends. Different problem shape; reusing Aryaa's pipeline would distort
both repos. Some primitives (numeric block builder, calibration, walk-forward
retraining) port over cleanly if either side wants to share later.

## Architecture as built

```
adapters/
  sec_13f.py             — EDGAR 13F-HR XML pull + CUSIP-level positions
  edgar_forms.py         — Form 4 (insider tx) + 13D/G (activist) metadata
  fred.py                — 12 FRED macro series + 2 derived YoY
  yfinance_news.py       — yfinance news + recommendations
  finnhub.py             — Finnhub Starter $50/mo: insider, news, recs
  ft_alphaville.py       — Playwright + stealth FT scraper (persistent profile)
  openfigi.py            — bulk CUSIP→ticker resolution with rate limiter
  gemma_embedder.py      — local embeddinggemma:300m via Ollama (768-dim)
  text_features.py       — embed Finnhub headlines, optional PCA reduce
  prices.py              — yfinance close history + quarterly_return()

models/
  baseline.py            — single LightGBM classifier, macro + cusip + investor
  per_investor.py        — separate LightGBM per investor (12 models)

compare.py               — top-K picks vs actual new_entries per (investor, Q)
backtest.py              — mimic-strategy: buy top-K at Q close, sell at Q+1
```

## Iterative compare result (current state, no Finnhub yet)

Per-investor models lift Buffett from AUC 0.52 (one-fit-all) to **0.73**.
Top-10 picks per held-out quarter:

| Investor | Quarters | Avg actual new | Avg overlap@10 | Avg recall@10 |
|---|---|---|---|---|
| ackman | 7 | 1.0 | 0.71 | 0.57 |
| burry | 5 | 3.2 | 1.6 | 0.48 |
| klarman | 7 | 4.7 | 1.9 | 0.46 |
| **buffett** | 7 | 2.3 | 1.3 | **0.40** |
| druckenmiller | 7 | 14.4 | 2.3 | 0.15 |
| greenblatt | 7 | 75.7 | 0.0 | 0.0 (K too small) |
| renaissance | 7 | 125.9 | 0.0 | 0.0 (K too small) |

**Buffett 2026-Q1**: model captured BOTH actual new entries
(Alphabet, Macy's) in top-10. 100% recall, 8 false positives (Ally,
Jefferies, Lennar, Liberty Media, ...).

Quant-tier (greenblatt / renaissance / two_sigma) needs K=50+ to be
evaluable; top-10 vs 100+ actual entries is structurally impossible.

## MVP — one-shot end-to-end (after data ingest)

```bash
# Train walk-forward + emit per-quarter picks
python scripts/train_per_investor_wf.py

# Backtest model top-10 vs SPY across 7 holdout quarters
python scripts/run_backtest.py --top-k 10 --hold-quarters 1 --source-subdir per_investor_wf

# Refresh REPORT_WF.md
python scripts/build_report.py --source-subdir per_investor_wf

# Export current-quarter picks (machine-readable JSON + markdown)
python scripts/live_picks.py --source-subdir per_investor_wf --top-k 15

# Launch interactive dashboard (Streamlit; http://localhost:8501)
streamlit run app.py
```

## Run order — data ingest (one-time)

```bash
python scripts/fetch_13fs.py             # 13F historical pull (~30 min once)
python scripts/build_labels.py           # new_entry / add / trim / hold / exit
python scripts/fetch_macro.py            # FRED macro per quarter
python scripts/map_cusips.py             # CUSIP→ticker via OpenFIGI (~50 min anon)
python scripts/finnhub_backfill.py       # top-N tickers × N quarters insider+news
python scripts/fetch_form4.py --top-tickers 100 --since 2020-01-01 --merge
python scripts/fetch_yfinance.py --top-tickers 200 --merge   # multi-year fundamentals
python scripts/fetch_8k.py --top-tickers 300 --merge         # SEC 8-K item codes
python scripts/fetch_hackernews.py --top-tickers 200 --merge # HN cashtag mentions
python scripts/fetch_stocktwits.py --top-tickers 100 --max-pages 10 --merge
python scripts/fetch_reddit.py --since 2015-01-01 --limit 200
python scripts/ft_login.py                                   # one-time interactive FT cookies
python scripts/ft_alphaville_pull.py --pages 1500 --max-posts 30000 --merge --pages-per-chunk 20
python scripts/build_text_features.py --source finnhub       # Gemma embed news
python scripts/build_text_features.py --source alphaville    # Gemma embed FT
```

For FT Alphaville scraping, see `docs/SETUP.md`:

```bash
python scripts/ft_login.py               # one-time interactive
python scripts/ft_alphaville_pull.py     # headless via persistent profile
```

## Status

- Phase A (free signals): SEC EDGAR + FRED macro + yfinance news + OpenFIGI CUSIP map ✓
- Phase B (paid + scraped):
  - Finnhub Premium $50/mo: live insider + news + recs ✓
  - FT Alphaville: Playwright code written; user runs `scripts/ft_login.py` once
  - Reddit, Matt Levine, GDELT: pending
- Phase C (model):
  - Baseline + per-investor LightGBM ✓
  - Local Gemma text-feature embedding (768-dim, optional PCA) ✓
  - Iterative compare loop (top-K precision / recall / hit-miss-fp) ✓
  - Mimic backtest (model picks vs actual vs SPY) ✓

See `docs/PLAN.md` for staged build-out and `docs/SETUP.md` for credentials.
