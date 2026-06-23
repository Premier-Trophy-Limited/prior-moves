"""Leak + schema + determinism tests for the filings-embeddings channel (Track 1).

Network-free: no Ollama, no prices. The leak test constructs a tiny synthetic
events frame and a tiny synthetic channel parquet and asserts the as-of join with
the +1 day availability lag never lets an event see an embedding from a
same-day-or-later filing. The schema test checks the channel shape the build
script writes is discoverable and joinable. The determinism test checks the
seeded random projection reproduces exactly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from super_investor import channel_join as CJ  # noqa: E402

CHANNEL = "filings_embeddings"
EMB_COLS = [f"emb_{j}" for j in range(32)]


def _load_build_module():
    """Import scripts/build_filings_embeddings_channel.py by path (scripts/ is
    not a package)."""
    path = REPO / "scripts" / "build_filings_embeddings_channel.py"
    spec = importlib.util.spec_from_file_location("build_filings_embeddings_channel", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_channel(d: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(d / f"{CHANNEL}_quarterly.parquet", index=False)


def test_lag_registered() -> None:
    # the +1 day availability is the leak guard; it must be registered.
    assert CJ._LAG_DAYS.get(CHANNEL) == 1


def test_leak_safe_never_uses_same_day_or_later_filing(tmp_path: Path) -> None:
    # Channel has an embedding from a PRIOR filing (period_end 2020-03-01, avail
    # 2020-03-02) and one from a SAME-DAY filing (period_end 2020-06-01, avail
    # 2020-06-02). The event is filed 2020-06-01. With lag=1:
    #   prior  -> avail 2020-03-02 <= 2020-06-01  ELIGIBLE
    #   sameday-> avail 2020-06-02 >  2020-06-01  LEAK if used
    emb_prior = {f"emb_{j}": float(j) for j in range(32)}
    emb_same = {f"emb_{j}": 99.0 for j in range(32)}
    _write_channel(tmp_path, [
        {"ticker": "AAA", "period_end": pd.Timestamp("2020-03-01"), **emb_prior},
        {"ticker": "AAA", "period_end": pd.Timestamp("2020-06-01"), **emb_same},
    ])
    ev = pd.DataFrame({"ticker": ["AAA"], "filing_date": [pd.Timestamp("2020-06-01")]})

    out, _ = CJ.join_channels(ev, channels=[CHANNEL], features_dir=tmp_path,
                              keep_avail=True)
    # must use the prior filing's embedding, never the same-day one
    assert out.loc[0, f"ch_{CHANNEL}__emb_5"] == 5.0
    assert out.loc[0, f"ch_{CHANNEL}__emb_5"] != 99.0
    assert out.loc[0, f"has_{CHANNEL}"] == 1.0
    # the matched availability date is strictly before the event filing date
    assert out.loc[0, f"avail_{CHANNEL}"] < ev.loc[0, "filing_date"]


def test_leak_safe_no_past_filing_neutral_fills(tmp_path: Path) -> None:
    # only a same-day filing exists -> avail (filing+1) is after the event -> nothing
    # eligible -> neutral fill and coverage flag off.
    emb = {f"emb_{j}": 7.0 for j in range(32)}
    _write_channel(tmp_path, [
        {"ticker": "BBB", "period_end": pd.Timestamp("2021-01-15"), **emb},
    ])
    ev = pd.DataFrame({"ticker": ["BBB"], "filing_date": [pd.Timestamp("2021-01-15")]})
    out, _ = CJ.join_channels(ev, channels=[CHANNEL], features_dir=tmp_path)
    assert out.loc[0, f"ch_{CHANNEL}__emb_0"] == 0.0
    assert out.loc[0, f"has_{CHANNEL}"] == 0.0


def test_schema_discoverable_and_columns(tmp_path: Path) -> None:
    emb = {f"emb_{j}": float(j) for j in range(32)}
    _write_channel(tmp_path, [
        {"ticker": "CCC", "period_end": pd.Timestamp("2020-03-31"), **emb},
        {"ticker": "DDD", "period_end": pd.Timestamp("2020-06-30"), **emb},
    ])
    # discover_channels lists it
    assert CHANNEL in CJ.discover_channels(features_dir=tmp_path)
    # channel_columns reports the emb_* features plus the coverage flag
    cols = CJ.channel_columns(CHANNEL, features_dir=tmp_path)
    assert f"ch_{CHANNEL}__emb_0" in cols
    assert f"ch_{CHANNEL}__emb_31" in cols
    assert f"has_{CHANNEL}" in cols
    # the parquet carries ticker + period_end + numeric emb_*
    df = pd.read_parquet(tmp_path / f"{CHANNEL}_quarterly.parquet")
    assert "ticker" in df.columns
    assert "period_end" in df.columns
    for c in EMB_COLS:
        assert c in df.columns
        assert pd.api.types.is_numeric_dtype(df[c])


def test_random_projection_is_seeded_and_deterministic() -> None:
    mod = _load_build_module()
    a = mod.random_projection()
    b = mod.random_projection()
    assert a.shape == (mod.EMBED_DIM, mod.PROJ_DIM)
    np.testing.assert_array_equal(a, b)  # same seed -> identical
    # a different seed gives a different matrix
    c = mod.random_projection(seed=mod.PROJ_SEED + 1)
    assert not np.array_equal(a, c)
    # projecting a fixed vector is reproducible
    rng = np.random.default_rng(0)
    v = rng.standard_normal(mod.EMBED_DIM)
    np.testing.assert_array_equal(v @ a, v @ b)


def test_event_text_is_pit_clean_and_nonempty() -> None:
    mod = _load_build_module()
    row = pd.Series({
        "ticker": "AAPL", "filing_date": pd.Timestamp("2022-01-01"),
        "source": "8k", "sector": "Information Technology",
        "is_activist": 0, "magnitude": 4,
        "item_2_02": 1, "item_7_01": 1, "item_8_01": 0, "item_1_01": 0,
    })
    text = mod.event_text(row)
    assert "8-K" in text
    assert "results of operations" in text
    assert "Regulation FD" in text
    assert "Information Technology" in text
    # a 13D activist event renders the activist clause
    row13d = pd.Series({
        "ticker": "XYZ", "filing_date": pd.Timestamp("2022-01-01"),
        "source": "13d", "sector": "Unknown", "is_activist": 1, "magnitude": 4,
    })
    t13 = mod.event_text(row13d)
    assert "13D" in t13
    assert "activist" in t13
