"""Text feature extractor — embed per-(ticker, quarter) text via local Gemma.

For each Finnhub news headlines_text blob (one per ticker × quarter), produce
a 768-dim mean-pooled embedding via the local embeddinggemma:300m model.
Optionally PCA-reduce to a smaller dim before joining as LightGBM features
(dense 768-dim columns are sub-optimal for tree learners).

Pipeline:
  1. Load data/features/finnhub_news.parquet  (ticker, quarter_end, headlines_text)
  2. Split each headlines_text into individual lines
  3. Embed lines in batches via GemmaEmbedder
  4. Mean-pool per (ticker, quarter)
  5. Optionally PCA-reduce
  6. Save data/features/text_embeddings.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from super_investor.adapters.gemma_embedder import OllamaEmbedder


def split_headlines(text: str, min_len: int = 30) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if len(ln) >= min_len]


def embed_news_blobs(
    df: pd.DataFrame, *,
    embedder: OllamaEmbedder,
    text_col: str = "headlines_text",
    key_cols: tuple[str, ...] = ("ticker", "quarter_end"),
    pool: str = "mean",
) -> pd.DataFrame:
    """Return one embedding row per (ticker, quarter_end).

    Empty / NaN headlines map to a zero vector to keep alignment with the
    label rows that don't have Finnhub coverage.
    """
    rows: list[dict] = []
    dim: int | None = None
    for _, r in df.iterrows():
        lines = split_headlines(r.get(text_col, ""))
        if not lines:
            vec = None
        else:
            vecs = embedder.embed(lines)
            if vecs.shape[0] == 0:
                vec = None
            else:
                if pool == "mean":
                    vec = vecs.mean(axis=0)
                elif pool == "length_weighted":
                    weights = np.sqrt(np.array([max(len(ln), 1) for ln in lines], dtype=np.float64))
                    weights /= weights.sum()
                    vec = (vecs * weights[:, None]).sum(axis=0)
                else:
                    raise ValueError(f"unknown pool {pool}")
                dim = vec.shape[0]
        row = {k: r[k] for k in key_cols if k in r.index}
        if vec is None:
            row["embedding"] = None
        else:
            row["embedding"] = vec.astype(np.float32).tolist()
        rows.append(row)
    out = pd.DataFrame(rows)
    # Fill missing with zero vectors of correct dim
    if dim is not None:
        zero = [0.0] * dim
        out["embedding"] = out["embedding"].apply(lambda v: v if v is not None else zero)
    return out


def reduce_with_pca(df: pd.DataFrame, *,
                    embedding_col: str = "embedding",
                    n_components: int = 32,
                    random_state: int = 0) -> pd.DataFrame:
    """PCA-reduce embeddings to `n_components` dims and unpack to separate columns.

    Returns a copy of `df` with the `embedding` column expanded into
    `emb_0`, ..., `emb_{n-1}`.
    """
    from sklearn.decomposition import PCA
    mat = np.stack(df[embedding_col].to_list())
    n_components = min(n_components, mat.shape[1], mat.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(mat)
    out = df.drop(columns=[embedding_col]).copy()
    for i in range(n_components):
        out[f"emb_{i}"] = reduced[:, i].astype(np.float32)
    out.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return out
