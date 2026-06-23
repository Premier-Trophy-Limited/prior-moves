"""Local Ollama embedder, drop-in alternative to Voyage.

Targets the `embeddinggemma:300m` model (Apple-Silicon-friendly Gemma 3 embedder) running
on a local Ollama server at OLLAMA_HOST (default `http://127.0.0.1:11434`). Same disk-cache
contract as `VoyageEmbedder`: every (model, text) is hashed to sha256 and persisted under
the supplied cache_dir, so re-runs read from disk and only new strings hit the model.

Why this exists:
- Voyage adds external cost and per-call latency; local embedder permits dense per-event
  scraping (every news article, every 8-K section) without quota anxiety.
- Lets the methodology in LIMITATIONS.md section 6 be retested with a 1024+ slot text block
  whose coverage is no longer rate-limited by an API budget.

Output dimension is queried at embed() time by inspecting the first response, so swapping
the underlying model (e.g. to `gemma4:e4b` if it exposes embeddings, or a future
`gemma-embed-1024`) does not require code changes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


class OllamaEmbedder:
    """Caching client around the Ollama embeddings endpoint."""

    def __init__(
        self,
        model: str = "embeddinggemma:300m",
        cache_dir: Path | None = None,
        host: str | None = None,
        request_timeout: float = 120.0,
    ):
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.request_timeout = request_timeout
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embed_dim()), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            fetched = self._embed_uncached(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                results[idx] = fetched[j]
                self._cache_put(uncached_texts[j], fetched[j])

        return np.stack([r for r in results if r is not None], axis=0)

    def _embed_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        probe = self._embed_uncached([" "])
        self._dim = int(probe.shape[1])
        return self._dim

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self._cache_key(text)}.npy"

    def _cache_get(self, text: str) -> np.ndarray | None:
        p = self._cache_path(text)
        if p and p.exists():
            try:
                return np.load(p)
            except (OSError, ValueError):
                return None
        return None

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        p = self._cache_path(text)
        if p:
            np.save(p, vec.astype(np.float32))

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        # Ollama's /api/embed accepts a list of inputs in one call. Chunk modestly so a
        # single bad payload does not cost a whole batch.
        out: list[np.ndarray] = []
        for chunk_start in range(0, len(texts), 32):
            chunk = texts[chunk_start : chunk_start + 32]
            chunk = [c if c.strip() else " " for c in chunk]
            payload = {"model": self.model, "input": chunk}
            with httpx.Client(timeout=self.request_timeout) as client:
                resp = client.post(f"{self.host}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
            embeddings_field = data.get("embeddings") or data.get("data") or []
            for item in embeddings_field:
                vec = item if isinstance(item, list) else item.get("embedding", [])
                out.append(np.asarray(vec, dtype=np.float32))
        return np.stack(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)


def embed_with_ollama(
    texts: list[str], *, model: str = "embeddinggemma:300m", cache_dir: Path | None = None
) -> np.ndarray:
    """Convenience helper; same shape contract as `embed_with_voyage`."""
    return OllamaEmbedder(model=model, cache_dir=cache_dir).embed(texts)
