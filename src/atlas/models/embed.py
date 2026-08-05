from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from atlas.config import EmbeddingSettings
from atlas.contracts import Embedder

# Bounds the loss from a transient failure (rate limit, network blip) to one batch's
# worth of vectors rather than the whole miss list, since nothing is cached until its
# batch returns.
_CACHE_BATCH_SIZE = 96


def cache_key(
    *, model_name: str, model_version: str, size: int, normalised: bool, text: str
) -> str:
    """Hashes the components in a fixed order, each prefixed by its byte length so one
    value cannot run into the next. Includes model identity so a cache never hands back
    another model's vectors after a swap."""
    parts = [model_name, model_version, str(size), str(normalised), text]
    payload = "".join(f"{len(part.encode('utf-8'))}:{part}" for part in parts)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()


class CachedEmbedder:
    """Wraps any Embedder and never buys the same vector twice. Two layers: an in-process
    dict, then an optional on-disk cache under `directory` that survives across runs.
    Both keyed by full model identity, so a hit can only come from the model that
    produced it."""

    def __init__(self, inner: Embedder, directory: str | Path | None = None) -> None:
        self._inner = inner
        self._seen: dict[str, np.ndarray] = {}
        self._directory = Path(directory) if directory is not None else None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def normalised(self) -> bool:
        return self._inner.normalised

    @property
    def input_tokens(self) -> int:
        return self._inner.input_tokens

    def _key(self, text: str) -> str:
        return cache_key(
            model_name=self.model_name, model_version=self.model_version,
            size=self.size, normalised=self.normalised, text=text,
        )

    def _path(self, text: str) -> Path | None:
        if self._directory is None:
            return None
        key = cache_key(
            model_name=self.model_name, model_version=self.model_version,
            size=self.size, normalised=self.normalised, text=text,
        )
        return self._directory / f"{key}.npy"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            # Shaped, not merely empty: np.array([]) is one dimensional and would give a
            # caller indexing a column an axis error instead of zero rows.
            return np.zeros((0, self.size), dtype=np.float32)

        for text in dict.fromkeys(texts):
            if text in self._seen:
                continue
            path = self._path(text)
            if path is not None and path.exists():
                self._seen[text] = np.load(path)

        missing = [t for t in dict.fromkeys(texts) if t not in self._seen]
        for start in range(0, len(missing), _CACHE_BATCH_SIZE):
            batch = missing[start:start + _CACHE_BATCH_SIZE]
            for text, vector in zip(batch, self._inner.encode(batch), strict=True):
                self._seen[text] = vector
                self._write(text, vector)
        return np.array([self._seen[t] for t in texts], dtype=np.float32)

    def _write(self, text: str, vector: np.ndarray) -> None:
        """Writes via a temp file plus `os.replace` so a process kill mid-write never
        leaves a truncated file for `np.load` to find. Temp name must end in `.npy`:
        `np.save` appends that suffix itself when it's missing, which would otherwise
        break the matching `os.replace` target."""
        path = self._path(text)
        if path is None:
            return
        temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(temp, vector)
        os.replace(temp, path)


def get_embedder(settings: EmbeddingSettings) -> Embedder:
    """The one embedder there is. Imported inside the function so this module, and
    `cache_key`, stay importable without the openai package."""
    from atlas.models.providers import OpenAIEmbedder

    return OpenAIEmbedder(settings)
