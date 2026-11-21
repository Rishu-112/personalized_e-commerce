"""Common interface for all recommenders."""
from __future__ import annotations

import numpy as np


class Recommender:
    """All models score items for a batch of users on the shared item index.

    score_batch(user_idx) -> float32 array (len(user_idx), n_items); higher = better.
    Seen-item masking and top-N selection are done by the caller (evaluation / inference),
    so every model is evaluated under identical candidate rules.
    """
    name = "base"

    def fit(self, ctx) -> "Recommender":  # pragma: no cover - interface
        raise NotImplementedError

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def score_user(self, u: int) -> np.ndarray:
        return self.score_batch(np.array([u]))[0]


def minmax_rows(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-user min-max normalisation to [0, 1] over non-masked candidate items."""
    s = np.asarray(scores, dtype=np.float32).copy()
    if mask is None:
        mask = np.zeros_like(s, dtype=bool)
    big = np.float32(1e30)
    lo = np.where(mask, big, s).min(axis=1, keepdims=True)
    hi = np.where(mask, -big, s).max(axis=1, keepdims=True)
    rng = hi - lo
    out = np.divide(s - lo, rng, out=np.zeros_like(s), where=rng > 0)
    out[mask] = 0.0
    return out


def top_n(scores_row: np.ndarray, n: int, mask_row: np.ndarray | None = None) -> np.ndarray:
    """Indices of the n best items (descending), excluding masked ones. Deterministic tie-break by index."""
    s = scores_row.astype(np.float64).copy()
    if mask_row is not None:
        s[mask_row] = -np.inf
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-s, n - 1)[:n]
    return part[np.lexsort((part, -s[part]))]
