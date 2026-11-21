"""Ranking metrics for top-N recommendation (binary relevance)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def precision_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for r in list(recommended)[:k] if r in relevant)
    return hits / k


def recall_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in list(recommended)[:k] if r in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    """AP@k normalised by min(|relevant|, k) (the usual Kaggle/H&M convention)."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 1) for rank, item in enumerate(list(recommended)[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommended: Sequence[Sequence], catalog_size: int) -> float:
    """Share of the candidate catalog that appears in at least one recommendation list."""
    if catalog_size <= 0:
        return 0.0
    uniq = set()
    for rec in all_recommended:
        uniq.update(rec)
    return len(uniq) / catalog_size


def intra_list_diversity(item_vectors, rec_indices: Sequence[int]) -> float:
    """1 - mean pairwise cosine similarity of recommended items.

    `item_vectors` must be L2-normalised rows (dense ndarray or scipy sparse).
    """
    idx = list(rec_indices)
    n = len(idx)
    if n < 2:
        return 0.0
    sub = item_vectors[idx]
    sim = sub @ sub.T
    sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
    mean_off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - mean_off_diag)
