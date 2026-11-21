"""Model E: hybrid ranking = weighted sum of per-user min-max-normalised component scores.

Components: popularity, content, collaborative, matrix_factorization,
            category_affinity (customer-category affinity) and recency (trending items).
Weights are tuned on VALIDATION data only (random search over the simplex).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from src.models.base import Recommender, minmax_rows

log = logging.getLogger("hybrid")

ALL_COMPONENTS = ["popularity", "content", "collaborative", "matrix_factorization", "category_affinity", "recency"]
COLD_OK = {"popularity", "recency"}   # components that work for customers without history


class HybridModel(Recommender):
    name = "hybrid"

    def __init__(self, components: list[str] | None = None, weights: dict[str, float] | None = None):
        self.components = components or list(ALL_COMPONENTS)
        self.weights = weights or {c: 1.0 / len(self.components) for c in self.components}
        self.models: dict[str, Recommender] = {}

    def fit(self, ctx, models: dict[str, Recommender] | None = None):
        """`models` may carry already-fitted component models (avoids refitting)."""
        self.models = dict(models or {})
        self.pop = np.log1p(ctx.item_pop_decayed).astype(np.float32)
        self.trend = np.log1p(ctx.item_recent).astype(np.float32)
        self.types, self.groups = ctx.types, ctx.groups
        self.type_aff, self.group_aff = ctx.user_type_affinity, ctx.user_group_affinity
        self.has_history = ctx.has_history
        missing = [c for c in self.components if c in ("content", "collaborative", "matrix_factorization") and c not in self.models]
        if missing:
            raise ValueError(f"Hybrid needs fitted component models for: {missing}")
        return self

    # ------------------------------------------------------------------
    def raw_component(self, name: str, user_idx: np.ndarray) -> np.ndarray:
        b = len(user_idx)
        if name == "popularity":
            return np.tile(self.pop, (b, 1))
        if name == "recency":
            return np.tile(self.trend, (b, 1))
        if name == "category_affinity":
            t = self.type_aff[user_idx][:, self.types].toarray()
            g = self.group_aff[user_idx][:, self.groups].toarray()
            return (0.7 * t + 0.3 * g).astype(np.float32)
        return self.models[name].score_batch(user_idx)

    def component_scores(self, user_idx, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        user_idx = np.asarray(user_idx)
        return {c: minmax_rows(self.raw_component(c, user_idx), mask) for c in self.components}

    def combine(self, comps: dict[str, np.ndarray], user_idx, weights: dict[str, float] | None = None) -> np.ndarray:
        user_idx = np.asarray(user_idx)
        w = weights or self.weights
        out = np.zeros_like(next(iter(comps.values())))
        warm = self.has_history[user_idx]
        for c, s in comps.items():
            out[warm] += w.get(c, 0.0) * s[warm]
        # customers without history: only history-free components, re-weighted (fallback to pop+recency)
        cold_w = {c: w.get(c, 0.0) for c in comps if c in COLD_OK}
        if sum(cold_w.values()) <= 0:
            cold_w = {c: 1.0 for c in comps if c in COLD_OK}
        tot = sum(cold_w.values())
        for c, wt in cold_w.items():
            out[~warm] += (wt / tot) * comps[c][~warm]
        return out

    def score_batch(self, user_idx, mask: np.ndarray | None = None):
        user_idx = np.asarray(user_idx)
        return self.combine(self.component_scores(user_idx, mask), user_idx)

    def explain(self, user_idx: int, item_idx: int, mask: np.ndarray | None = None) -> dict[str, float]:
        """Weighted contribution of each component to one recommended item's final score."""
        comps = self.component_scores(np.array([user_idx]), None if mask is None else mask[None, :])
        warm = bool(self.has_history[user_idx])
        return {c: float((self.weights.get(c, 0.0) if warm or c in COLD_OK else 0.0) * comps[c][0, item_idx]) for c in comps}


def random_weights(components: list[str], rng: np.random.Generator) -> dict[str, float]:
    w = rng.dirichlet(np.full(len(components), 0.7))
    return dict(zip(components, map(float, w)))


def tune_weights(hybrid: HybridModel, user_idx: np.ndarray, mask: np.ndarray | None,
                 objective: Callable[[np.ndarray], float], n_trials: int, seed: int) -> tuple[dict[str, float], list[dict]]:
    """Random search over simplex weights, maximising `objective(score_matrix)` on VALIDATION users only.

    Component score matrices are computed once; each trial is just a weighted sum.
    """
    comps = hybrid.component_scores(user_idx, mask)
    rng = np.random.default_rng(seed)
    cands: list[dict[str, float]] = [dict(hybrid.weights)]
    for c in hybrid.components:  # single-component corner points
        cands.append({k: float(k == c) for k in hybrid.components})
    cands += [random_weights(hybrid.components, rng) for _ in range(n_trials)]
    log_rows, best, best_w = [], -np.inf, cands[0]
    for w in cands:
        val = objective(hybrid.combine(comps, user_idx, w))
        log_rows.append({"weights": w, "objective": val})
        if val > best:
            best, best_w = val, w
    log.info("Hybrid tuning: %d candidates, best objective=%.4f, weights=%s", len(cands), best,
             {k: round(v, 3) for k, v in best_w.items()})
    return best_w, log_rows
