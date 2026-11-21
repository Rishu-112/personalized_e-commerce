"""Model A: popularity baselines (global and category-specific)."""
from __future__ import annotations

import numpy as np

from src.models.base import Recommender


class PopularityModel(Recommender):
    """mode='global'   : same recency-decayed popularity ranking for everyone.
    mode='category' : popularity re-weighted by each customer's category affinity
                      (share of their recency-weighted purchases per product type)."""

    def __init__(self, mode: str = "global"):
        assert mode in ("global", "category")
        self.mode = mode
        self.name = "popularity" if mode == "global" else "popularity_category"

    def fit(self, ctx):
        self.pop = np.log1p(ctx.item_pop_decayed).astype(np.float32)
        self.types = ctx.types
        self.type_names = ctx.type_names
        self.user_aff = ctx.user_type_affinity
        self.has_history = ctx.has_history
        return self

    def score_batch(self, user_idx):
        user_idx = np.asarray(user_idx)
        base = np.tile(self.pop, (len(user_idx), 1))
        if self.mode == "global":
            return base
        aff = self.user_aff[user_idx]                      # (b, types)
        item_aff = aff[:, self.types].toarray()            # (b, items)
        out = base * (0.05 + item_aff)                     # small floor keeps weak-signal users sensible
        cold = ~self.has_history[user_idx]
        out[cold] = base[cold]
        return out.astype(np.float32)

    def recommend_for_categories(self, categories: list[str], n: int) -> list[tuple[int, float]]:
        """Popularity restricted to chosen product types (used for cold-start)."""
        names = np.array(self.type_names)[self.types]
        s = np.where(np.isin(names, categories), self.pop, -np.inf)
        idx = np.argsort(-s)[:n]
        return [(int(i), float(self.pop[i])) for i in idx if np.isfinite(s[i])]
