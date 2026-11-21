"""Inference layer used by the Streamlit app and by tests.

Loads the fitted bundle and turns model scores into ranked, explained recommendations.
Every recommendation shown in the UI comes from here (no hardcoded examples).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.base import minmax_rows, top_n
from src.models.hybrid import COLD_OK, HybridModel
from src.utils.common import dataset_paths, load_config

log = logging.getLogger("inference")

ALGORITHMS = {
    "hybrid": "Hybrid ranking",
    "collaborative": "Item-based collaborative filtering",
    "matrix_factorization": "Matrix factorization (ALS)",
    "content": "Content-based (TF-IDF)",
    "popularity_category": "Category-aware popularity",
    "popularity": "Global popularity",
}
DISPLAY_COLS = ["article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name", "department_name"]


class ArtifactsMissingError(FileNotFoundError):
    pass


class RecommenderService:
    def __init__(self, bundle: dict, artifacts_dir: Path):
        self.ctx = bundle["ctx"]
        self.models = bundle["models"]
        self.meta = bundle["meta"]
        self.dir = artifacts_dir
        self.items = self.ctx.items
        self.price_by_item = self.ctx.item_price_mean
        self.hybrid: HybridModel = self.models["hybrid"]
        self.content = self.models["content"]
        self.is_synthetic = bool(self.meta.get("is_synthetic"))

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, dataset: str = "demo", cfg: dict | None = None) -> "RecommenderService":
        cfg = cfg or load_config()
        d = dataset_paths(cfg, dataset)["artifacts"]
        p = d / "bundle.joblib"
        if not p.exists():
            raise ArtifactsMissingError(f"No trained model artifacts at {p}. Run:\n"
                                        f"  python -m src.data_preprocessing --dataset {dataset}\n  python -m src.train --dataset {dataset}")
        return cls(joblib.load(p), d)

    # ---------------------------------------------------------------- helpers
    def has_customer(self, customer_id: str) -> bool:
        return customer_id in self.ctx.user_index

    def customer_idx(self, customer_id: str) -> int:
        if customer_id not in self.ctx.user_index:
            raise KeyError(f"Unknown customer_id: {customer_id}")
        return self.ctx.user_index[customer_id]

    def item_row(self, item_idx: int) -> dict:
        r = self.items.iloc[item_idx]
        d = {c: r[c] for c in DISPLAY_COLS if c in self.items.columns}
        d["item_idx"] = int(item_idx)
        d["avg_price"] = float(self.price_by_item[item_idx])
        d["detail_desc"] = r.get("detail_desc", "") if "detail_desc" in self.items.columns else ""
        return d

    def history(self, customer_id: str, n: int = 50) -> pd.DataFrame:
        """Customer's purchases (most recent first), straight from the training history."""
        u = self.customer_idx(customer_id)
        t = self.ctx.tx[self.ctx.tx["u"] == u].sort_values("age_days").head(n)
        rows = []
        for _, r in t.iterrows():
            d = self.item_row(int(r["i"]))
            d["days_ago"] = int(r["age_days"] - 1)
            d["price"] = float(r["price"])
            rows.append(d)
        return pd.DataFrame(rows)

    def _mask(self, u: int, exclude_seen: bool) -> np.ndarray | None:
        return (self.ctx.counts[u].toarray().ravel() > 0) if exclude_seen else None

    def _scores(self, algo: str, u: int, mask: np.ndarray | None) -> np.ndarray:
        model = self.models[algo]
        m2 = None if mask is None else mask
        if algo == "hybrid":
            return model.score_batch(np.array([u]), None if m2 is None else m2[None, :])[0]
        if not self.ctx.has_history[u]:               # new customer -> popularity fallback
            return self.models["popularity"].score_user(u)
        return model.score_user(u)

    # ------------------------------------------------------------ recommend
    def recommend(self, customer_id: str, n: int = 10, algo: str = "hybrid", exclude_seen: bool = True) -> pd.DataFrame:
        """Top-n items for an existing customer, with scores and a human-readable explanation."""
        if algo not in self.models:
            raise ValueError(f"Unknown algorithm '{algo}'. Choose from {list(self.models)}")
        u = self.customer_idx(customer_id)
        mask = self._mask(u, exclude_seen)
        scores = self._scores(algo, u, mask)
        idx = top_n(scores, n, None if mask is None else mask)
        norm = minmax_rows(scores[None, :], None if mask is None else mask[None, :])[0]
        rows = []
        for rank, i in enumerate(idx, start=1):
            d = self.item_row(int(i))
            d.update(rank=rank, score=float(scores[i]), score_normalised=float(norm[i]),
                     explanation=self.explain(u, int(i), algo, mask))
            rows.append(d)
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- explanations
    def _anchor(self, u: int, i: int, sim_fn) -> tuple[str, float] | None:
        """Purchased item of customer u that is most similar to candidate i under sim_fn(item)->array."""
        hist = np.where(self.ctx.counts[u].toarray().ravel() > 0)[0]
        if len(hist) == 0:
            return None
        sims = np.array([sim_fn(h, i) for h in hist])
        j = int(np.argmax(sims))
        if sims[j] <= 0:
            return None
        return self.items.iloc[hist[j]].get("prod_name", str(self.ctx.item_ids[hist[j]])), float(sims[j])

    def explain(self, u: int, i: int, algo: str, mask: np.ndarray | None = None) -> str:
        it = self.items.iloc[i]
        cat = it["product_type_name"]
        aff = float(self.ctx.user_type_affinity[u, self.ctx.types[i]])
        if not self.ctx.has_history[u] and algo != "popularity":
            return f"New customer: no history yet, so we show currently popular items (popular {cat})."
        if algo == "popularity":
            return f"One of the most purchased items recently ({int(self.ctx.item_pop_total[i])} purchases in history)."
        if algo == "popularity_category":
            return f"Popular {cat}; {aff:.0%} of your recent purchases are {cat}."
        if algo == "content":
            a = self._anchor(u, i, lambda h, j: float((self.content.item_vecs[h] @ self.content.item_vecs[j].T).toarray()[0, 0]))
            return f"Similar attributes to '{a[0]}' you bought (cosine {a[1]:.2f}); {cat}." if a else f"Matches the attributes of your history ({cat})."
        if algo == "collaborative":
            sim = self.models["collaborative"].sim
            a = self._anchor(u, i, lambda h, j: float(sim[h, j]))
            return f"Customers who bought '{a[0]}' also bought this (item similarity {a[1]:.2f})." if a else "Frequently bought by customers with similar baskets."
        if algo == "matrix_factorization":
            mf = self.models["matrix_factorization"]
            a = self._anchor(u, i, lambda h, j: float(mf.item_f[h] @ mf.item_f[j] / (np.linalg.norm(mf.item_f[h]) * np.linalg.norm(mf.item_f[j]) + 1e-9)))
            base = "Matches your latent taste profile learned from customers with similar purchases"
            return base + (f" (closest to '{a[0]}')." if a else ".")
        if algo == "hybrid":
            contrib = self.hybrid.explain(u, i, None if mask is None else mask)
            total = sum(contrib.values()) or 1.0
            top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
            names = {"popularity": "popularity", "content": "content similarity", "collaborative": "collaborative filtering",
                     "matrix_factorization": "matrix factorization", "category_affinity": f"your {cat} affinity ({aff:.0%})", "recency": "recent trend"}
            return "Driven by " + ", ".join(f"{names[k]} ({v / total:.0%})" for k, v in top if v > 0) + "."
        return ""

    # ------------------------------------------------------- similar products
    def similar_items(self, item_idx: int, n: int = 10, method: str = "hybrid") -> pd.DataFrame:
        """method: content | collaborative | matrix_factorization | hybrid (average of the three, normalised)."""
        def sims(m):
            if m == "content":
                s = (self.content.item_vecs[item_idx] @ self.content.item_vecs.T).toarray().ravel()
            elif m == "collaborative":
                s = self.models["collaborative"].sim[item_idx].toarray().ravel()
            else:
                f = self.models["matrix_factorization"].item_f
                s = (f @ f[item_idx]) / (np.linalg.norm(f, axis=1) * np.linalg.norm(f[item_idx]) + 1e-9)
            return s.astype(np.float64)
        if method == "hybrid":
            parts = []
            for m in ("content", "collaborative", "matrix_factorization"):
                s = sims(m); s[item_idx] = -np.inf
                finite = np.isfinite(s)
                lo, hi = s[finite].min(), s[finite].max()
                parts.append(np.where(finite, (s - lo) / (hi - lo + 1e-12), 0))
            s = np.mean(parts, axis=0)
        else:
            s = sims(method)
        s[item_idx] = -np.inf
        idx = np.argsort(-s)[:n]
        rows = []
        for r, i in enumerate(idx, start=1):
            d = self.item_row(int(i)); d.update(rank=r, similarity=float(s[i])); rows.append(d)
        return pd.DataFrame(rows)

    def search_items(self, query: str, limit: int = 30) -> pd.DataFrame:
        q = query.strip().lower()
        if not q:
            return pd.DataFrame()
        hay = (self.items["prod_name"].astype(str) + " " + self.items["product_type_name"].astype(str) + " " +
               self.items["colour_group_name"].astype(str) + " " + self.items["article_id"].astype(str)).str.lower()
        hits = np.where(hay.str.contains(q, regex=False))[0][:limit]
        return pd.DataFrame([self.item_row(int(i)) for i in hits])

    # ---------------------------------------------------------- cold start
    def cold_start_options(self) -> dict[str, list[str]]:
        pop = pd.Series(self.ctx.item_pop_total, index=np.arange(self.ctx.n_items))
        by_group = pop.groupby(self.items["product_group_name"].values).sum().sort_values(ascending=False)
        by_type = pop.groupby(self.items["product_type_name"].values).sum().sort_values(ascending=False)
        return {"groups": by_group.index.tolist(), "types": by_type.index.tolist(),
                "colours": sorted(self.items["colour_group_name"].dropna().unique().tolist())}

    def recommend_cold_start(self, groups: list[str] | None = None, types: list[str] | None = None,
                             colours: list[str] | None = None, n: int = 10) -> pd.DataFrame:
        """Recommendations for a brand-new visitor from stated preferences only.

        score = 0.6 * normalised popularity (decayed) + 0.4 * normalised content similarity to a pseudo-profile
        built from the *most popular items matching the stated preferences* (blended with trending items).
        """
        groups, types, colours = groups or [], types or [], colours or []
        ok = np.ones(self.ctx.n_items, dtype=bool)
        if groups:
            ok &= self.items["product_group_name"].isin(groups).to_numpy()
        if types:
            ok &= self.items["product_type_name"].isin(types).to_numpy()
        if not ok.any():
            return pd.DataFrame()
        pop = np.log1p(self.ctx.item_pop_decayed + 0.5 * self.ctx.item_recent)
        # pseudo-profile: top popular items inside the selected categories (+ colour match boost)
        colour_ok = self.items["colour_group_name"].isin(colours).to_numpy() if colours else np.zeros(self.ctx.n_items, bool)
        seed_score = np.where(ok, pop * (1 + 0.5 * colour_ok), -np.inf)
        seeds = np.argsort(-seed_score)[:10]
        seeds = seeds[np.isfinite(seed_score[seeds])]
        profile = self.content.profile_from_items(seeds.tolist(), weights=pop[seeds] + 1e-6)
        sim = (profile @ self.content.item_vecs.T).toarray().ravel()
        def mm(x):
            v = x[ok]; return np.where(ok, (x - v.min()) / (v.max() - v.min() + 1e-12), 0)
        score = 0.6 * mm(pop) + 0.4 * mm(sim) + 0.1 * colour_ok * ok
        score[~ok] = -np.inf
        idx = top_n(score, n)
        rows = []
        for r, i in enumerate(idx, start=1):
            it = self.items.iloc[int(i)]
            why = f"Popular {it['product_type_name']} in {it['product_group_name']}"
            if colours and it["colour_group_name"] in colours:
                why += f", in your chosen colour ({it['colour_group_name']})"
            d = self.item_row(int(i)); d.update(rank=r, score=float(score[i]), explanation=why + "."); rows.append(d)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ reporting
    def read_json(self, name: str) -> dict | None:
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
