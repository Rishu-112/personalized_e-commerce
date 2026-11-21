"""Model B: content-based filtering with TF-IDF + cosine similarity.

Optional sentence embeddings are intentionally not required (heavy dependency);
TF-IDF over structured metadata + descriptions is the default and is fully offline.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.models.base import Recommender


class ContentBasedModel(Recommender):
    name = "content"

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)

    def fit(self, ctx):
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range,
                                          sublinear_tf=True, stop_words="english", min_df=1)
        self.item_vecs = normalize(self.vectorizer.fit_transform(ctx.text())).tocsr().astype(np.float32)
        self.weighted = ctx.weighted  # recency-weighted purchase history
        return self

    def user_profiles(self, user_idx) -> sparse.csr_matrix:
        prof = self.weighted[np.asarray(user_idx)] @ self.item_vecs
        return normalize(prof, norm="l2")

    def score_batch(self, user_idx):
        prof = self.user_profiles(user_idx)
        return (prof @ self.item_vecs.T).toarray().astype(np.float32)

    def profile_from_items(self, item_idx: list[int], weights=None) -> sparse.csr_matrix:
        w = np.ones(len(item_idx)) if weights is None else np.asarray(weights, dtype=float)
        v = sparse.csr_matrix(w[None, :]) @ self.item_vecs[item_idx]
        return normalize(v, norm="l2")

    def similar_items(self, item: int, n: int = 10):
        sims = (self.item_vecs[item] @ self.item_vecs.T).toarray().ravel()
        sims[item] = -1
        idx = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in idx]
