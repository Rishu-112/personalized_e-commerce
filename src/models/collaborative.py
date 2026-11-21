"""Model C: item-based collaborative filtering on sparse implicit interactions."""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from src.models.base import Recommender

log = logging.getLogger("itemknn")


class ItemKNNModel(Recommender):
    """Cosine item-item similarity from the (users x items) recency-weighted matrix.

    Heavy users are down-weighted (1/sqrt(activity)); only the top-k neighbours per item are kept.
    Everything stays sparse: no dense items x items matrix is ever materialised.
    """
    name = "collaborative"

    def __init__(self, neighbors: int = 50, chunk: int = 1000):
        self.k = neighbors
        self.chunk = chunk

    def fit(self, ctx):
        W = ctx.weighted.tocsr().astype(np.float32)
        act = np.asarray((ctx.counts > 0).sum(axis=1)).ravel()
        scale = (1.0 / np.sqrt(np.maximum(act, 1))).astype(np.float32)
        Wd = sparse.diags(scale) @ W
        Xn = normalize(Wd.tocsc(), axis=0)          # L2-normalise item columns
        XT = Xn.T.tocsr()
        blocks = []
        for start in range(0, ctx.n_items, self.chunk):
            block = (XT[start:start + self.chunk] @ Xn).tocsr()
            blocks.append(self._topk_rows(block, start))
        self.sim = sparse.vstack(blocks).tocsr().astype(np.float32)
        self.weighted = W
        log.info("Item-kNN similarity: %d items, %d stored neighbour entries (k=%d)", ctx.n_items, self.sim.nnz, self.k)
        return self

    def _topk_rows(self, block: sparse.csr_matrix, offset: int) -> sparse.csr_matrix:
        data, indices, indptr = [], [], [0]
        for r in range(block.shape[0]):
            s, e = block.indptr[r], block.indptr[r + 1]
            cols, vals = block.indices[s:e], block.data[s:e]
            keep = cols != (offset + r)                       # drop self-similarity
            cols, vals = cols[keep], vals[keep]
            if len(vals) > self.k:
                top = np.argpartition(-vals, self.k - 1)[:self.k]
                cols, vals = cols[top], vals[top]
            indices.append(cols)
            data.append(vals)
            indptr.append(indptr[-1] + len(vals))
        d = np.concatenate(data) if data else np.array([], dtype=np.float32)
        i = np.concatenate(indices) if indices else np.array([], dtype=np.int32)
        return sparse.csr_matrix((d, i, indptr), shape=block.shape)

    def score_batch(self, user_idx):
        return (self.weighted[np.asarray(user_idx)] @ self.sim).toarray().astype(np.float32)

    def similar_items(self, item: int, n: int = 10):
        row = self.sim[item]
        order = np.argsort(-row.data)[:n]
        return [(int(row.indices[o]), float(row.data[o])) for o in order]
