"""Model D: implicit-feedback matrix factorization via Alternating Least Squares (ALS).

Uses the `implicit` library when it installs/imports (fast, multithreaded, CPU).
Otherwise falls back to a documented pure numpy/scipy ALS (Hu, Koren & Volinsky 2008)
— the fallback is NOT silent: `self.backend` records which one was used and it is logged.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from src.models.base import Recommender

log = logging.getLogger("mf")


def als_numpy(conf: sparse.csr_matrix, factors: int, reg: float, iters: int, alpha: float, seed: int):
    """Plain ALS for implicit data. `conf` holds raw interaction strengths r_ui (>0 where observed).

    Confidence c_ui = 1 + alpha * r_ui, preference p_ui = 1 where r_ui > 0.
    Each half-step solves, per row, (F^T F + F^T (C_u - I) F + reg*I) x = F^T C_u p_u.
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = conf.shape
    X = rng.normal(0, 0.01, (n_u, factors)).astype(np.float32)
    Y = rng.normal(0, 0.01, (n_i, factors)).astype(np.float32)
    R = conf.tocsr()
    RT = R.T.tocsr()
    eye = np.eye(factors, dtype=np.float32)

    def half_step(M: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        out = np.zeros((M.shape[0], factors), dtype=np.float32)
        FtF = fixed.T @ fixed + reg * eye
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e:
                continue
            F = fixed[M.indices[s:e]]
            c = alpha * M.data[s:e]
            A = FtF + (F.T * c) @ F
            b = F.T @ (1.0 + c)
            out[r] = np.linalg.solve(A, b)
        return out

    for it in range(iters):
        X = half_step(R, Y)
        Y = half_step(RT, X)
    return X, Y


class MatrixFactorizationModel(Recommender):
    name = "matrix_factorization"

    def __init__(self, factors: int = 32, regularization: float = 0.05, iterations: int = 15, alpha: float = 20.0,
                 prefer_implicit_library: bool = True, seed: int = 42):
        self.factors, self.reg, self.iterations, self.alpha = factors, regularization, iterations, alpha
        self.prefer_lib, self.seed = prefer_implicit_library, seed
        self.backend = None

    def fit(self, ctx):
        R = ctx.weighted.tocsr().astype(np.float32)
        fitted = False
        if self.prefer_lib:
            try:
                from implicit.cpu.als import AlternatingLeastSquares
                model = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, iterations=self.iterations,
                                                alpha=self.alpha, random_state=self.seed, num_threads=0)
                from threadpoolctl import threadpool_limits
                with threadpool_limits(1, "blas"):  # implicit recommends single-threaded BLAS
                    model.fit(R, show_progress=False)
                self.user_f = np.asarray(model.user_factors, dtype=np.float32)
                self.item_f = np.asarray(model.item_factors, dtype=np.float32)
                import implicit
                self.backend = f"implicit {implicit.__version__} (ALS, CPU)"
                fitted = True
            except Exception as exc:  # ImportError, DLL load failure, API change...
                log.warning("`implicit` unavailable or failed (%s). Falling back to built-in numpy ALS.", exc)
        if not fitted:
            self.user_f, self.item_f = als_numpy(R, self.factors, self.reg, self.iterations, self.alpha, self.seed)
            self.backend = "numpy ALS fallback (built-in)"
        log.info("Matrix factorization fitted with backend: %s", self.backend)
        return self

    def score_batch(self, user_idx):
        return (self.user_f[np.asarray(user_idx)] @ self.item_f.T).astype(np.float32)

    def similar_items(self, item: int, n: int = 10):
        v = self.item_f[item]
        norms = np.linalg.norm(self.item_f, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9
        sims = (self.item_f @ v) / norms
        sims[item] = -1
        idx = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in idx]
