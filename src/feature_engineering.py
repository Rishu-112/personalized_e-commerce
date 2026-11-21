"""Feature engineering + the `Context` object shared by all recommenders.

Leakage rule: every function here receives ONLY transactions strictly before the
prediction date (`now`) and computes ages/recency relative to `now`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

log = logging.getLogger("features")

TEXT_COLS = ["product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name",
             "perceived_colour_master_name", "department_name", "index_group_name", "section_name",
             "garment_group_name", "detail_desc"]


# ------------------------------------------------------------------ splitting
def chronological_split(tx: pd.DataFrame, val_days: int, test_days: int) -> dict[str, pd.DataFrame | pd.Timestamp]:
    """train < val_start <= val < test_start <= test. Windows are counted back from the last date."""
    last = tx["t_dat"].max().normalize()
    test_start = last - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    out = {
        "train": tx[tx["t_dat"] < val_start],
        "val": tx[(tx["t_dat"] >= val_start) & (tx["t_dat"] < test_start)],
        "test": tx[tx["t_dat"] >= test_start],
        "val_start": val_start,
        "test_start": test_start,
    }
    if len(out["train"]) == 0 or len(out["val"]) == 0 or len(out["test"]) == 0:
        raise ValueError("Chronological split produced an empty partition; adjust split.val_days/test_days.")
    return out


# --------------------------------------------------------------------- context
@dataclass
class Context:
    """Index spaces + sparse matrices built from history strictly before `now`."""
    now: pd.Timestamp
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict
    item_index: dict
    items: pd.DataFrame                      # articles aligned with item_ids
    tx: pd.DataFrame                         # columns: u, i, age_days, price (history only)
    counts: sparse.csr_matrix                # users x items raw purchase counts
    weighted: sparse.csr_matrix              # users x items recency-decayed counts
    types: np.ndarray                        # item -> product type code
    type_names: list
    groups: np.ndarray                       # item -> product group code
    group_names: list
    user_type_affinity: sparse.csr_matrix    # users x types, rows sum to 1 (recency weighted)
    user_group_affinity: sparse.csr_matrix   # users x groups
    item_pop_decayed: np.ndarray             # exp-decayed popularity (half-life from config)
    item_pop_total: np.ndarray
    item_recent: np.ndarray                  # purchases within trending window
    item_price_mean: np.ndarray
    has_history: np.ndarray = field(default=None)  # bool per user
    half_life_days: float = 60.0

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def text(self) -> list[str]:
        cols = [c for c in TEXT_COLS if c in self.items.columns]
        return self.items[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().tolist()


def build_context(tx_hist: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame, now: pd.Timestamp,
                  half_life_days: float = 60.0, trending_days: int = 14, pop_half_life_days: float = 30.0) -> Context:
    """Build matrices from `tx_hist` (all rows must satisfy t_dat < now)."""
    if len(tx_hist) and tx_hist["t_dat"].max() >= now:
        raise ValueError("Leakage guard: history contains transactions on/after the prediction date.")
    # candidate item universe = items purchased in history (documented in docs/model_comparison.md)
    item_ids = np.sort(tx_hist["article_id"].unique())
    user_ids = customers["customer_id"].to_numpy()
    item_index = {a: i for i, a in enumerate(item_ids)}
    user_index = {c: i for i, c in enumerate(user_ids)}
    items = articles.set_index("article_id").loc[item_ids].reset_index()

    h = tx_hist[tx_hist["customer_id"].isin(user_index)].copy()
    h["u"] = h["customer_id"].map(user_index).astype(np.int64)
    h["i"] = h["article_id"].map(item_index).astype(np.int64)
    h["age_days"] = (now - h["t_dat"]).dt.days.astype(np.float32) + 1.0
    tx = h[["u", "i", "age_days", "price"]].reset_index(drop=True)

    n_u, n_i = len(user_ids), len(item_ids)
    w = np.exp(-np.log(2) * tx["age_days"].to_numpy() / half_life_days).astype(np.float32)
    shape = (n_u, n_i)
    counts = sparse.coo_matrix((np.ones(len(tx), np.float32), (tx["u"], tx["i"])), shape=shape).tocsr()
    weighted = sparse.coo_matrix((w, (tx["u"], tx["i"])), shape=shape).tocsr()

    type_codes, type_names = pd.factorize(items["product_type_name"])
    group_codes, group_names = pd.factorize(items["product_group_name"])

    def affinity(codes: np.ndarray, k: int) -> sparse.csr_matrix:
        m = sparse.coo_matrix((w, (tx["u"], codes[tx["i"].to_numpy()])), shape=(n_u, k)).tocsr()
        return normalize(m, norm="l1")

    pw = np.exp(-np.log(2) * tx["age_days"].to_numpy() / pop_half_life_days)
    item_pop_decayed = np.bincount(tx["i"], weights=pw, minlength=n_i).astype(np.float32)
    item_pop_total = np.bincount(tx["i"], minlength=n_i).astype(np.float32)
    recent_mask = tx["age_days"].to_numpy() <= trending_days
    item_recent = np.bincount(tx["i"][recent_mask], minlength=n_i).astype(np.float32)
    price_sum = np.bincount(tx["i"], weights=tx["price"].to_numpy(), minlength=n_i)
    price_mean = np.divide(price_sum, item_pop_total, out=np.zeros(n_i), where=item_pop_total > 0).astype(np.float32)

    ctx = Context(
        now=now, user_ids=user_ids, item_ids=item_ids, user_index=user_index, item_index=item_index, items=items, tx=tx,
        counts=counts, weighted=weighted, types=type_codes, type_names=list(type_names), groups=group_codes,
        group_names=list(group_names), user_type_affinity=affinity(type_codes, len(type_names)),
        user_group_affinity=affinity(group_codes, len(group_names)), item_pop_decayed=item_pop_decayed,
        item_pop_total=item_pop_total, item_recent=item_recent, item_price_mean=price_mean,
        has_history=np.asarray(counts.sum(axis=1)).ravel() > 0, half_life_days=half_life_days,
    )
    log.info("Context: %d users (%d with history), %d items, %d interactions, now=%s",
             n_u, int(ctx.has_history.sum()), n_i, len(tx), now.date())
    return ctx


# ------------------------------------------------------------ feature tables
def customer_features(tx_hist: pd.DataFrame, articles: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Per-customer RFM + behavioural features, using only history before `now`."""
    df = tx_hist.merge(articles[["article_id", "product_type_name", "product_group_name"]], on="article_id", how="left")
    g = df.groupby("customer_id")
    feats = pd.DataFrame({
        "frequency": g.size(),
        "recency_days": (now - g["t_dat"].max()).dt.days,
        "monetary": g["price"].sum(),
        "avg_price": g["price"].mean(),
        "n_unique_items": g["article_id"].nunique(),
        "n_active_days": g["t_dat"].nunique(),
        "tenure_days": (g["t_dat"].max() - g["t_dat"].min()).dt.days,
    })

    def entropy(s: pd.Series) -> float:
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log(p)).sum())

    feats["category_diversity"] = g["product_type_name"].apply(entropy)
    feats["preferred_group"] = g["product_group_name"].agg(lambda s: s.value_counts().index[0])
    feats["preferred_type"] = g["product_type_name"].agg(lambda s: s.value_counts().index[0])
    return feats.reset_index()


def product_features(tx_hist: pd.DataFrame, articles: pd.DataFrame, now: pd.Timestamp, trending_days: int = 14) -> pd.DataFrame:
    g = tx_hist.groupby("article_id")
    f = pd.DataFrame({
        "popularity": g.size(),
        "n_buyers": g["customer_id"].nunique(),
        "price_mean": g["price"].mean(), "price_std": g["price"].std().fillna(0.0),
        "price_min": g["price"].min(), "price_max": g["price"].max(),
        "last_purchase_days": (now - g["t_dat"].max()).dt.days,
    })
    recent = tx_hist[tx_hist["t_dat"] >= now - pd.Timedelta(days=trending_days)].groupby("article_id").size()
    f["recent_popularity"] = recent.reindex(f.index).fillna(0).astype(int)
    f["repeat_rate"] = 1 - f["n_buyers"] / f["popularity"]
    keep = [c for c in ["article_id", "product_type_name", "product_group_name", "colour_group_name", "department_name"] if c in articles.columns]
    return f.reset_index().merge(articles[keep], on="article_id", how="left")


def interaction_features(ctx: Context) -> pd.DataFrame:
    """Customer-item frequency and recency-weighted score (sparse -> tidy)."""
    coo = ctx.weighted.tocoo()
    cnt = ctx.counts.tocsr()
    return pd.DataFrame({
        "customer_id": ctx.user_ids[coo.row], "article_id": ctx.item_ids[coo.col],
        "frequency": np.asarray(cnt[coo.row, coo.col]).ravel(), "recency_weighted_score": coo.data,
    })
