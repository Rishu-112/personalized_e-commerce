"""RFM customer segmentation: StandardScaler + K-Means (k by silhouette) + PCA.

Segment names are assigned AFTER fitting, from the measured cluster profiles
(never assumed up-front), by `label_clusters`.

CLI: python -m src.segmentation --dataset demo
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.common import dataset_paths, load_config, set_seed, setup_logging

log = logging.getLogger("segmentation")


def compute_rfm(tx_hist: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Recency (days since last purchase), Frequency (# purchase lines), Monetary (total spend) as of `now`."""
    g = tx_hist.groupby("customer_id")
    return pd.DataFrame({
        "recency_days": (now - g["t_dat"].max()).dt.days,
        "frequency": g.size(),
        "monetary": g["price"].sum(),
    }).reset_index()


def fit_segments(rfm: pd.DataFrame, k_range=(3, 4, 5, 6), seed: int = 42, silhouette_sample: int = 5000):
    X = np.column_stack([rfm["recency_days"], np.log1p(rfm["frequency"]), np.log1p(rfm["monetary"] * 100)])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Xs), size=min(len(Xs), silhouette_sample), replace=False)
    scores, models = {}, {}
    for k in k_range:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        scores[k] = float(silhouette_score(Xs[sample], km.labels_[sample]))
        models[k] = km
    if not scores:
        raise ValueError("Not enough customers to cluster.")
    best_k = max(scores, key=scores.get)
    km = models[best_k]
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    coords = pca.transform(Xs)
    out = rfm.copy()
    out["cluster"] = km.labels_
    out["pca1"], out["pca2"] = coords[:, 0], coords[:, 1]
    meta = {"silhouette_by_k": scores, "chosen_k": best_k, "pca_explained_variance": pca.explained_variance_ratio_.tolist()}
    return out, meta


def label_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their measured mean R/F/M (profile index = cluster id)."""
    labels: dict[int, str] = {}
    left = set(profile.index)
    hv = profile.loc[list(left), "monetary"].idxmax()
    labels[hv] = "High-value customers"; left.discard(hv)
    if left:
        risk = profile.loc[list(left), "recency_days"].idxmax()
        labels[risk] = "At-risk customers"; left.discard(risk)
    if left:
        freq = profile.loc[list(left), "frequency"].idxmax()
        labels[freq] = "Frequent shoppers"; left.discard(freq)
    for n, c in enumerate(sorted(left), start=1):
        labels[c] = "Occasional shoppers" if len(left) == 1 else f"Occasional shoppers ({n})"
    return labels


def describe_segments(seg: pd.DataFrame, tx_hist: pd.DataFrame, articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = seg.groupby("segment").agg(customers=("customer_id", "size"), recency_days=("recency_days", "mean"),
                                         frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(3)
    summary["share_of_customers"] = (summary["customers"] / summary["customers"].sum()).round(3)
    spend = seg.groupby("segment")["monetary"].sum()
    summary["share_of_revenue"] = (spend / spend.sum()).round(3)
    df = tx_hist.merge(seg[["customer_id", "segment"]], on="customer_id").merge(
        articles[["article_id", "product_group_name"]], on="article_id", how="left")
    cat = df.pivot_table(index="segment", columns="product_group_name", values="article_id", aggfunc="size", fill_value=0)
    cat = cat.div(cat.sum(axis=1), axis=0).round(4)
    return summary.sort_values("monetary", ascending=False), cat


def run_segmentation(tx_hist, articles, now, cfg, out_dir=None):
    rfm = compute_rfm(tx_hist, now)
    seg, meta = fit_segments(rfm, cfg["segmentation"]["k_range"], cfg["seed"])
    profile = seg.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
    names = label_clusters(profile)
    seg["segment"] = seg["cluster"].map(names)
    summary, cat = describe_segments(seg, tx_hist, articles)
    meta["cluster_labels"] = {int(k): v for k, v in names.items()}
    meta["as_of"] = str(now.date())
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        seg.to_parquet(out_dir / "segments.parquet", index=False)
        summary.to_csv(out_dir / "segment_summary.csv")
        cat.to_csv(out_dir / "segment_category_preferences.csv")
        (out_dir / "segmentation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Segmentation: k=%d silhouette=%.3f -> %s", meta["chosen_k"], meta["silhouette_by_k"][meta["chosen_k"]], list(names.values()))
    return seg, summary, cat, meta


def main(argv=None):
    from src.data_preprocessing import load_processed
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["demo", "real"], default="demo")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    tx, articles, _, _ = load_processed(a.dataset, cfg)
    now = tx["t_dat"].max() + pd.Timedelta(days=1)
    _, summary, cat, meta = run_segmentation(tx, articles, now, cfg, dataset_paths(cfg, a.dataset)["artifacts"] / "segmentation")
    print(summary.to_string())


if __name__ == "__main__":
    main()
