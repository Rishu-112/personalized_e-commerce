import pandas as pd

from src.segmentation import compute_rfm, fit_segments, label_clusters


def test_rfm_computed_from_history(split):
    hist = pd.concat([split["train"], split["val"]])
    rfm = compute_rfm(hist, split["test_start"])
    assert (rfm["recency_days"] >= 1).all() and (rfm["frequency"] >= 1).all()


def test_segments_and_labels(split):
    hist = pd.concat([split["train"], split["val"]])
    rfm = compute_rfm(hist, split["test_start"])
    seg, meta = fit_segments(rfm, (3, 4), seed=0)
    assert meta["chosen_k"] in (3, 4) and -1 <= meta["silhouette_by_k"][meta["chosen_k"]] <= 1
    prof = seg.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
    names = label_clusters(prof)
    assert set(names) == set(prof.index)
    assert names[prof["monetary"].idxmax()] == "High-value customers"
