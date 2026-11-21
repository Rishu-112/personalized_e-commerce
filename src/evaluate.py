"""Chronological offline evaluation of top-N recommenders.

Protocol (documented in README / docs/model_comparison.md)
  * history  = transactions strictly BEFORE the evaluation window
  * targets  = items each customer buys INSIDE the evaluation window
  * candidates = items purchased at least once in the history (catalog known at prediction time);
    targets outside the candidate universe are unreachable and reported, not silently dropped
  * previously purchased items are excluded from candidates when evaluation.exclude_seen is true
    (we evaluate discovery of NEW items); set it false to also reward repeat purchases
  * customers with no history ("new") fall back to popularity for every personalised model
CLI (final TEST evaluation of saved artifacts):  python -m src.evaluate --dataset demo
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import top_n
from src.utils.common import dataset_paths, load_config, setup_logging
from src.utils.metrics import (average_precision_at_k, catalog_coverage, intra_list_diversity, ndcg_at_k,
                               precision_at_k, recall_at_k)

log = logging.getLogger("evaluate")


@dataclass
class EvalSet:
    user_idx: np.ndarray
    relevant: list            # list[set[item_idx]]
    is_new: np.ndarray        # bool: customer had no history before the window
    unreachable_share: float  # share of target purchases on items outside the candidate universe
    n_targets: int


def build_eval_set(ctx, holdout: pd.DataFrame, max_users: int | None, seed: int) -> EvalSet:
    h = holdout[holdout["customer_id"].isin(ctx.user_index)]
    in_univ = h["article_id"].isin(ctx.item_index)
    unreachable = float(1 - in_univ.mean()) if len(h) else 0.0
    h = h[in_univ]
    rel = h.groupby("customer_id")["article_id"].apply(lambda s: {ctx.item_index[a] for a in s})
    users = np.array([ctx.user_index[c] for c in rel.index])
    rel_sets = list(rel.values)
    if max_users and len(users) > max_users:
        pick = np.random.default_rng(seed).choice(len(users), size=max_users, replace=False)
        users, rel_sets = users[pick], [rel_sets[i] for i in pick]
    return EvalSet(users, rel_sets, ~ctx.has_history[users], unreachable, int(sum(len(s) for s in rel_sets)))


def seen_mask(ctx, user_idx: np.ndarray) -> np.ndarray:
    return ctx.counts[user_idx].toarray() > 0


def score_for_eval(model, ctx, user_idx: np.ndarray, mask: np.ndarray | None, pop_scores: np.ndarray) -> np.ndarray:
    """Model scores with cold-start fallback to popularity for personalised models."""
    from src.models.hybrid import HybridModel
    s = model.score_batch(user_idx, mask) if isinstance(model, HybridModel) else model.score_batch(user_idx)
    if not isinstance(model, HybridModel):
        cold = ~ctx.has_history[user_idx]
        if cold.any():
            s = s.copy()
            s[cold] = pop_scores
    return s


def evaluate_model(model, ctx, es: EvalSet, ks=(5, 10), max_k: int = 20, exclude_seen: bool = True,
                   popular_items: np.ndarray | None = None, diversity_vecs=None, batch: int = 256) -> dict:
    pop_scores = np.log1p(ctx.item_pop_decayed)
    per_user = []
    all_rec10: list[list[int]] = []
    div_scores = []
    for s in range(0, len(es.user_idx), batch):
        ui = es.user_idx[s:s + batch]
        mask = seen_mask(ctx, ui) if exclude_seen else None
        scores = score_for_eval(model, ctx, ui, mask, pop_scores)
        for j, u in enumerate(ui):
            rel = es.relevant[s + j]
            rec = top_n(scores[j], max_k, None if mask is None else mask[j])
            rec_l = rec.tolist()
            row = {}
            for k in ks:
                row[f"precision@{k}"] = precision_at_k(rec_l, rel, k)
                row[f"recall@{k}"] = recall_at_k(rec_l, rel, k)
            row["map@10"] = average_precision_at_k(rec_l, rel, 10)
            row["ndcg@10"] = ndcg_at_k(rec_l, rel, 10)
            if popular_items is not None:
                head, tail = rel & set(popular_items.tolist()), rel - set(popular_items.tolist())
                row["recall@10_head"] = recall_at_k(rec_l, head, 10) if head else np.nan
                row["recall@10_tail"] = recall_at_k(rec_l, tail, 10) if tail else np.nan
            per_user.append(row)
            all_rec10.append(rec_l[:10])
            if diversity_vecs is not None:
                div_scores.append(intra_list_diversity(diversity_vecs, rec_l[:10]))
    df = pd.DataFrame(per_user)
    out = {c: float(df[c].mean()) for c in df.columns}   # NaN-aware mean (head/tail only over relevant users)
    out["coverage@10"] = catalog_coverage(all_rec10, ctx.n_items)
    out["diversity@10"] = float(np.mean(div_scores)) if div_scores else float("nan")
    core = ["precision@10", "recall@10", "map@10", "ndcg@10"]
    seg = {}
    for name, m in (("new_customers", es.is_new), ("returning_customers", ~es.is_new)):
        seg[name] = {"n_users": int(m.sum()), **({c: float(df.loc[m, c].mean()) for c in core} if m.any() else {})}
    out["by_customer_type"] = seg
    out["n_eval_users"] = int(len(es.user_idx))
    return out


def make_objective(ctx, es: EvalSet, mask: np.ndarray | None, k: int = 10, metric: str = "ndcg@10"):
    """Fast tuning objective: mean metric over the (fixed) tuning users for a score matrix."""
    fn = {"ndcg@10": ndcg_at_k, "map@10": average_precision_at_k, "recall@10": recall_at_k}[metric]

    def objective(scores: np.ndarray) -> float:
        vals = [fn(top_n(scores[j], k, None if mask is None else mask[j]).tolist(), es.relevant[j], k) for j in range(len(es.user_idx))]
        return float(np.mean(vals))
    return objective


def popular_item_set(ctx, quantile: float) -> np.ndarray:
    thr = np.quantile(ctx.item_pop_total, quantile)
    return np.where(ctx.item_pop_total >= thr)[0]


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    cols = ["precision@5", "precision@10", "recall@5", "recall@10", "map@10", "ndcg@10", "coverage@10", "diversity@10",
            "recall@10_head", "recall@10_tail"]
    rows = []
    for name, r in results.items():
        row = {"model": name, **{c: r.get(c, np.nan) for c in cols}}
        for seg, d in r.get("by_customer_type", {}).items():
            row[f"ndcg@10_{seg}"] = d.get("ndcg@10", np.nan)
            row[f"n_{seg}"] = d.get("n_users", 0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def make_charts(table: pd.DataFrame, out_dir, title_suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["ndcg@10", "recall@10", "map@10"]):
        table[col].sort_values().plot.barh(ax=ax, color="#1f6f78")
        ax.set_title(col.upper() + title_suffix)
        ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(out_dir / "metric_comparison.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(table["coverage@10"], table["diversity@10"], s=90, color="#c4452d")
    for n, r in table.iterrows():
        ax.annotate(n, (r["coverage@10"], r["diversity@10"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("catalog coverage@10"); ax.set_ylabel("intra-list diversity@10"); ax.set_title("Coverage vs diversity")
    fig.tight_layout(); fig.savefig(out_dir / "coverage_diversity.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def evaluate_all(models: dict, ctx, holdout: pd.DataFrame, cfg: dict, diversity_vecs, seed: int) -> tuple[dict, EvalSet]:
    ev = cfg["evaluation"]
    es = build_eval_set(ctx, holdout, ev["max_eval_users"], seed)
    if len(es.user_idx) == 0:
        raise ValueError("No evaluation users with held-out purchases inside the candidate universe.")
    popular = popular_item_set(ctx, ev["popular_item_quantile"])
    results = {name: evaluate_model(m, ctx, es, tuple(ev["ks"]), ev["max_k"], ev["exclude_seen"], popular, diversity_vecs)
               for name, m in models.items()}
    return results, es


def run_test(dataset: str, cfg: dict) -> pd.DataFrame:
    """Final held-out TEST evaluation of saved artifacts (models fitted on train+val)."""
    import joblib
    from src.data_preprocessing import load_processed
    paths = dataset_paths(cfg, dataset)
    bundle_path = paths["artifacts"] / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"No trained artifacts at {bundle_path}. Run: python -m src.train --dataset {dataset}")
    bundle = joblib.load(bundle_path)
    tx, _, _, _ = load_processed(dataset, cfg)
    test = tx[tx["t_dat"] >= bundle["meta"]["test_start"]]
    ctx = bundle["ctx"]
    results, es = evaluate_all(bundle["models"], ctx, test, cfg, bundle["models"]["content"].item_vecs, cfg["seed"])
    table = results_table(results)
    meta = {"dataset": dataset, "is_synthetic": paths["is_synthetic"], "split": "test (held-out, evaluated once after model selection)",
            "test_start": str(bundle["meta"]["test_start"].date()), "history_end": str(ctx.now.date()),
            "n_eval_users": int(len(es.user_idx)), "n_new_customers": int(es.is_new.sum()),
            "unreachable_target_share": es.unreachable_share, "candidate_items": int(ctx.n_items),
            "exclude_seen": cfg["evaluation"]["exclude_seen"], "hybrid_weights": bundle["meta"]["hybrid_weights"],
            "mf_backend": bundle["meta"]["mf_backend"]}
    out = paths["artifacts"]
    (out / "test_results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2, default=float), encoding="utf-8")
    table.to_csv(out / "test_metrics.csv")
    make_charts(table, out / "charts", " (test)")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["demo", "real"], default="demo")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    try:
        table = run_test(a.dataset, cfg)
    except FileNotFoundError as e:
        log.error("%s", e)
        raise SystemExit(2)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
