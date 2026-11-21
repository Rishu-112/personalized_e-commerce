"""Train, select and persist the recommenders.

Stages
  1. fit all models on TRAIN history, evaluate on VALIDATION, tune hybrid weights on VALIDATION only
  2. refit with the same hyper-parameters on TRAIN+VAL history (prediction date = test start)
  3. save artifacts, then (unless --skip-test) run the one-off TEST evaluation (src.evaluate)

CLI: python -m src.train --dataset demo
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import pandas as pd

from src import evaluate as ev
from src.data_preprocessing import DatasetMissingError, load_processed
from src.feature_engineering import build_context, chronological_split, customer_features, product_features
from src.models.collaborative import ItemKNNModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridModel, tune_weights
from src.models.matrix_factorization import MatrixFactorizationModel
from src.models.popularity import PopularityModel
from src.segmentation import run_segmentation
from src.utils.common import dataset_paths, load_config, set_seed, setup_logging

log = logging.getLogger("train")


def fit_models(ctx, cfg: dict, hybrid_weights: dict | None = None) -> dict:
    m = cfg["models"]
    t0 = time.time()
    models = {
        "popularity": PopularityModel("global").fit(ctx),
        "popularity_category": PopularityModel("category").fit(ctx),
        "content": ContentBasedModel(m["content"]["max_features"], m["content"]["ngram_range"]).fit(ctx),
        "collaborative": ItemKNNModel(m["collaborative"]["neighbors"], m["collaborative"]["chunk"]).fit(ctx),
        "matrix_factorization": MatrixFactorizationModel(
            m["matrix_factorization"]["factors"], m["matrix_factorization"]["regularization"],
            m["matrix_factorization"]["iterations"], m["matrix_factorization"]["alpha"],
            m["matrix_factorization"]["prefer_implicit_library"], cfg["seed"]).fit(ctx),
    }
    hyb = HybridModel(m["hybrid"]["components"], hybrid_weights)
    hyb.fit(ctx, {k: models[k] for k in ("content", "collaborative", "matrix_factorization")})
    models["hybrid"] = hyb
    log.info("Fitted %d models in %.1fs", len(models), time.time() - t0)
    return models


def run(dataset: str, cfg: dict, skip_test: bool = False) -> dict:
    paths = dataset_paths(cfg, dataset)
    tx, articles, customers, pmeta = load_processed(dataset, cfg)
    split = chronological_split(tx, cfg["split"]["val_days"], cfg["split"]["test_days"])
    fe = cfg["features"]
    ev_cfg = cfg["evaluation"]
    log.info("Split | train: %d tx (<%s) | val: %d tx | test: %d tx (>=%s)", len(split["train"]), split["val_start"].date(),
             len(split["val"]), len(split["test"]), split["test_start"].date())

    # ---------------- stage 1: train -> validation (model selection) --------
    ctx_tr = build_context(split["train"], articles, customers, split["val_start"], fe["recency_half_life_days"],
                           fe["trending_window_days"], cfg["models"]["popularity"]["half_life_days"])
    models = fit_models(ctx_tr, cfg)
    es_val = ev.build_eval_set(ctx_tr, split["val"], ev_cfg["max_eval_users"], cfg["seed"])
    if len(es_val.user_idx) == 0:
        raise ValueError("No validation users with purchases in the candidate universe.")
    hcfg = cfg["models"]["hybrid"]
    tune_n = min(hcfg["max_tune_users"], len(es_val.user_idx))
    tune_es = ev.EvalSet(es_val.user_idx[:tune_n], es_val.relevant[:tune_n], es_val.is_new[:tune_n], es_val.unreachable_share, 0)
    mask = ev.seen_mask(ctx_tr, tune_es.user_idx) if ev_cfg["exclude_seen"] else None
    objective = ev.make_objective(ctx_tr, tune_es, mask, 10, hcfg["tuning_metric"])
    best_w, tune_log = tune_weights(models["hybrid"], tune_es.user_idx, mask, objective, hcfg["n_trials"], cfg["seed"])
    models["hybrid"].weights = best_w

    popular = ev.popular_item_set(ctx_tr, ev_cfg["popular_item_quantile"])
    val_results = {n: ev.evaluate_model(m, ctx_tr, es_val, tuple(ev_cfg["ks"]), ev_cfg["max_k"], ev_cfg["exclude_seen"], popular,
                                        models["content"].item_vecs) for n, m in models.items()}
    val_table = ev.results_table(val_results)
    selected = val_table["ndcg@10"].idxmax()
    log.info("Validation NDCG@10 ranking:\n%s", val_table["ndcg@10"].sort_values(ascending=False).round(4).to_string())

    # ---------------- stage 2: refit on train+val, predict date = test start --
    hist_final = pd.concat([split["train"], split["val"]], ignore_index=True)
    ctx_fin = build_context(hist_final, articles, customers, split["test_start"], fe["recency_half_life_days"],
                            fe["trending_window_days"], cfg["models"]["popularity"]["half_life_days"])
    final_models = fit_models(ctx_fin, cfg, hybrid_weights=best_w)

    meta = {"dataset": dataset, "is_synthetic": paths["is_synthetic"], "selected_model_on_validation": selected,
            "hybrid_weights": best_w, "mf_backend": final_models["matrix_factorization"].backend,
            "val_start": split["val_start"], "test_start": split["test_start"], "history_end": ctx_fin.now,
            "seed": cfg["seed"], "n_users": ctx_fin.n_users, "n_items": ctx_fin.n_items,
            "val_unreachable_target_share": es_val.unreachable_share}
    out = paths["artifacts"]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ctx": ctx_fin, "models": final_models, "meta": meta}, out / "bundle.joblib", compress=3)

    # ---------------- reports --------------------------------------------------
    val_table.to_csv(out / "validation_metrics.csv")
    (out / "validation_results.json").write_text(json.dumps({"meta": {"split": "validation", "n_eval_users": len(es_val.user_idx),
                                                                     "unreachable_target_share": es_val.unreachable_share,
                                                                     "note": "hybrid weights were tuned on these validation users, so its validation score is optimistic; use TEST for the final comparison"},
                                                          "results": val_results, "hybrid_tuning_trials": tune_log}, indent=2, default=float), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    ev.make_charts(val_table, out / "charts_validation", " (validation)")
    customer_features(hist_final, articles, ctx_fin.now).to_parquet(out / "customer_features.parquet", index=False)
    product_features(hist_final, articles, ctx_fin.now, fe["trending_window_days"]).to_parquet(out / "product_features.parquet", index=False)
    run_segmentation(hist_final, articles, ctx_fin.now, cfg, out / "segmentation")

    if not skip_test:
        table = ev.run_test(dataset, cfg)
        log.info("TEST metrics:\n%s", table[["precision@10", "recall@10", "map@10", "ndcg@10", "coverage@10", "diversity@10"]].round(4).to_string())
    log.info("Artifacts saved to %s", out)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["demo", "real"], default="demo")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-test", action="store_true", help="train + validate only (do not touch the test window)")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    try:
        run(a.dataset, cfg, a.skip_test)
    except DatasetMissingError as e:
        log.error("%s", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
