# Architecture

## Layers (training / evaluation / inference / UI are separate)
| Layer | Module | Responsibility |
|---|---|---|
| Data | `src/data_preprocessing.py` | chunked + sampled loading, schema validation, cleaning, EDA plots, parquet export, metadata |
| Features | `src/feature_engineering.py` | chronological split; `Context` (as-of-date sparse matrices, affinities, popularity); customer/product/interaction feature tables |
| Models | `src/models/*` | common interface `score_batch(user_idx) -> (b, n_items)` |
| Training | `src/train.py` | fit → validate → tune hybrid → refit → persist (`artifacts/<ds>/bundle.joblib`) |
| Evaluation | `src/evaluate.py`, `src/utils/metrics.py` | ranking metrics, segment breakdowns, charts, final test run |
| Segmentation | `src/segmentation.py` | RFM, StandardScaler, K-Means (silhouette), PCA, data-driven labels |
| Inference | `src/inference.py` | `RecommenderService`: recommend, explain, similar items, cold start |
| UI | `app.py` | Streamlit, calls only `RecommenderService` and saved reports |

## Key design decisions
* **Shared index space.** All models use the same user/item indices from `Context`, so scores can be compared, normalised and blended, and seen-item masking is applied uniformly by the caller.
* **Leakage guard.** `build_context(tx_hist, ..., now)` raises if any transaction is on/after `now`; all ages/recency are relative to `now`.
* **Sparse everywhere.** User–item interactions are CSR matrices; item–item similarity is computed in chunks and pruned to top-k neighbours, never densified.
* **Two-stage fitting.** Models are fit on train for selection, then refit on train+val (same hyper-parameters / hybrid weights) so the test prediction date is the end of validation.
* **Graceful degradation.** Missing raw data → explanatory error; missing artifacts → app shows setup instructions; `implicit` unavailable → documented numpy ALS fallback (backend recorded).
* **Artifacts are git-ignored**; only code, config, docs and small chart images are versioned.

## Artifacts per dataset (`artifacts/<dataset>/`)
`bundle.joblib` (context + fitted models + meta), `meta.json`, `validation_metrics.csv`, `validation_results.json` (incl. all hybrid tuning trials), `test_metrics.csv`, `test_results.json`, `charts/`, `customer_features.parquet`, `product_features.parquet`, `segmentation/*`.
