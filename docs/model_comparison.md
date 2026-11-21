# Model comparison

## Models
| Model | Signal | How scores are made | Strengths | Weaknesses |
|---|---|---|---|---|
| Popularity (global) | exp-decayed purchase counts (half-life 30d) | same for everyone | strong, cheap baseline; works for new customers | no personalisation, tiny coverage |
| Popularity (category) | popularity × customer's product-type share | per-customer re-weighting | respects taste | over-concentrates on categories already bought (hurts when seen items are excluded) |
| Content (TF-IDF) | product metadata + description | cosine(user profile, item), profile = recency-weighted sum of purchased item vectors | works for new/long-tail items | "more of the same", depends on metadata quality |
| Item-kNN CF | co-purchase patterns | cosine item–item (top-50 neighbours, heavy users down-weighted) × user history | explainable ("bought X → also Y"), good coverage | cannot score items without purchases; popularity bias |
| Matrix factorization | latent taste factors | implicit ALS (32 factors, reg 0.05, confidence 1+20·r) | captures indirect similarity, compact | cold start, hyper-parameter sensitive |
| Hybrid | all of the above + category affinity + trending | weighted sum of per-user min-max normalised scores | combines complementary strengths; cold users fall back to popularity+trend | weights tuned on one validation window |

## Protocol
Chronological split (default 21-day validation, 21-day test, counted back from the last transaction). History for a window = all transactions before it. Candidate universe = items bought at least once in that history; targets outside it are reported as `unreachable_target_share`. Previously purchased items are excluded (configurable). Customers with no history are scored by popularity for every personalised model; the hybrid uses only popularity + recency components for them.

Hybrid weights: 6 single-component corner points + equal weights + 60 Dirichlet samples are scored by NDCG@10 on (up to 1,500) validation customers; the best is kept. The hybrid's validation score is therefore optimistic — compare models on the **test** table.

## Results
All numbers are generated into `artifacts/<dataset>/test_metrics.csv` and `validation_metrics.csv`. The only run performed while building this repo used **synthetic data** (see README for the table, clearly labelled). Real H&M results are not available until you run the pipeline with the Kaggle files; add them here after you do.

## Reading the metrics
* **Recall@K / Precision@K** – coverage of what the customer bought vs. cleanliness of the list.
* **MAP@10 / NDCG@10** – reward putting hits near the top.
* **Coverage** – share of the catalog ever recommended (low = popularity trap).
* **Diversity** – 1 − mean pairwise TF-IDF cosine in a list.
* **Head vs tail recall** – can the model recover purchases of popular items (top 20%) vs long-tail items?
