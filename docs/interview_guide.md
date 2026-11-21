# Interview guide

## 60-second pitch
"I built an end-to-end recommender for fashion e-commerce on the H&M transactions schema. It has five models — popularity, TF-IDF content-based, item-based collaborative filtering on sparse matrices, implicit-feedback ALS matrix factorization, and a hybrid that blends normalised scores with customer-category affinity and a trending signal. I evaluate with a chronological train/validation/test split so no future information leaks into features, tune the hybrid weights on validation only, and report Precision/Recall@K, MAP, NDCG, coverage and diversity, plus new-vs-returning and popular-vs-long-tail breakdowns. I also added RFM segmentation with K-Means and a Streamlit app with explanations, similar-products and cold-start flows. I validated the pipeline end to end on a synthetic dataset and it's designed to run on a sampled version of the real Kaggle data on a laptop." *(Say clearly which dataset your quoted numbers come from.)*

## Concepts
**What is a recommender system?** Software that ranks items per user to maximise a goal (clicks, purchases, revenue). Typical stack: candidate generation → ranking → re-ranking/business rules.

**Collaborative vs content-based.** CF uses behaviour (who bought what) — finds patterns without knowing item attributes, but cannot handle items/users with no interactions. Content-based uses item attributes (here TF-IDF on type/colour/department/description) — handles new items and explains itself, but gives "more of the same" and needs good metadata.

**Why sparse matrices?** A customer×item matrix is >99% empty (demo: density 1.4%; real H&M far lower). Dense storage is O(users×items) memory; CSR stores only non-zeros, and sparse products (`X.T @ X`) skip zeros. We also prune each item's neighbours to top-k so the similarity matrix stays sparse.

**How does matrix factorization work?** Approximate the interaction matrix R ≈ U·Vᵀ with k latent factors per user/item. Implicit ALS (Hu, Koren, Volinsky) treats every observed purchase as preference p=1 with confidence c=1+α·r and unobserved as p=0 with low confidence, then alternates solving closed-form ridge regressions for U (item factors fixed) and V (user factors fixed). Score = u·v.

**Cold start.** New users (no history), new items (no interactions), new system. Mitigations: popularity/trending fallback, ask for preferences (our cold-start page: popularity + content similarity within chosen categories/colours), content-based for new items, contextual features (age, channel), exploration.

**Why chronological splitting?** Production predicts the future from the past. A random split lets the model train on purchases that happen after test purchases (leakage), inflating metrics and misrepresenting seasonality/trends. We build all matrices as-of the prediction date and the code raises if later data slips in.

**Negative feedback in implicit data.** There are no explicit dislikes: a missing purchase may mean "not interested" or "never saw it". ALS treats unobserved as weak negatives (low confidence) and observed as positive with higher confidence; ranking metrics treat non-purchases as irrelevant only for evaluation. Better approaches: exposure logs, negative sampling (BPR), or learning-to-rank with impressions.

**Why isn't accuracy enough?** The task is ranking with a huge negative class: predicting "not bought" for everything is ~99.99% accurate. We need top-K ranking metrics (precision/recall/MAP/NDCG) plus beyond-accuracy metrics (coverage, diversity, long-tail performance).

**Precision@K vs Recall@K.** Precision@K = hits/K (how clean is the list). Recall@K = hits/|relevant| (how much of what they bought did we surface). With few purchases per window, recall is more interpretable; precision is capped at |relevant|/K.

**How does hybrid ranking work?** Each component produces a per-user score vector; each is min-max normalised to [0,1] over eligible candidates so scales are comparable; final = Σ wᵢ·scoreᵢ. Weights are chosen by random search on validation NDCG@10. Cold users use only history-free components (popularity, trend). Contributions per component make explanations possible.

**How to scale?** Sample/stream data (done: chunked reads, hash sampling); candidate generation via ANN (FAISS/ScaNN) on item embeddings; precompute item–item neighbours offline; batch-score users; feature store + two-stage retrieval/ranking (LightGBM or neural ranker); incremental retraining; cache top-N per user in Redis; distributed ALS (Spark) for hundreds of millions of interactions.

## 20+ likely technical questions
1. **Why item-based rather than user-based CF?** Items are far fewer and more stable than users, so item similarities can be precomputed and pruned; user vectors change with every purchase.
2. **Why cosine similarity?** It normalises for popularity/vector magnitude and is cheap on sparse vectors; we also down-weight heavy users (1/√activity).
3. **What is recency weighting and why?** Interaction weight = 2^(−age/half_life) (60 days default). Fashion tastes and stock turn over fast; older purchases are weaker signals.
4. **How did you choose the candidate universe?** Items purchased at least once before the prediction date; targets outside it are reported as unreachable rather than hidden.
5. **Why exclude already-purchased items?** We measure discovery of new items; H&M has repeat purchases, so a repeat-aware variant is available via `evaluation.exclude_seen: false`.
6. **How are new customers evaluated?** They have no history so every personalised model falls back to popularity; reported separately (with n, as the sample can be tiny).
7. **How did you prevent leakage in hybrid tuning?** Weights are tuned on the validation window only; the test window is evaluated once after refit on train+val. Validation score of the hybrid is acknowledged as in-sample.
8. **Why refit on train+validation?** The model for test should use the most recent history; hyper-parameters/weights are fixed from validation.
9. **Explain MAP@10 and NDCG@10.** MAP: mean over users of average precision (precision at each hit rank) normalised by min(|rel|,10). NDCG: DCG with 1/log2(rank+1) discount divided by ideal DCG; both reward early hits.
10. **What does coverage tell you?** Share of the catalog recommended to anyone; popularity scored ~1.6% on the synthetic demo vs 43% for the hybrid.
11. **How is diversity computed?** 1 − mean pairwise cosine of TF-IDF vectors within each top-10 list.
12. **Why does the content model do poorly on the synthetic data?** The generator's latent taste isn't expressed in metadata, so metadata similarity carries little signal. Don't generalise this to real data; test on H&M.
13. **Which ALS hyper-parameters matter?** factors, regularisation, confidence α, iterations; tune on validation (config exposes all).
14. **What if `implicit` doesn't install on Windows?** The code falls back to a numpy ALS that solves the same normal equations per row; it's slower but equivalent in objective; the backend is logged and saved.
15. **How do you normalise scores in the hybrid and why not z-score?** Per-user min-max over eligible items; scores are non-Gaussian/sparse (CF scores are mostly zero), so z-scores would be dominated by outliers.
16. **How do you explain a recommendation?** CF: nearest purchased item by item similarity; content: most similar purchased item (cosine); hybrid: share of each component in the final score; cold start: stated category/colour + popularity.
17. **How would you add images/embeddings?** Add CLIP/sentence-transformer item vectors as another content component or as ANN retrieval source; the component interface is just `score_batch`.
18. **How would you A/B test it?** Randomise users, compare CTR/conversion/revenue per user with guardrails (latency, diversity), sample-size/power calculation, sequential testing.
19. **What biases exist in offline evaluation?** Exposure/position bias (users only buy what they were shown), popularity bias, seasonality in the test window, survivorship (only purchasers evaluated).
20. **How does RFM segmentation connect to recommendations?** Segments choose strategy: high-value → premium/new arrivals, frequent → fresh/cross-category, at-risk → win-back in previously bought categories, occasional → popularity + category-aware. Also useful for cold-ish users via segment-level popularity.
21. **How did you pick k for K-Means and name clusters?** Silhouette over k∈[3,6]; names assigned from measured centroid R/F/M after fitting, not assumed.
22. **Why log-transform frequency/monetary before scaling?** They are heavy-tailed; K-Means (Euclidean) would otherwise be dominated by a few large customers.
23. **Time/space complexity of item-kNN?** Co-occurrence computation ~ Σ over users of (items per user)², chunked and pruned; storage O(items×k).
24. **How do you handle repeat purchases?** Counts feed the recency-weighted matrix (stronger signal); `exclude_seen` controls whether repeats are eligible; repeat share is reported in EDA.
25. **What would you do next?** Learning-to-rank over merged candidates, two-tower retrieval, ANN, session/sequence models, online learning, business constraints.

## Resume bullets (fill the bracketed values only with numbers you actually measured)
* Built an end-to-end fashion recommendation pipeline (popularity, TF-IDF content-based, sparse item-kNN collaborative filtering, implicit ALS matrix factorization and a validation-tuned hybrid ranker) on the H&M transactions schema, with chunked/sampled loading for 8–16 GB laptops.
* Designed a leakage-safe chronological train/validation/test evaluation (Precision/Recall@K, MAP@10, NDCG@10, coverage, diversity; new vs returning and popular vs long-tail breakdowns) and RFM + K-Means customer segmentation; [add your measured H&M test NDCG@10/Recall@10 vs the popularity baseline once you run it].
* Shipped a six-page Streamlit app serving live recommendations with explanations, similar-product search and cold-start flows, backed by a modular codebase with 34 automated tests.

*(Until you train on the real H&M data, don't claim performance numbers; the repo's only results so far are on synthetic data.)*
