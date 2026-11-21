import numpy as np
import pytest

from src.evaluate import build_eval_set, evaluate_model, popular_item_set
from src.feature_engineering import build_context
from src.inference import ALGORITHMS
from src.models.base import minmax_rows, top_n
from src.models.matrix_factorization import als_numpy


def _warm_user(ctx):
    return ctx.user_ids[int(np.argmax(np.asarray(ctx.counts.sum(axis=1)).ravel()))]


def test_context_rejects_future_data(tables, split):
    art, cus, _, _ = tables
    with pytest.raises(ValueError):
        build_context(split["test"], art, cus, split["test_start"])   # contains rows on/after `now`


@pytest.mark.parametrize("algo", list(ALGORITHMS))
def test_recommend_output_contract(service, ctx, algo):
    cid = _warm_user(ctx)
    n = 7
    r = service.recommend(cid, n, algo, exclude_seen=True)
    assert len(r) == n
    assert r["article_id"].is_unique
    assert list(r["rank"]) == list(range(1, n + 1))
    assert (r["score"].diff().dropna() <= 1e-9).all(), "scores must be sorted descending"
    assert r["explanation"].str.len().gt(0).all()
    bought = set(ctx.item_ids[ctx.counts[ctx.user_index[cid]].nonzero()[1]])
    assert not (set(r["article_id"]) & bought), "previously purchased items must be excluded"


def test_new_customer_gets_popularity_fallback(service, ctx):
    cold = np.where(~ctx.has_history)[0]
    if len(cold) == 0:
        pytest.skip("no cold customers in this sample")
    cid = ctx.user_ids[cold[0]]
    for algo in ("hybrid", "content", "matrix_factorization"):
        r = service.recommend(cid, 5, algo)
        assert len(r) == 5 and "New customer" in r.iloc[0]["explanation"]


def test_unknown_customer_raises(service):
    with pytest.raises(KeyError):
        service.recommend("does-not-exist")


def test_cold_start_respects_preferences(service):
    opts = service.cold_start_options()
    group = opts["groups"][0]
    r = service.recommend_cold_start(groups=[group], n=8)
    assert 0 < len(r) <= 8
    assert (r["product_group_name"] == group).all()
    assert r["score"].is_monotonic_decreasing
    assert len(service.recommend_cold_start(groups=[group], colours=[opts["colours"][0]], n=3)) > 0


def test_cold_start_no_match_is_empty(service):
    assert service.recommend_cold_start(groups=["not-a-group"]).empty


def test_similar_items(service):
    for method in ("content", "collaborative", "matrix_factorization", "hybrid"):
        s = service.similar_items(0, 5, method)
        assert len(s) == 5 and 0 not in set(s["item_idx"])


def test_minmax_and_topn_masking():
    s = np.array([[1.0, 5.0, 3.0, 9.0]])
    m = np.array([[False, False, False, True]])
    out = minmax_rows(s, m)
    assert out.max() == 1.0 and out[0, 3] == 0.0 and out[0, 0] == 0.0
    assert top_n(s[0], 2, m[0]).tolist() == [1, 2]


def test_hybrid_cold_users_use_only_history_free_components(models, ctx):
    h = models["hybrid"]
    cold = np.where(~ctx.has_history)[0]
    if len(cold) == 0:
        pytest.skip("no cold customers in this sample")
    u = np.array([cold[0]])
    comps = h.component_scores(u)
    out = h.combine(comps, u, {c: float(c == "collaborative") for c in h.components})
    assert np.allclose(out, 0.5 * comps["popularity"] + 0.5 * comps["recency"])


def test_numpy_als_learns_signal():
    from scipy import sparse
    rng = np.random.default_rng(0)
    R = np.zeros((20, 10), dtype=np.float32)
    R[:10, :5] = rng.random((10, 5)) > 0.3
    R[10:, 5:] = rng.random((10, 5)) > 0.3
    X, Y = als_numpy(sparse.csr_matrix(R), 4, 0.05, 8, 10.0, 0)
    scores = X @ Y.T
    assert scores[:10, :5].mean() > scores[:10, 5:].mean() + 0.2


def test_evaluation_runs_and_is_bounded(ctx, models, split):
    es = build_eval_set(ctx, split["test"], 100, 0)
    assert len(es.user_idx) > 0
    res = evaluate_model(models["popularity"], ctx, es, (5, 10), 20, True, popular_item_set(ctx, 0.8), models["content"].item_vecs)
    for k in ("precision@5", "recall@10", "map@10", "ndcg@10", "coverage@10", "diversity@10"):
        assert 0.0 <= res[k] <= 1.0
    assert set(res["by_customer_type"]) == {"new_customers", "returning_customers"}
