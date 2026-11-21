"""Streamlit dashboard for the personalized e-commerce recommendation system.

Run:  streamlit run app.py
All recommendations are produced live by src.inference.RecommenderService (no hardcoded examples).
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_preprocessing import DatasetMissingError, load_processed
from src.inference import ALGORITHMS, ArtifactsMissingError, RecommenderService
from src.utils.common import PROJECT_ROOT, dataset_paths, load_config

st.set_page_config(page_title="StyleSense | Recommendations", page_icon="🛍️", layout="wide")

ACCENT = "#1f6f78"
CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.5rem; max-width: 1250px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.hero {{ background: linear-gradient(120deg, {ACCENT} 0%, #2a9d8f 100%); color: white; padding: 22px 28px; border-radius: 16px; margin-bottom: 18px; }}
.hero h1 {{ color: white; margin: 0; font-size: 1.9rem; }} .hero p {{ margin: 4px 0 0 0; opacity: .9; }}
.card {{ border: 1px solid rgba(128,128,128,.25); border-radius: 14px; overflow: hidden; background: rgba(128,128,128,.06); height: 100%; }}
.ph {{ height: 150px; display:flex; align-items:center; justify-content:center; font-size: 2.2rem; color: rgba(255,255,255,.9); }}
.cbody {{ padding: 10px 12px 12px 12px; }}
.ctitle {{ font-weight: 600; font-size: .92rem; line-height: 1.25; margin-bottom: 2px; }}
.cmeta {{ font-size: .76rem; opacity: .7; }}
.badge {{ display:inline-block; background:{ACCENT}; color:white; border-radius: 999px; padding: 1px 9px; font-size: .72rem; margin: 6px 4px 4px 0; }}
.why {{ font-size: .78rem; margin-top: 4px; opacity: .85; }}
.banner {{ background:#fff3cd; color:#664d03; border:1px solid #ffecb5; padding:10px 14px; border-radius:10px; margin-bottom:12px; font-size:.9rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

COLOUR_HEX = {"Black": "#222", "White": "#c9c9c9", "Dark Blue": "#1d3557", "Light Blue": "#8ecae6", "Beige": "#d8c3a5", "Grey": "#8d99ae",
              "Red": "#c1121f", "Pink": "#e5989b", "Green": "#386641", "Yellow": "#e9c46a", "Brown": "#7f5539", "Orange": "#f4a261",
              "Dark Red": "#780000", "Dark Green": "#2d6a4f", "Light Pink": "#f4acb7", "Off White": "#d9d4c7", "Blue": "#457b9d",
              "Dark Grey": "#495057", "Light Beige": "#e9ddc8", "Dark Beige": "#a68a64", "Light Grey": "#ced4da", "Light Green": "#95d5b2"}
GROUP_ICON = {"Shoes": "👟", "Accessories": "👜", "Underwear": "🩲", "Kids": "🧒", "Sport": "🏃"}


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading trained models…")
def get_service(dataset: str) -> RecommenderService:
    return RecommenderService.load(dataset)


@st.cache_data(show_spinner="Loading processed data…")
def get_processed(dataset: str):
    cfg = load_config()
    tx, art, cust, meta = load_processed(dataset, cfg)
    return tx, art, cust, meta


def available_datasets() -> list[str]:
    cfg = load_config()
    return [d for d in cfg["datasets"] if (dataset_paths(cfg, d)["artifacts"] / "bundle.joblib").exists()]


def find_image(article_id: int) -> Path | None:
    """H&M images (if the user downloaded them) live in data/raw/images/<first 3 digits>/<10-digit id>.jpg"""
    s = f"{int(article_id):010d}"
    p = PROJECT_ROOT / "data" / "raw" / "images" / s[:3] / f"{s}.jpg"
    return p if p.exists() else None


def card(r: pd.Series, badges: list[str] | None = None, why: str | None = None) -> None:
    img = find_image(r["article_id"])
    if img is not None:
        st.image(str(img), use_container_width=True)
        top = ""
    else:
        col = COLOUR_HEX.get(str(r.get("colour_group_name")), "#6c757d")
        icon = GROUP_ICON.get(str(r.get("product_group_name")), "👕")
        top = f'<div class="ph" style="background:linear-gradient(135deg,{col},#00000055)">{icon}</div>'
    b = "".join(f'<span class="badge">{html.escape(x)}</span>' for x in (badges or []))
    w = f'<div class="why">{html.escape(why)}</div>' if why else ""
    st.markdown(
        f'<div class="card">{top}<div class="cbody"><div class="ctitle">{html.escape(str(r.get("prod_name", r["article_id"])))}</div>'
        f'<div class="cmeta">{html.escape(str(r.get("product_type_name", "")))} · {html.escape(str(r.get("colour_group_name", "")))}</div>'
        f'<div class="cmeta">{html.escape(str(r.get("department_name", "")))}</div>{b}{w}</div></div>', unsafe_allow_html=True)


def grid(df: pd.DataFrame, per_row: int = 5, badge_fn=None, why_col: str | None = "explanation") -> None:
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for c, (_, r) in zip(cols, df.iloc[start:start + per_row].iterrows()):
            with c:
                card(r, badge_fn(r) if badge_fn else None, r.get(why_col) if why_col else None)
        st.write("")


def hero(title: str, sub: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>', unsafe_allow_html=True)


def empty(msg: str) -> None:
    st.info(msg, icon="ℹ️")


# -------------------------------------------------------------------- pages
def page_overview(svc, dataset):
    hero("Overview", "Customers, catalog and purchase behaviour at a glance")
    try:
        tx, art, cust, meta = get_processed(dataset)
    except DatasetMissingError as e:
        return empty(str(e))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{cust['customer_id'].nunique():,}")
    c2.metric("Products (catalog)", f"{len(art):,}")
    c3.metric("Transactions", f"{len(tx):,}")
    c4.metric("Date range", f"{tx['t_dat'].min():%b %Y} – {tx['t_dat'].max():%b %Y}")
    weekly = tx.set_index("t_dat").resample("W").size().rename("transactions").reset_index()
    st.plotly_chart(px.line(weekly, x="t_dat", y="transactions", title="Weekly purchase trend", color_discrete_sequence=[ACCENT]), use_container_width=True)
    j = tx.merge(art[["article_id", "product_group_name", "product_type_name"]], on="article_id")
    a, b = st.columns(2)
    g = j["product_group_name"].value_counts().reset_index()
    a.plotly_chart(px.bar(g, x="count", y="product_group_name", orientation="h", title="Popular product groups", color_discrete_sequence=[ACCENT]).update_yaxes(categoryorder="total ascending"), use_container_width=True)
    top_types = j["product_type_name"].value_counts().head(8).index
    m = j[j["product_type_name"].isin(top_types)].assign(month=lambda d: d["t_dat"].dt.month)
    piv = m.pivot_table(index="product_type_name", columns="month", values="article_id", aggfunc="size", fill_value=0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    b.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="Teal", title="Seasonality: monthly share by product type"), use_container_width=True)
    with st.expander("Data quality report"):
        st.json(meta["quality_report"])


def page_recs(svc, dataset):
    hero("Personalized recommendations", "Pick a customer and an algorithm — every item below is scored live by the trained model")
    ctx = svc.ctx
    activity = np.asarray(ctx.counts.sum(axis=1)).ravel()
    order = np.argsort(-activity)
    with_hist = [ctx.user_ids[i] for i in order if activity[i] > 0][:3000]
    if not with_hist:
        return empty("No customers with purchase history in the trained model.")
    c1, c2, c3 = st.columns([3, 1.4, 1])
    cid = c1.selectbox("Customer (sorted by activity; showing the 3,000 most active)", with_hist, format_func=lambda x: f"{x[:10]}…")
    algo = c2.selectbox("Algorithm", list(ALGORITHMS), format_func=lambda k: ALGORITHMS[k])
    n = c3.selectbox("Top-N", [5, 10, 20], index=1)
    excl = st.toggle("Hide items the customer already bought", value=True)
    with st.spinner("Scoring candidates…"):
        hist = svc.history(cid, 10)
        recs = svc.recommend(cid, n, algo, excl)
    st.subheader("Recent purchases")
    if hist.empty:
        empty("This customer has no purchases before the prediction date.")
    else:
        grid(hist.head(5), 5, lambda r: [f"{r['days_ago']}d ago"], None)
    st.subheader(f"Top {n} · {ALGORITHMS[algo]}")
    if recs.empty:
        return empty("No eligible items to recommend.")
    grid(recs, 5, lambda r: [f"#{r['rank']}", f"score {r['score_normalised']:.2f}"])
    with st.expander("Scores table"):
        st.dataframe(recs[["rank", "article_id", "prod_name", "product_type_name", "colour_group_name", "score", "score_normalised", "explanation"]], use_container_width=True, hide_index=True)
    if algo == "hybrid":
        w = pd.Series(svc.hybrid.weights).rename("weight").reset_index().rename(columns={"index": "component"})
        st.plotly_chart(px.bar(w, x="component", y="weight", title="Hybrid weights (tuned on validation only)", color_discrete_sequence=[ACCENT]), use_container_width=True)


def page_similar(svc, dataset):
    hero("Similar products", "Item-to-item neighbours from content, collaborative and latent-factor similarity")
    q = st.text_input("Search by name, type, colour or article id", placeholder="e.g. dress, black, 100000001")
    items = svc.search_items(q) if q else svc.items.assign(item_idx=np.arange(len(svc.items))).sort_values("article_id").head(0)
    if q and items.empty:
        return empty("No products match your search.")
    if not q:
        pop_idx = np.argsort(-svc.ctx.item_pop_total)[:30]
        items = pd.DataFrame([svc.item_row(int(i)) for i in pop_idx])
        st.caption("No search yet — showing the 30 most popular products.")
    choice = st.selectbox("Product", items["item_idx"].tolist(), format_func=lambda i: f"{svc.items.iloc[i]['prod_name']} · {svc.items.iloc[i]['product_type_name']} ({svc.items.iloc[i]['article_id']})")
    c1, c2 = st.columns(2)
    method = c1.selectbox("Similarity", ["hybrid", "content", "collaborative", "matrix_factorization"])
    n = c2.slider("How many", 5, 20, 10)
    st.subheader("Selected product")
    _, mid, _ = st.columns([1, 1, 3])
    with mid:
        card(pd.Series(svc.item_row(int(choice))))
    sim = svc.similar_items(int(choice), n, method)
    st.subheader("Most similar")
    grid(sim, 5, lambda r: [f"sim {r['similarity']:.2f}"], None)


def page_segments(svc, dataset):
    hero("Customer segmentation", "RFM + K-Means; segment names are derived from measured cluster profiles")
    d = svc.dir / "segmentation"
    if not (d / "segments.parquet").exists():
        return empty("Segmentation artifacts not found. Run `python -m src.train`.")
    seg = pd.read_parquet(d / "segments.parquet")
    summ = pd.read_csv(d / "segment_summary.csv")
    cat = pd.read_csv(d / "segment_category_preferences.csv", index_col=0)
    meta = svc.read_json("segmentation/segmentation_meta.json") or {}
    st.caption(f"k = {meta.get('chosen_k')} chosen by silhouette · silhouette scores by k: " + ", ".join(f"{k}: {v:.3f}" for k, v in meta.get("silhouette_by_k", {}).items()) + f" · as of {meta.get('as_of')}")
    a, b = st.columns([1, 1.4])
    cnt = seg["segment"].value_counts().reset_index()
    a.plotly_chart(px.pie(cnt, names="segment", values="count", hole=.5, title="Segment distribution"), use_container_width=True)
    samp = seg.sample(min(len(seg), 5000), random_state=0)
    ev = meta.get("pca_explained_variance", [0, 0])
    b.plotly_chart(px.scatter(samp, x="pca1", y="pca2", color="segment", opacity=.6, title="PCA of standardised RFM", labels={"pca1": f"PC1 ({ev[0]:.0%})", "pca2": f"PC2 ({ev[1]:.0%})"}), use_container_width=True)
    st.subheader("RFM profile per segment")
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.plotly_chart(px.imshow(cat, aspect="auto", color_continuous_scale="Teal", title="Category preferences (share of segment purchases)"), use_container_width=True)
    st.markdown("**How segments feed recommendations:** high-value customers → hybrid/personalised ranking and premium or new-arrival slots; "
                "frequent shoppers → fresh items and cross-category suggestions (they exhaust popular items quickly); "
                "at-risk customers → win-back with categories they previously bought; occasional shoppers → popularity + category-aware picks.")


def page_perf(svc, dataset):
    hero("Model performance", "Chronological split · validation for selection · held-out test for the final comparison")
    if svc.is_synthetic:
        st.markdown('<div class="banner"><b>Synthetic data.</b> These metrics come from generated smoke-test data and are NOT real-world results.</div>', unsafe_allow_html=True)
    test = svc.read_json("test_results.json")
    if not test:
        return empty("No test results found. Run `python -m src.train` (or `python -m src.evaluate`).")
    meta = test["meta"]
    tab = pd.read_csv(svc.dir / "test_metrics.csv", index_col=0)
    st.caption(f"Test window starts {meta['test_start']} · {meta['n_eval_users']} evaluated customers ({meta['n_new_customers']} new) · {meta['candidate_items']} candidate items · seen items excluded: {meta['exclude_seen']} · MF backend: {meta['mf_backend']}")
    main_cols = ["precision@5", "precision@10", "recall@5", "recall@10", "map@10", "ndcg@10", "coverage@10", "diversity@10"]
    st.subheader("Test results")
    st.dataframe(tab[main_cols].round(4), use_container_width=True)
    long = tab[["ndcg@10", "recall@10", "map@10", "precision@10"]].reset_index().melt("model", var_name="metric", value_name="value")
    st.plotly_chart(px.bar(long, x="model", y="value", color="metric", barmode="group", title="Ranking metrics (test)"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(tab.reset_index(), x="coverage@10", y="diversity@10", text="model", title="Coverage vs diversity (test)").update_traces(textposition="top center", marker_size=12), use_container_width=True)
    seg_cols = [c for c in tab.columns if c.startswith("ndcg@10_")]
    if seg_cols:
        s = tab[seg_cols].reset_index().melt("model", var_name="segment", value_name="ndcg@10")
        s["segment"] = s["segment"].str.replace("ndcg@10_", "")
        b.plotly_chart(px.bar(s, x="model", y="ndcg@10", color="segment", barmode="group", title="New vs returning customers (NDCG@10)"), use_container_width=True)
    if {"recall@10_head", "recall@10_tail"} <= set(tab.columns):
        h = tab[["recall@10_head", "recall@10_tail"]].reset_index().melt("model", var_name="item_group", value_name="recall@10")
        st.plotly_chart(px.bar(h, x="model", y="recall@10", color="item_group", barmode="group", title="Recall@10: popular (top 20%) vs long-tail target items"), use_container_width=True)
    val = svc.dir / "validation_metrics.csv"
    with st.expander("Validation results (used for model selection / weight tuning)"):
        if val.exists():
            st.dataframe(pd.read_csv(val, index_col=0)[main_cols].round(4), use_container_width=True)
        st.caption("The hybrid's weights were tuned on these validation users, so its validation score is optimistic. Compare models on the test table.")
    with st.expander("Evaluation protocol"):
        st.json(meta)


def page_cold(svc, dataset):
    hero("Cold-start recommendations", "New here? Tell us what you like — no purchase history needed")
    opts = svc.cold_start_options()
    c1, c2, c3 = st.columns(3)
    groups = c1.multiselect("Product groups", opts["groups"])
    types = c2.multiselect("Product types", opts["types"])
    colours = c3.multiselect("Preferred colours", opts["colours"])
    n = st.slider("How many recommendations", 5, 20, 10)
    if not (groups or types):
        return empty("Select at least one product group or type to get started.")
    with st.spinner("Finding items…"):
        recs = svc.recommend_cold_start(groups, types, colours, n)
    if recs.empty:
        return empty("No products match that combination — try broadening your selection.")
    grid(recs, 5, lambda r: [f"#{r['rank']}"])


PAGES = {"Overview": page_overview, "Personalized recommendations": page_recs, "Similar products": page_similar,
         "Customer segmentation": page_segments, "Model performance": page_perf, "Cold-start": page_cold}


def main():
    st.sidebar.markdown("## 🛍️ StyleSense")
    ds = available_datasets()
    if not ds:
        hero("Setup needed", "No trained artifacts were found")
        st.error("Train a model first (PowerShell):")
        st.code("python scripts/make_demo_data.py\npython -m src.data_preprocessing --dataset demo\npython -m src.train --dataset demo", language="powershell")
        st.stop()
    dataset = st.sidebar.selectbox("Dataset", ds, index=ds.index("real") if "real" in ds else 0)
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    try:
        svc = get_service(dataset)
    except ArtifactsMissingError as e:
        st.error(str(e)); st.stop()
    if svc.is_synthetic:
        st.markdown('<div class="banner">⚠️ Running on <b>synthetic demo data</b> (smoke test). Train on the real H&M data for real results — see README.</div>', unsafe_allow_html=True)
    st.sidebar.caption(f"Models fitted on history up to {pd.Timestamp(svc.meta['history_end']):%Y-%m-%d} · {svc.ctx.n_items:,} candidate items")
    PAGES[page](svc, dataset)


main()
