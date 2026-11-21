"""Dataset loading, validation, cleaning, EDA and processed-data export.

CLI:
    python -m src.data_preprocessing --dataset demo
    python -m src.data_preprocessing --dataset real     # needs Kaggle files in data/raw/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import dataset_paths, load_config, set_seed, setup_logging

log = logging.getLogger("preprocess")

REQUIRED = {
    "articles.csv": ["article_id", "product_type_name", "product_group_name", "colour_group_name", "department_name"],
    "customers.csv": ["customer_id"],
    "transactions_train.csv": ["t_dat", "customer_id", "article_id", "price"],
}
KAGGLE_HELP = (
    "H&M dataset files not found in {raw}.\n"
    "To use the real data:\n"
    "  1. Create a Kaggle account and accept the competition rules at\n"
    "     https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules\n"
    "  2. Create an API token (Kaggle > Settings > API) and save it as %USERPROFILE%\\.kaggle\\kaggle.json\n"
    "  3. pip install kaggle ; kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data\\raw\n"
    "  4. Unzip articles.csv, customers.csv, transactions_train.csv into data\\raw\\\n"
    "Or run on the synthetic smoke-test data:  python scripts/make_demo_data.py  &&  --dataset demo"
)


class DatasetMissingError(FileNotFoundError):
    pass


def inspect_raw_dir(raw_dir: Path) -> dict[str, dict]:
    """Report which expected files exist (never assumes)."""
    info = {}
    for name in REQUIRED:
        p = raw_dir / name
        info[name] = {"exists": p.exists(), "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None}
    return info


def _stable_hash_mod(series: pd.Series, mod: int) -> np.ndarray:
    return (pd.util.hash_pandas_object(series, index=False).values % np.uint64(mod)).astype(np.int64)


def load_raw(raw_dir: Path, sampling: dict, is_synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load articles/customers/transactions with chunked, sampled reading of transactions."""
    info = inspect_raw_dir(raw_dir)
    missing = [k for k, v in info.items() if not v["exists"]]
    if missing:
        raise DatasetMissingError(f"Missing {missing}.\n" + KAGGLE_HELP.format(raw=raw_dir))
    log.info("Raw files: %s", info)

    articles = pd.read_csv(raw_dir / "articles.csv")
    customers = pd.read_csv(raw_dir / "customers.csv")

    frac = float(sampling.get("customer_fraction", 1.0))
    buckets = 10_000
    keep_cut = int(round(frac * buckets))
    # Determine window start without loading everything: scan only the date column.
    history_days = sampling.get("history_days")
    max_date = None
    for ch in pd.read_csv(raw_dir / "transactions_train.csv", usecols=["t_dat"], chunksize=int(sampling.get("chunksize", 1_000_000))):
        m = pd.to_datetime(ch["t_dat"]).max()
        max_date = m if max_date is None or m > max_date else max_date
    min_date = max_date - pd.Timedelta(days=int(history_days)) if history_days else None
    log.info("Transactions end on %s; keeping from %s; customer fraction=%.3f", max_date.date(), min_date.date() if min_date is not None else "start", frac)

    parts = []
    dtype = {"article_id": "int64", "price": "float32", "sales_channel_id": "int8"}
    for ch in pd.read_csv(raw_dir / "transactions_train.csv", chunksize=int(sampling.get("chunksize", 1_000_000)), dtype=dtype):
        ch["t_dat"] = pd.to_datetime(ch["t_dat"])
        if min_date is not None:
            ch = ch[ch["t_dat"] >= min_date]
        if frac < 1.0:
            ch = ch[_stable_hash_mod(ch["customer_id"], buckets) < keep_cut]
        parts.append(ch)
    tx = pd.concat(parts, ignore_index=True)
    log.info("Loaded %d transactions after sampling", len(tx))
    return articles, customers, tx


# ---------------------------------------------------------------- validation
def validate_and_clean(articles: pd.DataFrame, customers: pd.DataFrame, tx: pd.DataFrame, min_customer_tx: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Validate schemas, drop invalid rows, report quality issues."""
    for name, df in [("articles.csv", articles), ("customers.csv", customers), ("transactions_train.csv", tx)]:
        missing = [c for c in REQUIRED[name] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    report: dict = {}
    report["missing_values"] = {
        "articles": articles.isna().sum()[lambda s: s > 0].to_dict(),
        "customers": customers.isna().sum()[lambda s: s > 0].to_dict(),
        "transactions": tx.isna().sum()[lambda s: s > 0].to_dict(),
    }
    report["duplicates"] = {
        "articles_duplicate_ids": int(articles["article_id"].duplicated().sum()),
        "customers_duplicate_ids": int(customers["customer_id"].duplicated().sum()),
        "transactions_fully_duplicated_rows": int(tx.duplicated().sum()),
    }
    tx = tx.copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"], errors="coerce")
    n0 = len(tx)
    tx = tx.dropna(subset=["t_dat", "customer_id", "article_id", "price"])
    articles = articles.drop_duplicates("article_id")
    customers = customers.drop_duplicates("customer_id")
    # Note: identical (customer, article, date, price) rows are legitimate multi-unit purchases in H&M -> kept.
    bad_price = int((tx["price"] <= 0).sum())
    tx = tx[tx["price"] > 0]
    orphan_art = int((~tx["article_id"].isin(articles["article_id"])).sum())
    orphan_cus = int((~tx["customer_id"].isin(customers["customer_id"])).sum())
    tx = tx[tx["article_id"].isin(articles["article_id"]) & tx["customer_id"].isin(customers["customer_id"])]
    if min_customer_tx > 1:
        counts = tx["customer_id"].value_counts()
        tx = tx[tx["customer_id"].isin(counts[counts >= min_customer_tx].index)]
    report["dropped"] = {"invalid_rows": n0 - len(tx), "non_positive_price": bad_price,
                         "orphan_article_rows": orphan_art, "orphan_customer_rows": orphan_cus}
    if len(tx) == 0:
        raise ValueError("No valid transactions left after cleaning.")
    # restrict reference tables to what is actually used
    customers = customers[customers["customer_id"].isin(tx["customer_id"].unique())].reset_index(drop=True)
    articles = articles.reset_index(drop=True)
    report["n_customers"] = int(customers["customer_id"].nunique())
    report["n_articles_catalog"] = int(len(articles))
    report["n_articles_purchased"] = int(tx["article_id"].nunique())
    report["n_transactions"] = int(len(tx))
    report["date_min"] = str(tx["t_dat"].min().date())
    report["date_max"] = str(tx["t_dat"].max().date())
    return articles, customers, tx.reset_index(drop=True), report


def build_interaction_matrix(tx: pd.DataFrame):
    """Customer x item sparse purchase-count matrix (scipy CSR) + id lists."""
    from scipy import sparse
    users = pd.Index(tx["customer_id"].unique())
    items = pd.Index(tx["article_id"].unique())
    r = users.get_indexer(tx["customer_id"])
    c = items.get_indexer(tx["article_id"])
    m = sparse.coo_matrix((np.ones(len(tx), dtype=np.float32), (r, c)), shape=(len(users), len(items))).tocsr()
    return m, users, items


# ---------------------------------------------------------------------- EDA
def eda_statistics(articles: pd.DataFrame, customers: pd.DataFrame, tx: pd.DataFrame) -> dict:
    per_cust = tx.groupby("customer_id").size()
    per_item = tx.groupby("article_id").size()
    pair = tx.groupby(["customer_id", "article_id"]).size()
    stats = {
        "transactions_per_customer": per_cust.describe().round(2).to_dict(),
        "transactions_per_item": per_item.describe().round(2).to_dict(),
        "share_of_customers_with_repeat_purchase": float((pair.groupby(level=0).max() > 1).mean()),
        "share_of_customer_item_pairs_repeated": float((pair > 1).mean()),
        "top10_items_share_of_transactions": float(per_item.nlargest(10).sum() / len(tx)),
        "top_20pct_items_share_of_transactions": float(per_item.nlargest(max(1, int(0.2 * len(per_item)))).sum() / len(tx)),
        "matrix_sparsity": float(1 - len(pair) / (per_cust.shape[0] * per_item.shape[0])),
        "avg_price": float(tx["price"].mean()),
    }
    return stats


def make_eda_plots(articles: pd.DataFrame, customers: pd.DataFrame, tx: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="mako")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    df = tx.merge(articles[["article_id", "product_group_name", "product_type_name"]], on="article_id", how="left")

    def save(fig, name):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        saved.append(p)

    # 1 category distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["product_group_name"].value_counts().sort_values().plot.barh(ax=ax, color="#1f6f78")
    ax.set_title("Transactions by product group"); ax.set_xlabel("transactions")
    save(fig, "category_distribution.png")

    # 2 purchases per customer
    per_cust = tx.groupby("customer_id").size()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(per_cust.clip(upper=per_cust.quantile(0.99)), bins=40, ax=ax, color="#1f6f78")
    ax.set_title("Purchases per customer (clipped at 99th pct)"); ax.set_xlabel("purchases")
    save(fig, "customer_activity.png")

    # 3 popularity long tail
    pop = tx["article_id"].value_counts().values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(pop) + 1), pop, color="#c4452d")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Item popularity (long tail)"); ax.set_xlabel("item rank"); ax.set_ylabel("transactions")
    save(fig, "popularity_long_tail.png")

    # 4 weekly trend + monthly seasonality
    weekly = tx.set_index("t_dat").resample("W").size()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    weekly.plot(ax=ax, color="#1f6f78"); ax.set_title("Weekly transactions"); ax.set_ylabel("transactions")
    save(fig, "weekly_trend.png")

    df["month"] = df["t_dat"].dt.month
    top_types = df["product_type_name"].value_counts().head(8).index
    piv = df[df["product_type_name"].isin(top_types)].pivot_table(index="product_type_name", columns="month", values="article_id", aggfunc="size", fill_value=0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(piv, cmap="mako_r", ax=ax, cbar_kws={"label": "share of the type's purchases"})
    ax.set_title("Seasonality: monthly share of purchases for top product types"); ax.set_ylabel("")
    save(fig, "seasonality_heatmap.png")

    # 5 repeat purchases
    pair = tx.groupby(["customer_id", "article_id"]).size()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pair.clip(upper=6).value_counts().sort_index().plot.bar(ax=ax, color="#1f6f78")
    ax.set_title("Times a customer bought the same item (6 = 6+)"); ax.set_xlabel("repeat count"); ax.set_ylabel("customer-item pairs"); ax.set_yscale("log")
    save(fig, "repeat_purchases.png")

    # 6 price distribution & age
    if "age" in customers:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(customers["age"].dropna(), bins=30, ax=ax, color="#1f6f78"); ax.set_title("Customer age")
        save(fig, "customer_age.png")
    return saved


# ------------------------------------------------------------------- driver
def run(dataset: str, cfg: dict, make_plots: bool = True) -> dict:
    paths = dataset_paths(cfg, dataset)
    articles, customers, tx = load_raw(paths["raw"], cfg["sampling"] if not paths["is_synthetic"] else {"customer_fraction": 1.0}, paths["is_synthetic"])
    articles, customers, tx, report = validate_and_clean(articles, customers, tx, cfg["sampling"].get("min_customer_transactions", 1))
    stats = eda_statistics(articles, customers, tx)
    out = paths["processed"]
    out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(out / "transactions.parquet", index=False)
    articles.to_parquet(out / "articles.parquet", index=False)
    customers.to_parquet(out / "customers.parquet", index=False)
    meta = {"dataset": dataset, "is_synthetic": paths["is_synthetic"], "quality_report": report, "eda_statistics": stats,
            "sampling": None if paths["is_synthetic"] else cfg["sampling"], "seed": cfg["seed"]}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if make_plots:
        plots = make_eda_plots(articles, customers, tx, out / "eda")
        log.info("Saved %d EDA plots to %s", len(plots), out / "eda")
    log.info("Processed data saved to %s", out)
    return meta


def load_processed(dataset: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    out = dataset_paths(cfg, dataset)["processed"]
    needed = [out / f for f in ("transactions.parquet", "articles.parquet", "customers.parquet", "metadata.json")]
    if not all(p.exists() for p in needed):
        raise DatasetMissingError(f"Processed data for '{dataset}' not found in {out}. Run: python -m src.data_preprocessing --dataset {dataset}")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return pd.read_parquet(out / "transactions.parquet"), pd.read_parquet(out / "articles.parquet"), pd.read_parquet(out / "customers.parquet"), meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["demo", "real"], default="demo")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    try:
        meta = run(a.dataset, cfg, make_plots=not a.no_plots)
    except DatasetMissingError as e:
        log.error("%s", e)
        raise SystemExit(2)
    print(json.dumps(meta["quality_report"], indent=2, default=str))


if __name__ == "__main__":
    main()
