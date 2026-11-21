"""Synthetic H&M-shaped dataset generator.

!!! FOR SMOKE TESTING / DEMOS ONLY !!!
Metrics computed on this data are NOT real-world results. The generator plants a
latent "style cluster" structure plus seasonality so that the pipeline has some
signal to learn, which makes the numbers optimistic and unrepresentative.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TAXONOMY = {
    # product_group: (index_group, [product types], seasonal tag per type)
    "Garment Upper body": ("Ladieswear", ["T-shirt", "Blouse", "Sweater", "Shirt", "Top", "Cardigan"]),
    "Garment Lower body": ("Ladieswear", ["Trousers", "Jeans", "Skirt", "Shorts", "Leggings"]),
    "Garment Full body": ("Ladieswear", ["Dress", "Jumpsuit", "Coat", "Jacket"]),
    "Accessories": ("Divided", ["Bag", "Scarf", "Hat", "Belt", "Sunglasses"]),
    "Shoes": ("Divided", ["Sneakers", "Boots", "Sandals", "Heels"]),
    "Underwear": ("Lingeries", ["Bra", "Briefs", "Socks"]),
    "Menswear Basics": ("Menswear", ["Polo shirt", "Chinos", "Hoodie", "Blazer"]),
    "Kids": ("Baby/Children", ["Kids T-shirt", "Kids Dress", "Kids Pyjamas", "Kids Jacket"]),
    "Sport": ("Sport", ["Sports bra", "Running tights", "Track jacket", "Sports top"]),
}
WINTER = {"Coat", "Jacket", "Sweater", "Cardigan", "Scarf", "Boots", "Hoodie", "Kids Jacket", "Track jacket", "Leggings"}
SUMMER = {"Shorts", "Sandals", "Sunglasses", "T-shirt", "Top", "Kids T-shirt", "Dress", "Kids Dress"}
COLOURS = ["Black", "White", "Dark Blue", "Light Blue", "Beige", "Grey", "Red", "Pink", "Green", "Yellow", "Brown", "Orange"]
COLOUR_MASTER = {"Black": "Black", "White": "White", "Dark Blue": "Blue", "Light Blue": "Blue", "Beige": "Beige", "Grey": "Grey",
                 "Red": "Red", "Pink": "Pink", "Green": "Green", "Yellow": "Yellow", "Brown": "Brown", "Orange": "Orange"}
PATTERNS = ["Solid", "Stripe", "All over pattern", "Melange", "Denim", "Check"]
DESC_WORDS = ["soft", "cotton", "stretch", "relaxed", "slim", "fit", "lined", "ribbed", "organic", "lightweight", "warm",
              "oversized", "classic", "cropped", "high-waist", "wide", "straight", "printed", "pleated", "knitted", "recycled"]
N_STYLES = 8


def generate_demo_data(
    out_dir: str | Path,
    n_customers: int = 4000,
    n_articles: int = 1500,
    start_date: str = "2018-09-20",
    end_date: str = "2020-09-22",
    avg_tx: float = 30.0,
    seed: int = 42,
) -> dict[str, Path]:
    """Write articles.csv, customers.csv, transactions_train.csv to `out_dir`."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- articles -------------------------------------------------------
    groups = list(TAXONOMY)
    rows = []
    for k in range(n_articles):
        g = groups[rng.choice(len(groups), p=_group_probs(len(groups)))]
        index_group, types = TAXONOMY[g]
        t = types[rng.integers(len(types))]
        colour = COLOURS[rng.integers(len(COLOURS))]
        words = rng.choice(DESC_WORDS, size=4, replace=False)
        aid = 100_000_000 + k * 7 + 1
        rows.append({
            "article_id": aid,
            "product_code": aid // 1000,
            "prod_name": f"{colour} {t} {k % 97}",
            "product_type_no": 250 + groups.index(g) * 10 + types.index(t),
            "product_type_name": t,
            "product_group_name": g,
            "graphical_appearance_name": PATTERNS[rng.integers(len(PATTERNS))],
            "colour_group_name": colour,
            "perceived_colour_value_name": rng.choice(["Dark", "Light", "Medium Dusty", "Bright"]),
            "perceived_colour_master_name": COLOUR_MASTER[colour],
            "department_name": f"{index_group} {t} dept",
            "index_name": index_group,
            "index_group_name": index_group,
            "section_name": f"{index_group} {g.split()[-1]} section",
            "garment_group_name": g,
            "detail_desc": f"{colour.lower()} {t.lower()} in {' '.join(words)} fabric." if rng.random() > 0.03 else np.nan,
            "style": int(rng.integers(N_STYLES)),
            "base_price": float(np.round(rng.lognormal(mean=-3.6, sigma=0.5), 4)),
            "popularity": float(rng.pareto(1.3) + 1.0),
        })
    art = pd.DataFrame(rows)

    # ---- customers ------------------------------------------------------
    cust_ids = [f"c{idx:06d}" + "a1b2c3d4" * 6 for idx in range(n_customers)]
    # activity ~ Gamma; user taste = mixture over style clusters + favourite groups
    activity = rng.gamma(shape=1.2, scale=avg_tx / 1.2, size=n_customers)
    taste = rng.dirichlet(np.full(N_STYLES, 0.25), size=n_customers)
    fav_group = rng.integers(len(groups), size=n_customers)
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "FN": np.where(rng.random(n_customers) > 0.6, 1.0, np.nan),
        "Active": np.where(rng.random(n_customers) > 0.6, 1.0, np.nan),
        "club_member_status": rng.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], p=[0.92, 0.07, 0.01], size=n_customers),
        "fashion_news_frequency": rng.choice(["NONE", "Regularly", "Monthly"], p=[0.6, 0.38, 0.02], size=n_customers),
        "age": np.where(rng.random(n_customers) > 0.01, rng.integers(16, 75, size=n_customers), np.nan),
        "postal_code": [f"pc{rng.integers(0, 5000):05d}" for _ in range(n_customers)],
    })

    # ---- transactions ---------------------------------------------------
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    n_days = (end - start).days + 1
    days = pd.date_range(start, end)
    month = days.month.values
    types_arr = art["product_type_name"].values
    style_arr = art["style"].values
    group_arr = art["product_group_name"].values
    pop = art["popularity"].values
    # seasonal multiplier matrix: day -> item
    is_win = np.isin(types_arr, list(WINTER))
    is_sum = np.isin(types_arr, list(SUMMER))
    winter_month = np.isin(month, [11, 12, 1, 2])
    summer_month = np.isin(month, [5, 6, 7, 8])
    season = np.ones((n_days, len(art)))
    season[np.ix_(winter_month, is_win)] = 2.5
    season[np.ix_(winter_month, is_sum)] = 0.4
    season[np.ix_(summer_month, is_sum)] = 2.5
    season[np.ix_(summer_month, is_win)] = 0.4
    day_w = np.ones(n_days)
    day_w[(month == 12) | (month == 11)] = 1.4  # holiday bump
    day_w /= day_w.sum()

    recs = []
    for u in range(n_customers):
        n = int(max(1, rng.poisson(activity[u])))
        # customers appear (first purchase) at a random point, then buy until the end
        first = int(rng.integers(0, int(n_days * 0.85)))
        d_idx = np.sort(first + rng.choice(n_days - first, size=n, p=_renorm(day_w[first:])))
        affinity = taste[u][style_arr] * (1.0 + 2.0 * (group_arr == groups[fav_group[u]]))
        bought: list[int] = []
        for d in d_idx:
            if bought and rng.random() < 0.12:  # repeat purchase
                item = bought[rng.integers(len(bought))]
            else:
                p = affinity * pop * season[d]
                item = int(rng.choice(len(art), p=p / p.sum()))
            bought.append(item)
            price = art["base_price"].iat[item] * float(rng.uniform(0.7, 1.05))
            recs.append((days[d], cust_ids[u], int(art["article_id"].iat[item]), round(price, 6), int(rng.integers(1, 3))))
    tx = pd.DataFrame(recs, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
    tx["t_dat"] = tx["t_dat"].dt.strftime("%Y-%m-%d")

    art = art.drop(columns=["style", "base_price", "popularity"])
    paths = {
        "articles": out / "articles.csv",
        "customers": out / "customers.csv",
        "transactions": out / "transactions_train.csv",
    }
    art.to_csv(paths["articles"], index=False)
    customers.to_csv(paths["customers"], index=False)
    tx.to_csv(paths["transactions"], index=False)
    (out / "SYNTHETIC_DATA_README.txt").write_text(
        "This folder contains SYNTHETIC data produced by src/utils/synthetic.py for smoke testing.\n"
        "It is NOT the H&M Kaggle dataset. Never report metrics from it as real results.\n",
        encoding="utf-8",
    )
    return paths


def _group_probs(n: int) -> np.ndarray:
    p = np.array([0.22, 0.14, 0.12, 0.10, 0.08, 0.08, 0.10, 0.08, 0.08][:n], dtype=float)
    return p / p.sum()


def _renorm(p: np.ndarray) -> np.ndarray:
    return p / p.sum()
