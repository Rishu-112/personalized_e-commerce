import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import validate_and_clean  # noqa: E402
from src.feature_engineering import build_context, chronological_split  # noqa: E402
from src.inference import RecommenderService  # noqa: E402
from src.models.collaborative import ItemKNNModel  # noqa: E402
from src.models.content_based import ContentBasedModel  # noqa: E402
from src.models.hybrid import HybridModel  # noqa: E402
from src.models.matrix_factorization import MatrixFactorizationModel  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402
from src.utils.synthetic import generate_demo_data  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate_demo_data(d, n_customers=200, n_articles=150, start_date="2020-01-01", end_date="2020-09-22", avg_tx=14, seed=1)
    return d


@pytest.fixture(scope="session")
def tables(raw_dir):
    art = pd.read_csv(raw_dir / "articles.csv")
    cus = pd.read_csv(raw_dir / "customers.csv")
    tx = pd.read_csv(raw_dir / "transactions_train.csv")
    return validate_and_clean(art, cus, tx)


@pytest.fixture(scope="session")
def split(tables):
    return chronological_split(tables[2], val_days=21, test_days=21)


@pytest.fixture(scope="session")
def ctx(tables, split):
    art, cus, _, _ = tables
    hist = pd.concat([split["train"], split["val"]])
    return build_context(hist, art, cus, split["test_start"])


@pytest.fixture(scope="session")
def models(ctx):
    m = {
        "popularity": PopularityModel("global").fit(ctx),
        "popularity_category": PopularityModel("category").fit(ctx),
        "content": ContentBasedModel(2000).fit(ctx),
        "collaborative": ItemKNNModel(20, 100).fit(ctx),
        "matrix_factorization": MatrixFactorizationModel(8, 0.05, 3, 20.0, prefer_implicit_library=False).fit(ctx),
    }
    h = HybridModel().fit(ctx, {k: m[k] for k in ("content", "collaborative", "matrix_factorization")})
    m["hybrid"] = h
    return m


@pytest.fixture(scope="session")
def service(ctx, models, tmp_path_factory):
    meta = {"is_synthetic": True, "history_end": ctx.now}
    return RecommenderService({"ctx": ctx, "models": models, "meta": meta}, tmp_path_factory.mktemp("art"))
