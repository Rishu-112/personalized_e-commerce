import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DatasetMissingError, build_interaction_matrix, load_raw, validate_and_clean


def test_validate_drops_bad_rows():
    art = pd.DataFrame({"article_id": [1, 2, 2], "product_type_name": "a", "product_group_name": "g",
                        "colour_group_name": "c", "department_name": "d"})
    cus = pd.DataFrame({"customer_id": ["u1", "u2"]})
    tx = pd.DataFrame({"t_dat": ["2020-01-01", "bad-date", "2020-01-02", "2020-01-03", "2020-01-04"],
                       "customer_id": ["u1", "u1", "u2", "ghost", "u2"], "article_id": [1, 1, 2, 1, 99],
                       "price": [1.0, 1.0, -5.0, 1.0, 1.0]})
    a, c, t, rep = validate_and_clean(art, cus, tx)
    assert len(t) == 1 and t.iloc[0]["customer_id"] == "u1"
    assert rep["duplicates"]["articles_duplicate_ids"] == 1
    assert rep["dropped"]["non_positive_price"] == 1
    assert np.issubdtype(t["t_dat"].dtype, np.datetime64)


def test_validate_missing_columns():
    with pytest.raises(ValueError):
        validate_and_clean(pd.DataFrame({"x": [1]}), pd.DataFrame({"customer_id": ["a"]}), pd.DataFrame({"t_dat": []}))


def test_missing_dataset_is_graceful(tmp_path):
    with pytest.raises(DatasetMissingError) as e:
        load_raw(tmp_path, {}, False)
    assert "kaggle" in str(e.value).lower()


def test_chunked_sampled_loading(raw_dir):
    _, _, tx = load_raw(raw_dir, {"customer_fraction": 0.5, "history_days": 90, "chunksize": 5000}, False)
    assert 0 < tx["customer_id"].nunique() < 200
    assert tx["t_dat"].min() >= tx["t_dat"].max() - pd.Timedelta(days=90)
    _, _, tx2 = load_raw(raw_dir, {"customer_fraction": 0.5, "history_days": 90, "chunksize": 777}, False)
    assert set(tx["customer_id"]) == set(tx2["customer_id"]), "customer sampling must be deterministic"


def test_interaction_matrix_is_sparse(tables):
    from scipy import sparse
    m, users, items = build_interaction_matrix(tables[2])
    assert sparse.issparse(m) and m.shape == (len(users), len(items)) and m.sum() == len(tables[2])


def test_chronological_split_has_no_overlap(split):
    assert split["train"]["t_dat"].max() < split["val_start"] <= split["val"]["t_dat"].min()
    assert split["val"]["t_dat"].max() < split["test_start"] <= split["test"]["t_dat"].min()
