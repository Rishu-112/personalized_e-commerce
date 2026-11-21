import numpy as np
import pytest

from src.utils.metrics import (average_precision_at_k, catalog_coverage, intra_list_diversity, ndcg_at_k,
                               precision_at_k, recall_at_k)


def test_precision_recall():
    rec, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert precision_at_k(rec, rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(rec, rel, 5) == pytest.approx(2 / 3)
    assert precision_at_k(rec, rel, 2) == pytest.approx(1 / 2)
    assert recall_at_k(rec, set(), 5) == 0.0


def test_map_known_value():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / min(2, 3)
    assert average_precision_at_k([1, 9, 2], {1, 2}, 3) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_k([9, 8], {1}, 2) == 0.0


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert ndcg_at_k([7, 8, 9], {1}, 3) == 0.0
    assert ndcg_at_k([5, 1], {1}, 2) == pytest.approx(1 / np.log2(3))


def test_coverage_and_diversity():
    assert catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    assert intra_list_diversity(np.eye(3), [0, 1, 2]) == pytest.approx(1.0)
    assert intra_list_diversity(np.ones((3, 1)), [0, 1, 2]) == pytest.approx(0.0)
