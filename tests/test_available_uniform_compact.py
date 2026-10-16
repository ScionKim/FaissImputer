"""Regressions for available uniform means using captured donor values."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


EVIDENCE = json.loads(
    (Path(__file__).parent / "data/available_uniform_float32_38681953970.json")
    .read_text(encoding="utf-8")
)
CASES = EVIDENCE["cases"]


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def case_id(case):
    row, column = case["original_query_index"]
    return f"seed{case['seed']}-row{row}-column{column}"


@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("weights", ["uniform", None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_captured_five_donor_means(case, weights, dtype):
    # Unique increasing distances retain the captured donor order without
    # relying on another NumPy version's argpartition ordering in KNNImputer.
    train = np.full((16, 2), np.nan, dtype=dtype)
    train[:, 0] = np.arange(1, 17)
    train[case["selected_slots"], 1] = case["ordered_donor_values"]
    query = np.array([[0, np.nan]], dtype=dtype)
    train_before, query_before = train.copy(), query.copy()

    result = FaissImputer(
        n_neighbors=5, donor_policy="available", weights=weights,
    ).fit(train).transform(query)

    if dtype == np.float32:
        # The diagnostic captured this compact-order mean and independently
        # found it equal to the archived KNN output for each of the 12 cells.
        expected = case["archived_knn_output"]
    else:
        expected = math.fsum(case["ordered_donor_values"]) / 5
    np.testing.assert_allclose(result[0, 1], expected, rtol=3e-6, atol=0)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result[:, 0], query[:, 0])
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partial_donor_counts_and_no_overlap_fallback(dtype):
    train = np.array(
        [[0, 10, np.nan], [1, np.nan, 100], [2, 30, np.nan],
         [np.nan, np.nan, 200], [np.nan, np.nan, np.nan]], dtype=dtype,
    )
    query = np.array(
        [[0, np.nan, np.nan], [np.nan, np.nan, 150],
         [np.nan, np.nan, np.nan]], dtype=dtype,
    )
    original = query.copy()

    result = FaissImputer(
        n_neighbors=5, donor_policy="available",
    ).fit(train).transform(query)

    # First row: two donors for column 1, one finite-distance donor for column 2.
    # Second row: no finite-distance donor for column 1; use its fitted mean.
    # Third row: the entirely missing query uses all fitted statistics.
    np.testing.assert_array_equal(result, [[0, 20, 100], [1, 20, 150], [1, 20, 150]])
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(query, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_uniform_mean_retains_overflow_repair(dtype):
    maximum = np.finfo(dtype).max
    values = np.full((1, 5), maximum, dtype=dtype)

    result = FaissImputer()._aggregate(values, axis=1, ignore_nan=True)

    np.testing.assert_array_equal(result, np.array([maximum], dtype=dtype))
    assert result.dtype == np.dtype(dtype)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "strategy,weights,expected",
    [("median", "uniform", 8), ("mean", "distance", 76 / 11)],
)
def test_other_available_aggregation_modes(dtype, strategy, weights, expected):
    train = np.array([[1, 2], [2, 8], [3, 20], [4, np.nan]], dtype=dtype)
    query = np.array([[0, np.nan]], dtype=dtype)

    result = FaissImputer(
        n_neighbors=3, donor_policy="available",
        strategy=strategy, weights=weights,
    ).fit(train).transform(query)

    np.testing.assert_allclose(result, [[0, expected]], rtol=3e-6, atol=0)
    assert result.dtype == np.dtype(dtype)
