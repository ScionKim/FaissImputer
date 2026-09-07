"""Representable aggregates must remain finite despite large float32 inputs."""

import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def reference_aggregate(values, strategy):
    # Python floats preserve these already-converted float32 inputs exactly.
    # Do not use the NumPy mean/median routines exercised by the estimator.
    values = sorted(float(value) for value in values if not np.isnan(value))
    if strategy == "mean":
        return math.fsum(values) / len(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return math.fsum(values[middle - 1:middle + 1]) / 2


def assert_transform_contract(
    model, train, original_train, queries, original_queries, actual, expected,
):
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all(), "Finite inputs produced a nonfinite imputation"
    np.testing.assert_array_equal(actual, expected)
    observed = ~np.isnan(original_queries)
    np.testing.assert_array_equal(actual[observed], original_queries[observed])
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(queries, original_queries)
    assert not np.shares_memory(actual, queries)
    if model.donor_policy == "available":
        index = model.available_index_
        assert index.query_ref is None
        assert index.matrix is None
        assert not index.precise_rows


@pytest.mark.parametrize("donor_policy", ["complete", "available"])
@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_fitted_statistics_and_all_missing_fallback_remain_finite(
    donor_policy, strategy, sign,
):
    # H is exactly representable, but H + H exceeds the float32 range.
    # Mean is sign * H/2; median is sign * H. Both are representable.
    high = np.float32(sign * 2**127)
    train = np.array(
        [[0, high], [1, high], [2, high], [3, -high], [4, np.nan]],
        dtype=np.float32,
    )
    queries = np.full((1, 2), np.nan, dtype=np.float32)
    original_train, original_queries = train.copy(), queries.copy()
    expected = np.array(
        [[reference_aggregate(train[:, col], strategy) for col in range(2)]],
        dtype=np.float32,
    )
    model = FaissImputer(
        n_neighbors=2, donor_policy=donor_policy, strategy=strategy,
    )

    # Reach the value assertions even when the old reductions warn about
    # overflow. An all-missing query bypasses neighbor-distance calculations.
    with np.errstate(over="ignore", invalid="ignore"):
        model.fit(train)
        actual = model.transform(queries)

    assert np.isfinite(model.statistics_).all(), "Fitted statistics overflowed"
    np.testing.assert_array_equal(model.statistics_, expected[0])
    assert_transform_contract(
        model, train, original_train, queries, original_queries, actual, expected,
    )


@pytest.mark.parametrize("donor_policy", ["complete", "available"])
@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_selected_neighbor_aggregation_remains_finite(
    donor_policy, strategy, sign,
):
    high = np.float32(sign * 2**127)
    # Alternating signs keep fitted statistics finite. Distance ordering
    # selects [H, H, H, -H], exposing overflow specifically during transform.
    train = np.array(
        [[0, high], [3, -high], [1, high], [10, -high],
         [2, high], [11, -high], [12, high], [13, -high]],
        dtype=np.float32,
    )
    queries = np.array([[0.25, np.nan], [50, high]], dtype=np.float32)
    original_train, original_queries = train.copy(), queries.copy()
    distances = (train[:, 0].astype(np.float64) - float(queries[0, 0])) ** 2
    selected = np.argsort(distances, kind="stable")[:4]
    expected = queries.copy()
    expected[0, 1] = reference_aggregate(train[selected, 1], strategy)
    model = FaissImputer(
        n_neighbors=4, donor_policy=donor_policy, strategy=strategy,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        model.fit(train)
    # A fit failure must not masquerade as a selected-neighbor failure.
    assert np.isfinite(model.statistics_).all()
    assert model.statistics_[1] == 0
    with np.errstate(over="ignore", invalid="ignore"):
        actual = model.transform(queries)

    assert_transform_contract(
        model, train, original_train, queries, original_queries, actual, expected,
    )


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_available_no_overlap_fallback_remains_finite(strategy, sign):
    high = np.float32(sign * 2**127)
    train = np.array(
        [[0, np.nan], [np.nan, high], [np.nan, high]], dtype=np.float32,
    )
    queries = np.array([[0.25, np.nan]], dtype=np.float32)
    original_train, original_queries = train.copy(), queries.copy()
    expected = queries.copy()
    expected[0, 1] = reference_aggregate(train[:, 1], strategy)
    model = FaissImputer(
        n_neighbors=3, donor_policy="available", strategy=strategy,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        actual = model.fit(train).transform(queries)

    assert_transform_contract(
        model, train, original_train, queries, original_queries, actual, expected,
    )
