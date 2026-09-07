"""Keep complete-donor aggregation behavior when several queries are reduced together."""

import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import faiss_imputer.faiss_imputer as implementation
from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def install_fixed_neighbors(monkeypatch, ids):
    """Isolate aggregation from distance rounding and neighbor tie decisions."""
    ids = np.asarray(ids, dtype=np.int64)

    class FixedIndex:
        def train(self, values):
            pass

        def add(self, values):
            pass

        def search(self, queries, k):
            assert k == ids.shape[1]
            # Every fixture retains feature zero as an observed row id.
            query_ids = queries[:, 0].astype(np.int64)
            selected = ids[query_ids].copy()
            return np.zeros(selected.shape, dtype=np.float32), selected

    monkeypatch.setattr(implementation.faiss, "index_factory", lambda *args: FixedIndex())


def legacy_rowwise_result(model, queries, ids):
    """Reference the previous reduction layout, including finite float32 rounding."""
    result = queries.copy()
    aggregate = np.mean if model.strategy == "mean" else np.median
    for row, query in enumerate(queries):
        missing = np.flatnonzero(np.isnan(query))
        if not missing.size:
            continue
        if missing.size == query.size:
            result[row] = model.statistics_
            continue
        neighbors = ids[int(query[0])]
        neighbors = neighbors[neighbors >= 0]
        values = model.donors_[neighbors][:, missing]
        result[row, missing] = aggregate(values, axis=0)
    return result


def assert_contract(actual, expected, train, train_before, queries, queries_before):
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    observed = ~np.isnan(queries_before)
    np.testing.assert_array_equal(actual[observed], queries_before[observed])
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(queries, queries_before)
    assert not np.shares_memory(actual, queries)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("k", [1, 5, 16, 129])
def test_complete_many_queries_preserve_rowwise_float32_results(monkeypatch, strategy, k):
    rng = np.random.default_rng(20260907)
    # Small residuals mixed with larger opposite signs expose changes in
    # reduction order. These values do not overflow float32 aggregation.
    values = np.array([-2**24, 2**24, 1, -1, 0.25, -0.25, 9], dtype=np.float32)
    train = rng.choice(values, size=(257, 9))
    train[:, 0] = np.arange(len(train))
    count = 773
    queries = np.zeros((count + 2, 9), dtype=np.float32)
    queries[:, 0] = np.arange(len(queries))
    for row in range(count):
        queries[row, [2, 4] if row % 3 == 0 else [1, 3, 5, 7]] = np.nan
    queries[-1] = np.nan
    # Two repeated patterns exceed 256 rows; an observed row and an all-missing
    # row also exercise the bypass and fitted-statistic fallback paths.
    ids = (np.arange(count)[:, None] * 13 + np.arange(k)[None, :] * 7) % len(train)
    train_before, queries_before = train.copy(), queries.copy()
    model = FaissImputer(n_neighbors=k, strategy=strategy).fit(train)
    donors_before = model.donors_.copy()
    expected = legacy_rowwise_result(model, queries, ids)
    install_fixed_neighbors(monkeypatch, ids)

    actual = model.transform(queries)
    assert_contract(actual, expected, train, train_before, queries, queries_before)
    np.testing.assert_array_equal(model.transform(queries[::-1]), expected[::-1])
    np.testing.assert_array_equal(model.donors_, donors_before)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_complete_mixed_safe_and_overflowing_aggregates_remain_finite(monkeypatch, strategy):
    high = np.float32(2**127)
    train = np.array([
        [0, high, -high, 4, high],
        [1, high, -high, 5, high],
        [2, high, -high, 6, -high],
        [3, -high, high, 7, -high],
        [4, -high, high, 8, -high],
        [5, -high, high, 9, -high],
        [6, -high, high, 10, high],
        [7, high, -high, 11, high],
        [8, 1, -2, 12, 4],
        [9, 2, -3, 13, 8],
        [10, 3, -4, 14, 12],
        [11, 4, -5, 15, 16],
    ], dtype=np.float32)
    count = 513
    queries = np.zeros((count, 5), dtype=np.float32)
    queries[:, 0] = np.arange(count)
    queries[:, [1, 2, 4]] = np.nan
    ids = np.arange(12).reshape(3, 4)[np.arange(count) % 3]
    expected = queries.copy()
    # Independent float64/Python reference for representable extreme results.
    for row, neighbors in enumerate(ids):
        for col in (1, 2, 4):
            selected = sorted(float(value) for value in train[neighbors, col])
            expected[row, col] = (
                math.fsum(selected) / len(selected) if strategy == "mean"
                else math.fsum(selected[1:3]) / 2
            )
    train_before, queries_before = train.copy(), queries.copy()
    model = FaissImputer(n_neighbors=4, strategy=strategy).fit(train)
    install_fixed_neighbors(monkeypatch, ids)

    actual = model.transform(queries)
    assert_contract(actual, expected, train, train_before, queries, queries_before)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_complete_partial_neighbors_preserve_counts_and_duplicates(monkeypatch, strategy):
    train = np.arange(24, dtype=np.float32).reshape(6, 4)
    queries = np.zeros((5, 4), dtype=np.float32)
    queries[:, 0] = np.arange(len(queries))
    queries[:, [1, 3]] = np.nan
    ids = np.array([
        [0, 1, 2, 3],
        [0, -1, 2, -1],
        [-1, 2, -1, -1],
        [3, 3, 1, -1],
        [-2, 0, 1, 2],
    ], dtype=np.int64)
    train_before, queries_before = train.copy(), queries.copy()
    model = FaissImputer(n_neighbors=4, strategy=strategy).fit(train)
    expected = legacy_rowwise_result(model, queries, ids)
    install_fixed_neighbors(monkeypatch, ids)

    actual = model.transform(queries)
    assert_contract(actual, expected, train, train_before, queries, queries_before)


@pytest.mark.parametrize("empty_row", [0, 256, 512])
def test_complete_no_valid_neighbors_still_raises(monkeypatch, empty_row):
    train = np.arange(24, dtype=np.float32).reshape(6, 4)
    queries = np.zeros((513, 4), dtype=np.float32)
    queries[:, 0] = np.arange(len(queries))
    queries[:, [1, 3]] = np.nan
    ids = np.tile(np.arange(4), (len(queries), 1))
    ids[empty_row] = -1
    train_before, queries_before = train.copy(), queries.copy()
    model = FaissImputer(n_neighbors=4).fit(train)
    install_fixed_neighbors(monkeypatch, ids)

    with pytest.raises(ValueError, match="FAISS did not return any valid neighbors"):
        model.transform(queries)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(queries, queries_before)
