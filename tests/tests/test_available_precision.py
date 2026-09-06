"""Focused public and backend regressions using actual computed distances."""

import faiss
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
from faiss_imputer._matrix import MatrixNaNIndex


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def assert_released(model):
    index = model.available_index_
    assert index.query_ref is None
    assert index.matrix is None
    assert not index.precise_rows


def test_whole_tie_group_includes_eligible_donors_outside_probe():
    # Adapt only to FAISS's unspecified order among equal float32 values.
    # Both donors capable of filling the query are deliberately outside its
    # first k+1 candidates; no distance or product routine is mocked.
    query = np.array([[1e-8, np.nan, 0]], dtype=np.float32)
    train = np.array(
        [[i / 100, np.nan, 0] for i in range(1, 16)]
        + [[-1, np.nan, np.nan] for _ in range(6)], dtype=np.float32,
    )
    initial = MatrixNaNIndex(train)
    rounded = initial._prepared_distances(query.astype(np.float64)).astype(np.float32)
    _, probe_ids = faiss.kmin(rounded, 17)
    omitted = np.setdiff1d(np.arange(15, 21), probe_ids[0])
    assert omitted.size >= 2
    positive, negative = omitted[:2]
    train[positive] = [1, 20, np.nan]
    train[negative] = [-1, 10, np.nan]
    index = MatrixNaNIndex(train)
    actual = index._prepared_distances(query.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(actual, rounded)
    assert positive not in probe_ids[0] and negative not in probe_ids[0]
    _, ids = index.search(query, 16)
    assert ids[0, -1] == positive


def test_true_ties_keep_training_order_and_cache_across_expansion(monkeypatch):
    train = np.array(
        [[(-1) ** i, 10 + i, np.nan] for i in range(24)], dtype=np.float32,
    )
    query = np.array([[0, np.nan, 0]], dtype=np.float32)
    index = MatrixNaNIndex(train)
    direct = index._direct_distances
    calls = []

    def counted(row):
        calls.append(row.copy())
        return direct(row)

    monkeypatch.setattr(index, "_direct_distances", counted)
    for k in (1, 3, 7, 16, 24):
        values, ids = index.search(query, k)
        np.testing.assert_array_equal(ids[0], np.arange(k))
        np.testing.assert_array_equal(values[0], np.full(k, 3.0))
    assert len(calls) == 1
    index.clear_cache()
    assert index.query_ref is None and index.matrix is None
    assert not index.precise_rows
    index.search(query.copy(), 3)
    assert len(calls) == 2


def test_ordinary_unique_distances_keep_fast_path(monkeypatch):
    train = np.array([[i, 10 * i, np.nan] for i in range(1, 10)], dtype=np.float32)
    query = np.array([[0.25, np.nan, 0]], dtype=np.float32)
    index = MatrixNaNIndex(train)

    def unexpected(_):
        pytest.fail("Unique ordinary distances should not require direct refinement")

    monkeypatch.setattr(index, "_direct_distances", unexpected)
    for k in (1, 3, 9):
        values, ids = index.search(query, k)
        np.testing.assert_array_equal(ids[0], np.arange(k))
        assert values.dtype == np.float32
    assert not index.precise_rows


@pytest.mark.parametrize("offset, expected", [(1e-8, 20), (-1e-8, 10)])
@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_public_mixed_row_can_expand_search_without_changing_target(
    monkeypatch, offset, expected, strategy,
):
    train = np.array(
        [[i / 100, np.nan, 0, np.nan] for i in range(1, 16)]
        + [[-1, 10, np.nan, np.nan], [1, 20, np.nan, np.nan]]
        + [[10 + i, np.nan, 0, np.nan] for i in range(23)]
        + [[100, 100, 0, 50]], dtype=np.float32,
    )
    target = np.array([[offset, np.nan, 0, 0]], dtype=np.float32)
    mixed = np.vstack((target, [offset, 0, 0, np.nan])).astype(np.float32)
    model = FaissImputer(n_neighbors=1, donor_policy="available", strategy=strategy).fit(train)
    original = MatrixNaNIndex.search
    calls = []

    def counted(index, queries, k):
        calls.append(k)
        return original(index, queries, k)

    monkeypatch.setattr(MatrixNaNIndex, "search", counted)
    alone = model.transform(target)
    assert calls == [16]
    assert_released(model)
    calls.clear()
    together = model.transform(mixed)
    assert calls == [16, 32, 41]
    assert alone[0, 1] == together[0, 1] == expected
    assert together[1, 3] == 50
    np.testing.assert_array_equal(together[0], alone[0])
    assert_released(model)
    reversed_result = model.transform(mixed[::-1].copy())
    np.testing.assert_array_equal(reversed_result[::-1], together)
    assert_released(model)


@pytest.mark.parametrize("strategy, expected", [("mean", 40), ("median", 20)])
def test_public_true_ties_aggregate_training_prefix(strategy, expected):
    values = [10, 20, 90] + list(range(100, 117))
    train = np.array(
        [[(-1) ** i, value, np.nan] for i, value in enumerate(values)]
        + [[np.nan, np.nan, 0]],
        dtype=np.float32,
    )
    target = np.array([[0, np.nan, 0]], dtype=np.float32)
    mixed = np.array([[0, np.nan, 0], [1, np.nan, 0]], dtype=np.float32)
    model = FaissImputer(n_neighbors=3, donor_policy="available", strategy=strategy).fit(train)
    alone = model.transform(target)
    together = model.transform(mixed)
    assert alone[0, 1] == together[0, 1] == expected
    assert_released(model)
