"""Candidate expansion must preserve results while avoiding finished queries."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
from faiss_imputer._matrix import MatrixNaNIndex


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def reference_imputation(train, queries, k, strategy):
    """Small direct-distance reference, independent of search and its cache."""
    aggregate = np.mean if strategy == "mean" else np.median
    donors = train.astype(np.float64)
    expected = queries.copy()
    for row, query in enumerate(queries.astype(np.float64)):
        observed = ~np.isnan(query)
        shared = ~np.isnan(donors) & observed
        counts = shared.sum(axis=1)
        delta = np.where(shared, donors - query, 0.0)
        distances = np.full(len(donors), np.inf)
        valid = counts > 0
        distances[valid] = (
            (delta[valid] ** 2).sum(axis=1)
            * donors.shape[1] / counts[valid]
        )
        for col in np.flatnonzero(~observed):
            eligible = np.flatnonzero(valid & ~np.isnan(donors[:, col]))
            order = np.argsort(distances[eligible], kind="stable")
            selected = eligible[order[:k]]
            if selected.size:
                values = donors[selected, col]
            else:
                values = donors[~np.isnan(donors[:, col]), col]
            expected[row, col] = aggregate(values)
    return expected


def track_search(monkeypatch, index):
    calls = {"search": [], "prepared": 0, "direct": []}
    search = index.search
    prepared = index._prepared_distances
    direct = index._direct_distances

    def counted_search(queries, k):
        calls["search"].append((len(queries), k))
        return search(queries, k)

    def counted_prepared(queries):
        calls["prepared"] += 1
        return prepared(queries)

    def counted_direct(query):
        calls["direct"].append(query.copy())
        return direct(query)

    monkeypatch.setattr(index, "search", counted_search)
    monkeypatch.setattr(index, "_prepared_distances", counted_prepared)
    monkeypatch.setattr(index, "_direct_distances", counted_direct)
    return calls


def assert_released(index):
    assert index.query_ref is None
    assert index.matrix is None
    assert not index.precise_rows


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("reverse", [False, True])
def test_finished_queries_leave_expansion_and_sparse_targets_stop_early(
    monkeypatch, strategy, reverse,
):
    x = np.arange(64, dtype=np.float32)
    train = np.column_stack((x, 100 + x * x, np.full(64, np.nan)))
    train = train.astype(np.float32)
    train[20, 2], train[21, 2] = 10, 30
    queries = np.array(
        [[0.1, np.nan, 0], [0.1, 100, np.nan]], dtype=np.float32,
    )
    if reverse:
        queries = queries[::-1].copy()
    original_train, original_queries = train.copy(), queries.copy()
    model = FaissImputer(
        n_neighbors=3, donor_policy="available", strategy=strategy,
    ).fit(train)
    calls = track_search(monkeypatch, model.available_index_)

    actual = model.transform(queries)

    np.testing.assert_allclose(
        actual, reference_imputation(train, queries, 3, strategy), rtol=1e-6,
    )
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(queries, original_queries)
    assert_released(model.available_index_)
    # The hard target has only two observed donors; asking for three must
    # not force a final scan of all 64 donors after both have been found.
    assert calls["search"] == [(2, 16), (1, 32)]
    assert calls["prepared"] == 1


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("with_overlap", [False, True])
def test_finite_neighbor_exhaustion_preserves_partial_fill_and_fallback(
    monkeypatch, strategy, with_overlap,
):
    train = np.full((64, 2), np.nan, dtype=np.float32)
    train[0, 0] = 0
    train[1:, 1] = 10 + np.arange(63, dtype=np.float32) ** 2
    if with_overlap:
        train[1, 0] = 1
    queries = np.array([[0.1, np.nan], [np.nan, np.nan]], dtype=np.float32)
    original_train, original_queries = train.copy(), queries.copy()
    model = FaissImputer(
        n_neighbors=3, donor_policy="available", strategy=strategy,
    ).fit(train)
    calls = track_search(monkeypatch, model.available_index_)

    actual = model.transform(queries)

    np.testing.assert_allclose(
        actual, reference_imputation(train, queries, 3, strategy), rtol=1e-6,
    )
    np.testing.assert_array_equal(actual[1], model.statistics_)
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(queries, original_queries)
    assert_released(model.available_index_)
    # Most target donors share no observed feature with the query. Once
    # returned candidates contain infinity, expansion cannot add a donor.
    assert calls["search"] == [(1, 16)]
    assert calls["prepared"] == 1


def test_retained_queries_reuse_precise_rows_and_allow_later_refinement(monkeypatch):
    train = np.array(
        [[i / 100, np.nan, 0] for i in range(1, 16)]
        + [[-1, 10, np.nan], [1, 20, np.nan]], dtype=np.float32,
    )
    queries = np.array(
        [[-0.2, np.nan, 0], [1, np.nan, 0], [1e-8, np.nan, 0]],
        dtype=np.float32,
    )
    index = MatrixNaNIndex(train)
    calls = track_search(monkeypatch, index)
    index.search(queries, 1)
    assert set(index.precise_rows) == {1}
    exact = index.precise_rows[1]
    matrix = index.matrix

    retained = index.retain_queries(np.array([1, 2]))

    assert retained is index.query_ref
    np.testing.assert_array_equal(retained, queries[[1, 2]])
    np.testing.assert_array_equal(index.matrix, matrix[[1, 2]])
    assert set(index.precise_rows) == {0}
    assert index.precise_rows[0] is exact
    # Expansion exposes the +/-1 float32 tie for the second retained row.
    # Its positive donor is closer in float64, including across the cutoff.
    _, ids = index.search(retained, 16)
    assert ids[0, 0] == 16
    assert ids[1, -1] == 16
    assert set(index.precise_rows) == {0, 1}
    assert index.precise_rows[0] is exact
    index.search(retained, 17)
    assert calls["prepared"] == 1
    assert len(calls["direct"]) == 2
    np.testing.assert_array_equal(calls["direct"], queries[[1, 2]])
    index.clear_cache()
    assert_released(index)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("scale", [1e-30, 1e20])
def test_compaction_reuses_underflow_and_overflow_refinement(
    monkeypatch, strategy, scale,
):
    x = np.arange(64, dtype=np.float32)
    train = np.column_stack(
        ((x + 1) * scale, 100 + x * x, np.full(64, np.nan)),
    ).astype(np.float32)
    train[20, 2], train[21, 2] = 10, 30
    queries = np.array(
        [[1.1 * scale, np.nan, 0], [1.1 * scale, np.nan, np.nan]],
        dtype=np.float32,
    )
    original_train, original_queries = train.copy(), queries.copy()
    model = FaissImputer(
        n_neighbors=3, donor_policy="available", strategy=strategy,
    ).fit(train)
    calls = track_search(monkeypatch, model.available_index_)

    actual = model.transform(queries)

    np.testing.assert_allclose(
        actual, reference_imputation(train, queries, 3, strategy), rtol=1e-6,
        atol=0,
    )
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(queries, original_queries)
    assert_released(model.available_index_)
    assert calls["search"] == [(2, 16), (1, 32)]
    assert calls["prepared"] == 1
    assert len(calls["direct"]) == 2
    np.testing.assert_array_equal(calls["direct"], queries)
