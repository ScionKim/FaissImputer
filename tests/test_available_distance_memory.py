"""Regression checks for bounded search-preparation memory."""

import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from faiss_imputer._matrix import MatrixNaNIndex


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def _make_case(dtype, reverse):
    donors = np.empty((4096, 3), dtype=dtype)
    donors[:, 0] = 4 * np.arange(4096)
    donors[:, 1] = 10 + np.arange(4096)
    donors[:, 2] = np.nan
    donors[::2, 2] = 5

    queries = np.full((129, 3), np.nan, dtype=dtype)
    queries[:, 0] = 4 * np.arange(129) + 1

    # Exercise the start, both sides of chunk boundaries, and the tail.
    queries[0, 0] = 1e-23
    queries[63, 0] = donors[63, 0]
    queries[64, 0] = 1e20
    queries[127, 0] = donors[127, 0]
    queries[128, 0] = 1e-23
    precise_rows = np.array([0, 63, 64, 127, 128])

    if reverse:
        queries = queries[::-1].copy()
        precise_rows = len(queries) - 1 - precise_rows

    return donors, queries, precise_rows


def _reference_neighbors(donors, queries, k):
    # Only column 0 is observed in these queries and shared by every donor.
    donor_x = donors[:, 0].astype(np.float64)
    values = np.empty((len(queries), k), dtype=np.float64)
    ids = np.empty((len(queries), k), dtype=np.int64)

    for row, query_x in enumerate(queries[:, 0]):
        distances = (donor_x - float(query_x)) ** 2 * donors.shape[1]
        ids[row] = np.argsort(distances, kind="stable")[:k]
        values[row] = distances[ids[row]]

    return values, ids


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_chunked_search_preserves_precision_cache_and_inputs(
    dtype, reverse, monkeypatch,
):
    donors, queries, precise_rows = _make_case(dtype, reverse)
    donors_before = donors.copy()
    queries_before = queries.copy()
    donors.setflags(write=False)
    queries.setflags(write=False)

    expected_values, expected_ids = _reference_neighbors(donors, queries, 3)
    index = MatrixNaNIndex(donors)

    prepared_refs = []
    original_prepared = index._prepared_distances
    original_direct = index._direct_distances

    def tracked_prepared(rows):
        result = original_prepared(rows)
        prepared_refs.append(weakref.ref(result))
        return result

    def checked_direct(row):
        # No full float64 preparation matrix may remain alive here.
        assert prepared_refs
        assert all(reference() is None for reference in prepared_refs)
        return original_direct(row)

    monkeypatch.setattr(index, "_prepared_distances", tracked_prepared)
    monkeypatch.setattr(index, "_direct_distances", checked_direct)

    for k in (1, 3):
        values, ids = index.search(queries, k)
        assert_array_equal(ids, expected_ids[:, :k])
        assert_allclose(
            values, expected_values[:, :k], rtol=1e-12, atol=0,
        )
        assert set(index.precise_rows) == set(precise_rows.tolist())

    assert len(prepared_refs) == 1

    selected = np.array([128, 64, 0, 127, 63, 1])
    retained = index.retain_queries(selected)
    values, ids = index.search(retained, 3)
    assert_array_equal(ids, expected_ids[selected])
    assert_allclose(
        values, expected_values[selected], rtol=1e-12, atol=0,
    )
    assert len(prepared_refs) == 1

    index.clear_cache()
    values, ids = index.search(queries, 3)
    assert_array_equal(ids, expected_ids)
    assert_allclose(values, expected_values, rtol=1e-12, atol=0)
    assert len(prepared_refs) == 2

    assert_array_equal(donors, donors_before)
    assert_array_equal(queries, queries_before)
    assert not donors.flags.writeable
    assert not queries.flags.writeable


def test_chunked_suspicion_requires_a_shared_and_donating_feature():
    donors = np.tile([0.0, np.nan, 5.0], (4096, 1))
    donors[-1] = [2.0, 10.0, np.nan]
    queries = np.tile([0.0, np.nan, 5.0], (131, 1))

    precise_rows = np.array([63, 64, 127, 128])
    queries[precise_rows, 0] = 2.0
    queries[129] = [np.nan, 12.0, np.nan]
    queries[130] = np.nan

    donors_before = donors.copy()
    queries_before = queries.copy()
    donors.setflags(write=False)
    queries.setflags(write=False)

    matrix, suspect = MatrixNaNIndex(donors)._prepare_search_matrix(queries)

    # Ordinary zero-distance matches cannot fill the missing target.
    # Rows 129 and 130 also contain pairs with no shared observed feature.
    assert_array_equal(np.flatnonzero(suspect), precise_rows)
    assert matrix.shape == (131, 4096)
    assert matrix.dtype == np.float32

    first_donor = np.zeros(129, dtype=np.float32)
    first_donor[precise_rows] = 6
    last_donor = np.full(129, 12, dtype=np.float32)
    last_donor[precise_rows] = 0

    assert_array_equal(matrix[:129, 0], first_donor)
    assert_array_equal(matrix[:129, -1], last_donor)
    assert np.isinf(matrix[129, :-1]).all()
    assert matrix[129, -1] == 12
    assert np.isinf(matrix[130]).all()

    assert_array_equal(donors, donors_before)
    assert_array_equal(queries, queries_before)