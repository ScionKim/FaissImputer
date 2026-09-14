"""Regression checks for prepared available-donor distances."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from faiss_imputer._matrix import MatrixNaNIndex


DONORS = [
    [0, 10, np.nan, 4, np.nan],
    [2, np.nan, 20, 8, 1],
    [np.nan, 30, 40, np.nan, 3],
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [4, 50, 60, 12, 5],
]

QUERIES = [
    [1, np.nan, 22, 6, np.nan],
    [np.nan, 12, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, 10, np.nan, 4, np.nan],
    [3, 18, 35, 7, 2],
]


def _readonly_array(values, dtype, layout):
    array = np.array(values, dtype=dtype)
    if layout == "fortran":
        array = np.asfortranarray(array)
    elif layout == "strided":
        backing = np.full(
            (array.shape[0], array.shape[1] * 2),
            -99,
            dtype=dtype,
        )
        backing[:, ::2] = array
        array = backing[:, ::2]
    array.setflags(write=False)
    return array


def _reference_distances(donors, queries, n_features):
    donors = np.asarray(donors, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    expected = np.full(
        (len(queries), len(donors)),
        np.nan,
        dtype=np.float64,
    )

    for query_id, query in enumerate(queries):
        for donor_id, donor in enumerate(donors):
            shared = ~np.isnan(query) & ~np.isnan(donor)
            count = np.count_nonzero(shared)
            if count:
                delta = query[shared] - donor[shared]
                expected[query_id, donor_id] = (
                    np.sum(delta * delta) / count * n_features
                )
    return expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("n_features", [None, 9])
def test_prepared_distances_match_direct_reference(
    dtype, layout, n_features,
):
    donors = _readonly_array(DONORS, dtype, layout)
    queries = _readonly_array(QUERIES, dtype, layout)
    donors_before = donors.copy()
    queries_before = queries.copy()
    feature_count = donors.shape[1] if n_features is None else n_features

    index = MatrixNaNIndex(donors, n_features=n_features)
    result = index._prepared_distances(queries)
    expected = _reference_distances(donors, queries, feature_count)

    assert result.dtype == np.float64
    assert_allclose(
        result, expected, rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    assert_array_equal(donors, donors_before)
    assert_array_equal(queries, queries_before)
    assert not donors.flags.writeable
    assert not queries.flags.writeable
    assert not np.shares_memory(result, donors)
    assert not np.shares_memory(result, queries)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shared_feature_counts_above_255(dtype):
    donors = np.ones((4, 257), dtype=dtype)
    donors[1, -1] = np.nan
    donors[2] = np.nan
    donors[2, 0] = 2
    donors[3] = np.nan

    queries = np.zeros((2, 257), dtype=dtype)
    queries[1, 0] = np.nan

    result = MatrixNaNIndex(donors)._prepared_distances(queries)

    # Shared counts include 257, 256, 255, 1, and 0.
    expected = np.array([
        [257, 257, 1028, np.nan],
        [257, 257, np.nan, np.nan],
    ])
    assert_allclose(
        result, expected, rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_later_distance_calculation_preserves_previous_result(dtype):
    donors = np.array(DONORS, dtype=dtype)
    first_query = np.array([[1, np.nan, 22, 6, np.nan]], dtype=dtype)
    next_query = np.array([[4, np.nan, 18, 9, np.nan]], dtype=dtype)
    index = MatrixNaNIndex(donors)

    first = index._prepared_distances(first_query)
    first_before = first.copy()
    second = index._prepared_distances(next_query)

    assert_array_equal(first, first_before)
    assert not np.shares_memory(first, second)
    assert_allclose(
        second,
        _reference_distances(donors, next_query, donors.shape[1]),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )