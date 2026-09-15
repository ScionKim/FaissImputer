"""Check complete-donor neighbor selection across distance scales."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["farther-first", "nearer-first"]
)
@pytest.mark.parametrize(
    "far,near",
    [
        pytest.param(2.0, 1.0, id="normal-scale"),
        pytest.param(2e-23, 1e-23, id="squared-distance-underflow"),
        pytest.param(2e20, 1e20, id="squared-distance-overflow"),
    ],
)
def test_complete_flat_l2_selects_nearest_across_scales(
    dtype, reverse, far, near
):
    train = np.array([[far, 10], [near, 20]], dtype=dtype)
    if reverse:
        train = train[::-1].copy()

    query = np.array([[0, np.nan]], dtype=dtype)
    train_before = train.copy()
    query_before = query.copy()

    # Convert stored coordinates before arithmetic so the reference
    # does not repeat float32 squared-distance underflow or overflow.
    distances = [
        (float(row[0]) - float(query[0, 0])) ** 2
        for row in train
    ]
    assert all(
        math.isfinite(distance) and distance > 0
        for distance in distances
    )
    assert distances[0] != distances[1]

    nearest = min(range(len(train)), key=distances.__getitem__)
    expected = query.copy()
    expected[0, 1] = train[nearest, 1]
    assert expected[0, 1] == 20

    with threadpool_limits(limits=1):
        result = FaissImputer(
            n_neighbors=1,
            donor_policy="complete",
            metric="l2",
            index_factory="Flat",
        ).fit(train).transform(query)

    assert result.dtype == np.dtype(dtype)
    assert_array_equal(result, expected)
    assert_array_equal(train, train_before)
    assert_array_equal(query, query_before)