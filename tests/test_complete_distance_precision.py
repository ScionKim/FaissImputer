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

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-23, 1e20])
@pytest.mark.parametrize("metric", ["l2", "nan_euclidean"])
@pytest.mark.parametrize(
    "strategy,weights",
    [
        ("mean", "uniform"),
        ("median", "uniform"),
        ("mean", "distance"),
    ],
)
def test_repaired_neighbors_support_aggregation_and_weights(
    dtype, scale, metric, strategy, weights
):
    train = np.array(
        [
            [4 * scale, 400],
            [2 * scale, 20],
            [3 * scale, 90],
            [scale, 10],
        ],
        dtype=dtype,
    )
    query = np.array([[0, np.nan]], dtype=dtype)

    distances = [
        (float(row[0]) - float(query[0, 0])) ** 2
        for row in train
    ]
    nearest = sorted(
        range(len(train)), key=distances.__getitem__
    )[:3]
    values = [float(train[index, 1]) for index in nearest]

    if strategy == "median":
        replacement = sorted(values)[1]
    elif weights == "distance":
        distance_weights = [
            1.0 / math.sqrt(distances[index])
            for index in nearest
        ]
        replacement = math.fsum(
            value * weight
            for value, weight in zip(values, distance_weights)
        ) / math.fsum(distance_weights)
    else:
        replacement = math.fsum(values) / len(values)

    expected = query.copy()
    expected[0, 1] = replacement

    with threadpool_limits(limits=1):
        result = FaissImputer(
            n_neighbors=3,
            donor_policy="complete",
            metric=metric,
            index_factory="Flat",
            strategy=strategy,
            weights=weights,
        ).fit(train).transform(query)

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=5 * np.finfo(dtype).eps,
        atol=0,
    )


@pytest.mark.parametrize("n_queries", [1, 32])
def test_large_coordinates_preserve_neighbors_across_query_batches(
    n_queries,
):
    origin = np.float32(1e20)
    step = float(np.spacing(origin))
    train = np.array(
        [
            [float(origin) + 4 * step, 40],
            [float(origin) + step, 10],
            [float(origin) - 2 * step, 20],
        ],
        dtype=np.float32,
    )
    query = np.full((n_queries, 2), np.nan, dtype=np.float32)
    query[:, 0] = origin
    expected = query.copy()
    expected[:, 1] = 10

    with threadpool_limits(limits=1):
        result = FaissImputer(
            n_neighbors=1,
            donor_policy="complete",
            metric="l2",
            index_factory="Flat",
        ).fit(train).transform(query)

    assert_array_equal(result, expected)

def test_repaired_distance_ties_follow_training_row_order():
    train = np.tile(
        np.array([[2e20, 100]], dtype=np.float32),
        (70000, 1),
    )
    train[[0, -2, -1], 0] = np.float32(1e20)
    train[[0, -2, -1], 1] = [10, 20, 90]
    query = np.array([[0, np.nan]], dtype=np.float32)

    with threadpool_limits(limits=1):
        result = FaissImputer(
            n_neighbors=2,
            donor_policy="complete",
            metric="l2",
            index_factory="Flat",
        ).fit(train).transform(query)

    # The first two tied donors in training order contain 10 and 20.
    assert_array_equal(result, np.array([[0, 15]], dtype=np.float32))