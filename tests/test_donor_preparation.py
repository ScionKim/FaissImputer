import numpy as np
import pytest
import sklearn.metrics.pairwise as pairwise
from sklearn.exceptions import NotFittedError

from faiss_imputer import FaissImputer
from faiss_imputer._matrix import MatrixNaNIndex


def test_search_avoids_uncached_nan_distance_path(monkeypatch):
    donors = np.array([
        [0, 10, np.nan],
        [2, 30, np.nan],
        [1, np.nan, 20],
        [3, np.nan, 40],
    ], dtype=np.float64)
    before = donors.copy()
    donors.setflags(write=False)
    index = MatrixNaNIndex(donors)

    def reject_uncached_path(*args, **kwargs):
        raise AssertionError("Donor preparation must not repeat per query")

    monkeypatch.setattr(
        pairwise, "nan_euclidean_distances", reject_uncached_path
    )
    # Different queries must work without the old full-distance path.
    for value, expected_ids in [(0.1, [0, 2]), (2.9, [3, 1])]:
        query = np.array([[value, np.nan, np.nan]], dtype=np.float32)
        distances, ids = index.search(query, 2)
        np.testing.assert_array_equal(ids[0], expected_ids)
        expected = 3 * (
            before[expected_ids, 0] - float(query[0, 0])
        ) ** 2
        np.testing.assert_allclose(distances[0], expected, rtol=1e-6)
        index.clear_cache()

    np.testing.assert_array_equal(donors, before)
    assert not np.shares_memory(index.donors64, donors)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("neighbors", [1, 3])
def test_prepared_donors_preserve_results_and_inputs(strategy, neighbors):
    train = np.array([
        [0, 10, np.nan], [2, 30, np.nan], [4, 100, np.nan],
        [1, np.nan, 20], [3, np.nan, 40], [5, np.nan, 200],
    ], dtype=np.float32)
    query = np.array([
        [0.1, np.nan, np.nan],
        [2.9, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    train_before, query_before = train.copy(), query.copy()
    train.setflags(write=False)
    query.setflags(write=False)
    aggregate = np.nanmean if strategy == "mean" else np.nanmedian
    statistics = aggregate(train, axis=0)
    expected = query.copy()
    expected[:2, 1:] = (
        [[10, 20], [30, 40]] if neighbors == 1 else statistics[1:]
    )
    expected[2] = statistics

    model = FaissImputer(
        donor_policy="available",
        n_neighbors=neighbors,
        strategy=strategy,
    ).fit(train)
    for _ in range(2):
        result = model.transform(query)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        assert result.dtype == np.float32
        assert not np.shares_memory(result, query)
        assert model.available_index_.query_ref is None
        assert model.available_index_.matrix is None

    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(model.donors_, train_before)


@pytest.mark.parametrize("far,near,origin", [
    (2e-23, 1e-23, 0.0),
    (1.0001, 1.0, 0.0),
    (1e8 + 16, 1e8 + 8, 1e8),
    (2e20, 1e20, 0.0),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_preparation_keeps_numerical_fallback(far, near, origin, reverse):
    donors = [[far, 10, np.nan], [near, 20, np.nan]]
    if reverse:
        donors.reverse()
    donors.append([np.nan, np.nan, 0])
    model = FaissImputer(
        donor_policy="available", n_neighbors=1
    ).fit(donors)
    result = model.transform([[origin, np.nan, 0]])
    np.testing.assert_array_equal(
        result, np.array([[origin, 20, 0]], dtype=np.float32)
    )


@pytest.mark.parametrize("previously_fitted", [False, True])
def test_failed_refit_discards_prepared_state(previously_fitted):
    model = FaissImputer(donor_policy="available", n_neighbors=1)
    if previously_fitted:
        model.fit([[0, 10, np.nan], [1, np.nan, 20]])
        model.fit([[10, 100, np.nan], [11, np.nan, 200]])
        np.testing.assert_array_equal(
            model.transform([[10, np.nan, np.nan]]),
            [[10, 100, 200]],
        )

    with pytest.raises(ValueError):
        model.fit([[np.nan, 0, np.nan]])
    assert not hasattr(model, "available_index_")
    with pytest.raises(NotFittedError):
        model.transform([[0, np.nan, np.nan]])
