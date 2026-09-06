"""Integral neighbor counts through public estimator and model-selection APIs."""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("donor_policy", ["complete", "available"])
@pytest.mark.parametrize("integer_type", [int, np.int32, np.int64, np.uint8, np.uint64])
def test_integral_neighbors_preserve_parameters_and_predictions(
    donor_policy, integer_type,
):
    train = np.array([[0, 10], [2, 20], [4, 40], [6, np.nan]], dtype=np.float32)
    queries = np.array([[0.25, np.nan], [np.nan, 25], [np.nan, np.nan]], dtype=np.float32)
    original_train, original_queries = train.copy(), queries.copy()
    neighbors = integer_type(2)
    estimator = FaissImputer(n_neighbors=neighbors, donor_policy=donor_policy)
    assert estimator.get_params()["n_neighbors"] is neighbors

    copied = clone(estimator)
    copied_neighbors = copied.get_params()["n_neighbors"]
    assert type(copied_neighbors) is integer_type
    assert copied.fit(train) is copied
    assert copied.get_params()["n_neighbors"] is copied_neighbors

    result = copied.transform(queries)
    native = FaissImputer(n_neighbors=2, donor_policy=donor_policy).fit(train)
    np.testing.assert_array_equal(result, native.transform(queries))
    assert result[0, 1] == 15
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(queries, original_queries)

    copied_after_fit = clone(copied)
    assert type(copied_after_fit.n_neighbors) is integer_type
    with pytest.raises(NotFittedError):
        copied_after_fit.transform(queries)
    copied_after_fit.fit(train)
    np.testing.assert_array_equal(copied_after_fit.transform(queries), result)


@pytest.mark.parametrize("donor_policy", ["complete", "available"])
@pytest.mark.parametrize("invalid", [
    pytest.param(0, id="python-zero"),
    pytest.param(np.int64(-1), id="numpy-negative"),
    pytest.param(np.uint8(0), id="numpy-unsigned-zero"),
    pytest.param(2.0, id="python-float"),
    pytest.param(np.float64(2), id="numpy-float"),
    pytest.param(True, id="python-true"),
    pytest.param(False, id="python-false"),
    pytest.param(np.bool_(True), id="numpy-true"),
    pytest.param(np.bool_(False), id="numpy-false"),
    pytest.param("2", id="string"),
    pytest.param(None, id="none"),
    pytest.param(np.array(2), id="zero-dimensional-array"),
])
def test_invalid_neighbors_clear_failed_refit_state(donor_policy, invalid):
    train = np.array([[0, 10], [2, 20], [4, 40]], dtype=np.float32)
    original = train.copy()
    estimator = FaissImputer(n_neighbors=1, donor_policy=donor_policy).fit(train)
    estimator.set_params(n_neighbors=invalid)
    with pytest.raises(ValueError, match="n_neighbors must be a positive integer"):
        estimator.fit(train)
    with pytest.raises(NotFittedError):
        estimator.transform([[0.25, np.nan]])
    with pytest.raises(NotFittedError):
        estimator.get_feature_names_out()
    np.testing.assert_array_equal(train, original)


@pytest.mark.parametrize("neighbors", [np.int64(4), np.uint64(2**64 - 1)])
def test_large_integral_neighbors_keep_policy_specific_donor_limits(neighbors):
    train = np.array([[0, 10], [1, np.nan], [2, 30]], dtype=np.float32)
    complete = FaissImputer(n_neighbors=neighbors, donor_policy="complete")
    with pytest.raises(ValueError, match="number of complete donors"):
        complete.fit(train)
    with pytest.raises(NotFittedError):
        complete.transform([[0.25, np.nan]])

    available = FaissImputer(n_neighbors=neighbors, donor_policy="available")
    available.fit(train)
    np.testing.assert_array_equal(available.transform([[0.25, np.nan]]), [[0.25, 20]])


def test_available_uint8_neighbors_do_not_overflow_during_candidate_expansion():
    x = np.arange(300, dtype=np.float32)
    train = np.column_stack((x, 2 * x))
    estimator = FaissImputer(n_neighbors=np.uint8(200), donor_policy="available")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimator.fit(train)
        result = estimator.transform([[0.25, np.nan]])
    np.testing.assert_array_equal(result, [[0.25, 199]])


@pytest.mark.parametrize("donor_policy", ["complete", "available"])
def test_numpy_neighbor_grid_matches_python_integer_grid(donor_policy):
    x = np.arange(12, dtype=np.float32)
    train = np.column_stack((x, 10 * x + 10))
    train[[1, 5, 9], 1] = np.nan
    before = train.copy()
    target = 3 * x + 5
    queries = np.array([[2.5, np.nan], [8.5, np.nan]], dtype=np.float32)
    queries_before = queries.copy()

    def run_search(candidates):
        pipeline = make_pipeline(
            FaissImputer(donor_policy=donor_policy), Ridge()
        )
        search = GridSearchCV(
            pipeline,
            {"faissimputer__n_neighbors": candidates},
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(train, target)
        return search, search.predict(queries)

    native, native_predictions = run_search([1, 2])
    numpy_grid, numpy_predictions = run_search(np.arange(1, 3))

    assert np.isfinite(numpy_grid.cv_results_["mean_test_score"]).all()
    np.testing.assert_allclose(
        numpy_grid.cv_results_["mean_test_score"],
        native.cv_results_["mean_test_score"],
    )
    np.testing.assert_allclose(numpy_predictions, native_predictions)
    assert numpy_grid.best_params_ == native.best_params_
    assert isinstance(
        numpy_grid.best_estimator_.named_steps["faissimputer"].n_neighbors,
        np.integer,
    )
    np.testing.assert_array_equal(train, before)
    np.testing.assert_array_equal(queries, queries_before)
