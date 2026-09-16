"""Projected non-Flat indexes report actionable failures."""

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def test_projected_factory_failure_reports_context_and_preserves_state():
    train = (
        np.random.default_rng(42)
        .normal(size=(128, 4))
        .astype(np.float32)
    )
    query = train[[0]].copy()
    query[0, 3] = np.nan
    train_before = train.copy()
    query_before = query.copy()

    # Two subquantizers support four dimensions, but not three.
    model = FaissImputer(
        n_neighbors=1,
        donor_policy="complete",
        index_factory="PQ2x1",
    ).fit(train)

    with pytest.raises(RuntimeError) as caught:
        model.transform(query)

    message = str(caught.value)
    assert "index_factory='PQ2x1'" in message
    assert "3 observed features" in message
    assert "128 donors" in message
    assert "index_factory='Flat'" in message
    assert isinstance(caught.value.__cause__, RuntimeError)

    check_is_fitted(model)
    np.testing.assert_array_equal(model.donors_, train_before)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compatible_non_flat_factory_handles_different_query_masks(dtype):
    train = (
        np.random.default_rng(17)
        .normal(size=(128, 4))
        .astype(dtype)
    )
    expected = train[[3, 7]].copy()
    query = expected.copy()
    query[0, 3] = np.nan
    query[1, [1, 3]] = np.nan
    train_before = train.copy()
    query_before = query.copy()

    # One IVF list searches all donors in each projected space.
    model = FaissImputer(
        n_neighbors=1,
        donor_policy="complete",
        metric="l2",
        index_factory="IVF1,Flat",
    ).fit(train)

    result = model.transform(query)

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)