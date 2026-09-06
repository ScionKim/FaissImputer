"""Retain Flat metadata without a redundant full-dimensional donor index."""

import faiss
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("metric, faiss_metric, expected", [
    ("l2", faiss.METRIC_L2, 300),
    ("ip", faiss.METRIC_INNER_PRODUCT, 700),
])
def test_flat_keeps_empty_metadata_index_and_imputes_with_projected_donors(
    metric, faiss_metric, expected,
):
    train = np.array([[0, 100], [3, 300], [7, 700]], dtype=np.float32)
    query = np.array([[2, np.nan]], dtype=np.float32)
    original_train, original_query = train.copy(), query.copy()
    model = FaissImputer(n_neighbors=1, metric=metric).fit(train)

    assert model.index_.ntotal == 0
    assert model.index_.is_trained
    assert model.index_.d == train.shape[1]
    assert model.index_.metric_type == faiss_metric
    np.testing.assert_array_equal(model.donors_, train)
    np.testing.assert_array_equal(model.transform(query), [[2, expected]])
    assert model.index_.ntotal == 0
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(query, original_query)


@pytest.mark.parametrize("metric, faiss_metric", [
    ("l2", faiss.METRIC_L2),
    ("ip", faiss.METRIC_INNER_PRODUCT),
])
def test_non_flat_fit_still_trains_and_stores_donors(metric, faiss_metric):
    train = np.random.default_rng(17).normal(size=(128, 4)).astype(np.float32)
    model = FaissImputer(
        n_neighbors=1, metric=metric, index_factory="IVF2,Flat",
    ).fit(train)

    assert model.index_.is_trained
    assert model.index_.ntotal == len(train)
    quantizer = faiss.downcast_index(model.index_.quantizer)
    assert quantizer.ntotal == 2
    assert quantizer.metric_type == faiss_metric


@pytest.mark.parametrize("factory", [
    pytest.param("not-a-factory", id="invalid-description"),
    pytest.param("IDMap,Flat", id="requires-explicit-ids"),
    pytest.param("IVF8,Flat", id="insufficient-training-donors"),
])
def test_factory_failure_during_refit_clears_fitted_state(factory):
    train = np.array([[0, 10], [2, 20], [4, 40]], dtype=np.float32)
    original = train.copy()
    model = FaissImputer(n_neighbors=1).fit(train)
    model.set_params(index_factory=factory)

    with pytest.raises(RuntimeError):
        model.fit(train)
    with pytest.raises(NotFittedError):
        model.transform([[0.25, np.nan]])
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()
    assert not hasattr(model, "index_")
    assert not hasattr(model, "donors_")
    np.testing.assert_array_equal(train, original)

    model.set_params(index_factory="Flat").fit(train)
    np.testing.assert_array_equal(model.transform([[0.25, np.nan]]), [[0.25, 10]])
