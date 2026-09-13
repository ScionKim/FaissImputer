import faiss
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)


@pytest.mark.parametrize("policy", ["complete", "available"])
@pytest.mark.parametrize(
    "strategy,weights",
    [
        ("mean", "uniform"),
        ("mean", "distance"),
        ("mean", shifted_inverse),
        ("median", "uniform"),
    ],
)
def test_alias_matches_l2_and_preserves_parameters(policy, strategy, weights):
    train = np.array(
        [[0, 10, np.nan], [2, np.nan, 200],
         [4, 40, 400], [8, 80, 800]],
        dtype=np.float32,
    )
    query = np.array(
        [[1, np.nan, 150], [np.nan, 30, np.nan],
         [np.nan, np.nan, np.nan], [4, 40, 400]],
        dtype=np.float32,
    )
    train_before = train.copy()
    query_before = query.copy()
    params = dict(
        n_neighbors=2,
        donor_policy=policy,
        strategy=strategy,
        weights=weights,
        add_indicator=True,
    )
    template = FaissImputer(metric="nan_euclidean", **params)
    model = clone(template).fit(train)
    reference = FaissImputer(metric="l2", **params).fit(train)

    result = model.transform(query)

    assert_array_equal(result, reference.transform(query))
    assert_array_equal(
        model.get_feature_names_out(),
        reference.get_feature_names_out(),
    )
    assert model.get_params()["metric"] == "nan_euclidean"
    assert clone(model).metric == "nan_euclidean"
    assert not hasattr(template, "n_features_in_")
    assert result.dtype == np.float32
    assert not np.shares_memory(result, query)
    assert_array_equal(train, train_before)
    assert_array_equal(query, query_before)


@pytest.mark.parametrize("policy", ["complete", "available"])
@pytest.mark.parametrize("keep", [False, True])
def test_alias_supports_numeric_markers_empty_features_and_pandas(policy, keep):
    pd = pytest.importorskip("pandas")
    columns = ["age", "empty", "score"]
    train = pd.DataFrame(
        [[0, -1, 10], [2, -1, 20]],
        columns=columns,
    )
    query = pd.DataFrame(
        [[0.5, 99, -1], [-1, -1, 15]],
        columns=columns,
        index=["a", "b"],
    )
    model = FaissImputer(
        n_neighbors=2,
        metric="nan_euclidean",
        donor_policy=policy,
        missing_values=-1,
        keep_empty_features=keep,
        add_indicator=True,
    ).set_output(transform="pandas").fit(train)

    result = model.transform(query)

    if keep:
        expected = [[0.5, 0, 15, 0], [1, 0, 15, 1]]
        names = columns + ["missingindicator_empty"]
    else:
        expected = [[0.5, 15, 0], [1, 15, 1]]
        names = ["age", "score", "missingindicator_empty"]

    assert_array_equal(result.to_numpy(), expected)
    assert result.columns.tolist() == names
    assert result.index.tolist() == ["a", "b"]
    assert all(dtype == np.float64 for dtype in result.dtypes)


def test_alias_uses_l2_for_nonflat_index_and_quantizer():
    train = (
        np.random.default_rng(17)
        .normal(size=(128, 4))
        .astype(np.float32)
    )
    model = FaissImputer(
        n_neighbors=1,
        metric="nan_euclidean",
        index_factory="IVF2,Flat",
    ).fit(train)

    assert model.index_.is_trained
    assert model.index_.ntotal == len(train)
    assert model.index_.metric_type == faiss.METRIC_L2
    quantizer = faiss.downcast_index(model.index_.quantizer)
    assert quantizer.metric_type == faiss.METRIC_L2


@pytest.mark.parametrize(
    "policy,factory",
    [
        ("complete", "Flat"),
        ("complete", "IVF2,Flat"),
        ("available", "Flat"),
    ],
)
def test_alias_handles_entirely_empty_training_data(
    monkeypatch, policy, factory
):
    recorded_metrics = []
    original_factory = faiss.index_factory

    def tracked_factory(dimension, description, metric):
        recorded_metrics.append(metric)
        return original_factory(dimension, description, metric)

    monkeypatch.setattr(faiss, "index_factory", tracked_factory)
    model = FaissImputer(
        metric="nan_euclidean",
        donor_policy=policy,
        index_factory=factory,
        keep_empty_features=True,
        add_indicator=True,
    ).fit([[np.nan, np.nan]])

    assert_array_equal(
        model.transform([[np.nan, 7]]),
        [[0, 0, 1, 0]],
    )
    expected_metrics = [faiss.METRIC_L2] if factory != "Flat" else []
    assert recorded_metrics == expected_metrics


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"metric": "unknown"}, "metric"),
        ({"strategy": "median", "weights": "distance"}, "strategy"),
        (
            {"donor_policy": "available", "index_factory": "IVF2,Flat"},
            "donor_policy",
        ),
    ],
)
def test_alias_retains_validation_and_failed_refit_cleanup(parameters, message):
    train = [[0, 10], [2, np.nan]]
    model = FaissImputer(
        n_neighbors=1,
        metric="nan_euclidean",
        add_indicator=True,
    ).fit(train)
    model.set_params(**parameters)

    with pytest.raises(ValueError, match=message):
        model.fit(train)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()
    assert not hasattr(model, "indicator_")