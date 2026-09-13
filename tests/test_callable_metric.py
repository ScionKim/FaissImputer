"""Callable distances across donor policies and output options."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from faiss_imputer import FaissImputer


@pytest.fixture(params=["complete", "available"])
def policy(request):
    return request.param


def nan_l1(x, y, *, missing_values):
    assert np.isnan(missing_values)
    shared = ~np.isnan(x) & ~np.isnan(y)
    if not shared.any():
        return np.nan
    return np.abs(
        x[shared].astype(np.float64)
        - y[shared].astype(np.float64)
    ).sum()


def distance_weights(distances):
    return distances.copy()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_callable_controls_neighbor_selection_and_preserves_dtype(policy, dtype):
    train = np.array([[0, 3, 10], [2, 2, 20]], dtype=dtype)
    query = np.array([[0, 0, np.nan]], dtype=dtype)

    def metric(x, y, *, missing_values):
        assert x.dtype == y.dtype == np.dtype(dtype)
        assert np.isnan(x[-1])
        assert np.isfinite(y[-1])
        return nan_l1(x, y, missing_values=missing_values)

    result = FaissImputer(
        n_neighbors=1, donor_policy=policy, metric=metric
    ).fit(train).transform(query)

    # L1 selects the first donor; L2 would select the second.
    assert_array_equal(result, [[0, 0, 10]])
    assert result.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "strategy, weights, expected",
    [
        ("mean", "uniform", 110 / 3),
        ("mean", None, 110 / 3),
        ("median", "uniform", 20),
        ("mean", "distance", 160 / 7),
        ("mean", distance_weights, 370 / 7),
    ],
)
def test_callable_aggregation_uses_actual_distances(
    policy, strategy, weights, expected
):
    train = np.array([[1, 10], [2, 20], [4, 80]], dtype=np.float64)
    model = FaissImputer(
        n_neighbors=3,
        donor_policy=policy,
        metric=nan_l1,
        strategy=strategy,
        weights=weights,
    ).fit(train)

    result = model.transform([[0, np.nan]])

    assert_allclose(result, [[0, expected]], rtol=1e-14, atol=0)


def test_callable_respects_donor_policy(policy):
    train = np.array(
        [[0, 10, np.nan], [1, np.nan, 20], [2, 30, 40]],
        dtype=np.float64,
    )
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, metric=nan_l1
    ).fit(train)

    result = model.transform([[0, np.nan, np.nan]])

    expected = (
        [[0, 30, 40]]
        if policy == "complete"
        else [[0, 10, 20]]
    )
    assert_array_equal(result, expected)


def test_nan_distances_allow_fewer_neighbors_and_statistics_fallback(policy):
    def metric(x, y, *, missing_values):
        assert not np.isnan(x).all()
        if x[0] == 8 or y[0] == 2:
            return np.nan
        return nan_l1(x, y, missing_values=missing_values)

    train = np.array([[0, 10], [1, 20], [2, 90]], dtype=np.float64)
    query = np.array(
        [[0, np.nan], [8, np.nan], [np.nan, np.nan]],
        dtype=np.float64,
    )
    model = FaissImputer(
        n_neighbors=3, donor_policy=policy, metric=metric
    ).fit(train)

    result = model.transform(query)

    assert_array_equal(result, [[0, 15], [8, 40], [1, 40]])


@pytest.mark.parametrize("reverse, expected", [(False, 20), (True, 40)])
def test_callable_distance_ties_follow_training_order(policy, reverse, expected):
    train = np.array([[1, 10], [-1, 30], [1, 50]], dtype=np.float64)
    if reverse:
        train = train[::-1].copy()

    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, metric=nan_l1
    ).fit(train)

    assert_array_equal(model.transform([[0, np.nan]]), [[0, expected]])


def test_zero_distances_exclude_nonzero_distance_donors(policy):
    train = np.array([[1, 10], [1, 30], [3, 100]], dtype=np.float64)
    model = FaissImputer(
        n_neighbors=3,
        donor_policy=policy,
        metric=nan_l1,
        weights="distance",
    ).fit(train)

    assert_array_equal(model.transform([[1, np.nan]]), [[1, 20]])


@pytest.mark.parametrize("scale", [1e-320, 1e-200, 1e200])
def test_callable_distances_do_not_require_squaring_or_reciprocals(policy, scale):
    train = np.array(
        [[scale, 10], [2 * scale, 20]],
        dtype=np.float64,
    )
    model = FaissImputer(
        n_neighbors=2,
        donor_policy=policy,
        metric=nan_l1,
        weights="distance",
    ).fit(train)

    result = model.transform([[0, np.nan]])

    assert result.dtype == np.float64
    assert_allclose(result, [[0, 40 / 3]], rtol=1e-14, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("keep", [False, True])
def test_callable_receives_normalized_schema_and_supports_pandas(
    policy, dtype, keep
):
    pd = pytest.importorskip("pandas")
    columns = ["age", "empty", "score"]
    train = pd.DataFrame(
        np.array(
            [[0, -1, 10], [2, -1, 20], [4, -1, -1]],
            dtype=dtype,
        ),
        columns=columns,
    )
    query = pd.DataFrame(
        np.array([[0.5, 99, -1], [-1, -1, 15]], dtype=dtype),
        columns=columns,
        index=["a", "b"],
    )
    train_before = train.copy(deep=True)
    query_before = query.copy(deep=True)

    def metric(x, y, *, missing_values):
        assert x.shape == y.shape == (3,)
        assert x.dtype == y.dtype == np.dtype(dtype)
        assert np.isnan(x[1]) and np.isnan(y[1])
        assert not (x == -1).any()
        assert not (y == -1).any()
        return nan_l1(x, y, missing_values=missing_values)

    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        metric=metric,
        missing_values=-1,
        keep_empty_features=keep,
        add_indicator=True,
    ).set_output(transform="pandas").fit(train)

    result = model.transform(query)

    if keep:
        values = [[0.5, 0, 10, 0, 1], [0, 0, 15, 1, 0]]
        names = columns.copy()
    else:
        values = [[0.5, 10, 0, 1], [0, 15, 1, 0]]
        names = ["age", "score"]
    names += ["missingindicator_empty", "missingindicator_score"]

    expected = pd.DataFrame(
        np.array(values, dtype=dtype),
        columns=names,
        index=query.index,
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(query, query_before)
    assert_array_equal(model.get_feature_names_out(), names)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("add_indicator", [False, True])
def test_callback_mutation_is_isolated_with_copy_false(
    policy, order, add_indicator
):
    def metric(x, y, *, missing_values):
        distance = nan_l1(x, y, missing_values=missing_values)
        x[:] = 123
        y[:] = 456
        return distance

    train = np.array(
        [[0, 10], [2, 20], [4, np.nan]], dtype=np.float64
    )
    train_before = train.copy()
    query = np.array(
        [[0.25, np.nan], [1.75, np.nan]],
        dtype=np.float64,
        order=order,
    )
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        metric=metric,
        copy=False,
        add_indicator=add_indicator,
    ).fit(train)
    donors_before = model.donors_.copy()

    result = model.transform(query)

    expected = np.array([[0.25, 10], [1.75, 20]], dtype=np.float64)
    assert_array_equal(query, expected)
    assert_array_equal(train, train_before)
    assert_array_equal(model.donors_, donors_before)
    assert result.dtype == np.float64

    if add_indicator:
        assert_array_equal(result[:, :2], expected)
        assert_array_equal(result[:, 2], [1, 1])
        assert not np.shares_memory(result, query)
    else:
        assert result is query


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(True, id="boolean"),
        pytest.param([1.0], id="vector"),
        pytest.param("1", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_invalid_callable_distance_is_rejected(policy, value):
    def metric(x, y, *, missing_values):
        return value

    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, metric=metric
    ).fit([[0, 10], [2, 20]])

    with pytest.raises(ValueError, match="metric"):
        model.transform([[0.25, np.nan]])


def test_callback_exception_propagates_and_model_remains_usable(policy):
    fail = True

    def metric(x, y, *, missing_values):
        if fail:
            raise RuntimeError("custom distance failed")
        return nan_l1(x, y, missing_values=missing_values)

    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, metric=metric
    ).fit([[0, 10], [2, 20]])
    query = np.array([[0.25, np.nan]], dtype=np.float64)
    before = query.copy()

    with pytest.raises(RuntimeError, match="custom distance failed"):
        model.transform(query)

    assert_array_equal(query, before)
    fail = False
    assert_array_equal(model.transform(query), [[0.25, 10]])


@pytest.mark.parametrize("empty", [False, True])
def test_non_flat_callable_refit_clears_fitted_state(policy, empty):
    train = np.array([[0, 10], [2, 20]], dtype=np.float64)
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, metric=nan_l1
    ).fit(train)
    model.set_params(index_factory="IVF2,Flat")

    invalid_train = np.full_like(train, np.nan) if empty else train
    with pytest.raises(ValueError, match="index_factory='Flat'"):
        model.fit(invalid_train)

    for name in ("donors_", "metric_callable_", "index_", "available_index_"):
        assert not hasattr(model, name)
    with pytest.raises(NotFittedError):
        model.transform([[0, np.nan]])
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()


def test_cloning_and_refitting_between_builtin_and_callable_metrics(policy):
    train = np.array([[0, 3, 10], [2, 2, 20]], dtype=np.float64)
    query = np.array([[0, 0, np.nan]], dtype=np.float64)
    model = clone(
        FaissImputer(
            n_neighbors=1, donor_policy=policy, metric=nan_l1
        )
    )
    assert model.get_params()["metric"] is nan_l1

    model.fit(train)
    assert_array_equal(model.transform(query), [[0, 0, 10]])

    model.set_params(metric="l2").fit(train)
    assert_array_equal(model.transform(query), [[0, 0, 20]])

    model.set_params(metric=nan_l1).fit(train)
    assert_array_equal(model.transform(query), [[0, 0, 10]])


@pytest.mark.parametrize("keep", [False, True])
def test_all_empty_training_data_does_not_call_metric(policy, keep):
    def metric(x, y, *, missing_values):
        raise AssertionError("Empty training data must not call the metric")

    model = FaissImputer(
        donor_policy=policy,
        metric=metric,
        keep_empty_features=keep,
        add_indicator=True,
    ).fit(np.full((2, 2), np.nan, dtype=np.float64))

    result = model.transform([[1, np.nan]])

    expected = [[0, 0, 0, 1]] if keep else [[0, 1]]
    assert_array_equal(result, expected)
    assert result.dtype == np.float64


@pytest.mark.parametrize("weights", ["distance", distance_weights])
def test_callable_metric_keeps_nonuniform_median_restriction(policy, weights):
    with pytest.raises(ValueError, match="strategy='mean'"):
        FaissImputer(
            n_neighbors=1,
            donor_policy=policy,
            metric=nan_l1,
            strategy="median",
            weights=weights,
        ).fit([[0, 10], [2, 20]])