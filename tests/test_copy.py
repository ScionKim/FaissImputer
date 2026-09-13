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
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "strategy,weights",
    [
        ("mean", "uniform"),
        ("mean", "distance"),
        ("mean", shifted_inverse),
        ("median", "uniform"),
    ],
)
def test_copy_false_reuses_input_and_matches_copy_true(
    policy, order, strategy, weights
):
    train = np.array(
        [[0, 10, 100], [2, 30, 200],
         [6, 90, 600], [1, np.nan, 150]],
        dtype=np.float32,
    )
    rows = [
        [0.1, np.nan, 110],
        [np.nan, 25, np.nan],
        [np.nan, np.nan, np.nan],
        [6, 90, 600],
        [4, np.nan, np.nan],
    ]
    # More than 256 partially missing rows exercise multiple batches.
    query = np.array(
        np.tile(rows, (100, 1)), dtype=np.float32, order=order
    )
    train_before = train.copy()
    query_before = query.copy()
    params = dict(
        n_neighbors=2,
        metric="nan_euclidean",
        donor_policy=policy,
        strategy=strategy,
        weights=weights,
    )

    reference = FaissImputer(copy=True, **params).fit(train)
    expected = reference.transform(query)
    assert_array_equal(query, query_before)
    assert not np.shares_memory(expected, query)

    model = FaissImputer(copy=False, **params).fit(train)
    assert_array_equal(train, train_before)
    result = model.transform(query)

    assert result is query
    assert result.dtype == np.float32
    assert_array_equal(result, expected)
    observed = ~np.isnan(query_before)
    assert_array_equal(result[observed], query_before[observed])
    assert_array_equal(train, train_before)


@pytest.mark.parametrize("policy", ["complete", "available"])
def test_indicators_use_missingness_before_input_is_modified(policy):
    train = np.array(
        [[0, 10, 100], [2, np.nan, 200],
         [4, 40, np.nan], [6, 60, 600]],
        dtype=np.float32,
    )
    query = np.array(
        [[1, np.nan, 150], [np.nan, 25, np.nan],
         [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    params = dict(
        n_neighbors=1, donor_policy=policy, add_indicator=True
    )
    expected = FaissImputer(copy=True, **params).fit(train).transform(query)
    model = FaissImputer(copy=False, **params).fit(train)

    result = model.transform(query)

    assert_array_equal(result, expected)
    assert_array_equal(result[:, -2:], [[1, 0], [0, 1], [1, 1]])
    assert_array_equal(query, expected[:, :3])
    assert not np.shares_memory(result, query)


@pytest.mark.parametrize("policy", ["complete", "available"])
@pytest.mark.parametrize("kind", ["readonly", "float64", "numeric-marker"])
def test_copy_false_handles_inputs_requiring_conversion_or_copy(policy, kind):
    query = np.array(
        [[0.25, np.nan], [1.75, np.nan]], dtype=np.float32
    )
    marker = np.nan
    if kind == "readonly":
        query.setflags(write=False)
    elif kind == "float64":
        query = query.astype(np.float64)
    else:
        marker = -1
        query[np.isnan(query)] = marker

    before = query.copy()
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        missing_values=marker,
        copy=False,
    ).fit([[0, 10], [2, 20]])

    result = model.transform(query)

    assert_array_equal(result, [[0.25, 10], [1.75, 20]])
    assert result.dtype == np.float32
    assert result.flags.writeable
    if kind != "numeric-marker":
        assert_array_equal(query, before)
        assert not np.shares_memory(result, query)


@pytest.mark.parametrize("policy", ["complete", "available"])
def test_overlapping_view_is_copied_to_preserve_independent_results(policy):
    storage = np.array([0, np.nan, 2], dtype=np.float32)
    before = storage.copy()
    query = np.lib.stride_tricks.as_strided(
        storage,
        shape=(2, 2),
        strides=(storage.itemsize, storage.itemsize),
        writeable=True,
    )
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, copy=False
    ).fit([[0, 10], [2, 20]])

    result = model.transform(query)

    assert_array_equal(result, [[0, 10], [0, 2]])
    assert_array_equal(storage, before)
    assert not np.shares_memory(result, storage)


@pytest.mark.parametrize("policy", ["complete", "available"])
@pytest.mark.parametrize("keep", [False, True])
def test_copy_false_preserves_empty_feature_and_indicator_output(policy, keep):
    query = np.array(
        [[0.5, 99, np.nan], [np.nan, np.nan, 15]],
        dtype=np.float32,
    )
    before = query.copy()
    query.setflags(write=False)
    training_cases = [
        np.array([[0, np.nan, 10], [2, np.nan, 20]], dtype=np.float32),
        np.full((2, 3), np.nan, dtype=np.float32),
    ]
    params = dict(
        n_neighbors=1,
        donor_policy=policy,
        keep_empty_features=keep,
        add_indicator=True,
    )

    for train in training_cases:
        reference = FaissImputer(copy=True, **params).fit(train)
        model = FaissImputer(copy=False, **params).fit(train)

        assert_array_equal(model.transform(query), reference.transform(query))
        assert_array_equal(
            model.get_feature_names_out(),
            reference.get_feature_names_out(),
        )
        assert_array_equal(query, before)


@pytest.mark.parametrize("invalid", [None, 0, 1, "False", np.array(False)])
def test_invalid_copy_parameter_clears_failed_refit_state(invalid):
    train = [[0, 10], [2, np.nan]]
    model = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
        add_indicator=True,
    ).fit(train)
    model.set_params(copy=invalid)

    with pytest.raises(ValueError, match="copy must be a boolean"):
        model.fit(train)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "indicator_")
    assert not hasattr(model, "available_index_")


def test_copy_parameter_defaults_cloning_and_numpy_boolean():
    assert FaissImputer().copy is True
    template = FaissImputer(n_neighbors=1, copy=np.bool_(False))
    model = clone(template).fit([[0, 10], [2, 20]])
    query = np.array([[0.25, np.nan]], dtype=np.float32)

    assert isinstance(model.copy, np.bool_)
    assert model.transform(query) is query
    assert_array_equal(query, [[0.25, 10]])

    model.set_params(copy=True)
    fresh_query = np.array([[0.25, np.nan]], dtype=np.float32)
    before = fresh_query.copy()
    result = model.transform(fresh_query)

    assert_array_equal(fresh_query, before)
    assert not np.shares_memory(result, fresh_query)