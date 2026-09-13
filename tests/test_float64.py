import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=["complete", "available"])
def policy(request):
    return request.param


def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_and_output_precision(policy, dtype):
    train = np.array([[0, 2**24 + 1], [2, 2**24 + 3]], dtype=dtype)
    query = np.array([[np.nextafter(0.25, 1.0), np.nan]], dtype=dtype)
    train_before, query_before = train.copy(), query.copy()
    model = FaissImputer(n_neighbors=1, donor_policy=policy).fit(train)

    result = model.transform(query)

    assert model.donors_.dtype == dtype
    assert model.statistics_.dtype == dtype
    assert result.dtype == dtype
    assert_array_equal(result, [[query[0, 0], train[0, 1]]])
    assert_array_equal(train, train_before)
    assert_array_equal(query, query_before)
    assert not np.shares_memory(result, query)


@pytest.mark.parametrize("train_dtype,query_dtype", [
    (np.float32, np.float64),
    (np.float64, np.float32),
])
def test_output_dtype_follows_query(policy, train_dtype, query_dtype):
    train = np.array([[0, 2**24 + 1], [2, 2**24 + 3]], dtype=train_dtype)
    query = np.array([[0.25, np.nan]], dtype=query_dtype)
    model = FaissImputer(n_neighbors=1, donor_policy=policy).fit(train)

    result = model.transform(query)

    assert result.dtype == query_dtype
    expected = np.array([[0.25, train[0, 1]]], dtype=query_dtype)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("weights", ["distance", shifted_inverse])
def test_weighted_output_uses_float64_values_and_distances(policy, weights):
    train = np.array(
        [[0, 10 + 2**-30], [2, 20 + 2**-29]], dtype=np.float64,
    )
    position = 0.123456789012345
    query = np.array([[position, np.nan]], dtype=np.float64)
    distances = np.sqrt(2 * (train[:, 0] - position) ** 2)
    reference_weights = (
        1.0 / distances if weights == "distance"
        else shifted_inverse(distances)
    )
    expected = (
        np.sum(train[:, 1] * reference_weights) / reference_weights.sum()
    )
    model = FaissImputer(
        n_neighbors=2,
        donor_policy=policy,
        metric="nan_euclidean",
        weights=weights,
    ).fit(train)

    result = model.transform(query)

    assert result.dtype == np.float64
    assert result[0, 0] == position
    assert_allclose(result[0, 1], expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("strategy,weights", [
    ("mean", "uniform"),
    ("median", "uniform"),
    ("mean", "distance"),
    ("mean", shifted_inverse),
])
def test_large_representable_aggregates_remain_finite(
    policy, strategy, weights,
):
    train = np.array([[0, 1e308], [2, 1e308]], dtype=np.float64)
    query = np.array([[1, np.nan], [np.nan, np.nan]], dtype=np.float64)
    model = FaissImputer(
        n_neighbors=2,
        donor_policy=policy,
        strategy=strategy,
        weights=weights,
    ).fit(train)

    result = model.transform(query)

    assert result.dtype == np.float64
    assert_array_equal(result, [[1, 1e308], [1, 1e308]])


def test_numeric_marker_preserves_distinct_float64_values(policy):
    marker = 2**24
    train = np.array(
        [[0, marker], [1, marker + 1], [2, marker + 3]],
        dtype=np.float64,
    )
    query = np.array(
        [[1, marker], [1.5, marker + 1]], dtype=np.float64,
    )
    before = query.copy()
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        missing_values=marker,
        add_indicator=True,
    ).fit(train)

    result = model.transform(query)

    assert result.dtype == np.float64
    assert_array_equal(
        result, [[1, marker + 1, 1], [1.5, marker + 1, 0]],
    )
    assert_array_equal(query, before)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("add_indicator", [False, True])
def test_copy_false_reuses_float64_and_captures_indicators(
    policy, order, add_indicator,
):
    low, high = 10 + 2**-30, 20 + 2**-30
    train = np.array(
        [[0, low], [2, high], [4, np.nan]], dtype=np.float64,
    )
    query = np.array(
        [[0.25, np.nan], [1.75, np.nan]],
        dtype=np.float64,
        order=order,
    )
    before = train.copy()
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        copy=False,
        add_indicator=add_indicator,
    ).fit(train)

    result = model.transform(query)

    assert result.dtype == np.float64
    assert_array_equal(query, [[0.25, low], [1.75, high]])
    assert_array_equal(train, before)
    if add_indicator:
        assert_array_equal(result, [[0.25, low, 1], [1.75, high, 1]])
        assert not np.shares_memory(result, query)
    else:
        assert result is query


@pytest.mark.parametrize("keep_empty", [False, True])
def test_empty_features_and_indicators_preserve_float64(
    policy, keep_empty,
):
    low, high = 10 + 2**-30, 20 + 2**-30
    train = np.array(
        [[0, np.nan, low], [2, np.nan, high], [4, np.nan, np.nan]],
        dtype=np.float64,
    )
    query = np.array(
        [[0.25, 99, np.nan], [np.nan, np.nan, 15.125]],
        dtype=np.float64,
    )
    before = query.copy()
    query.setflags(write=False)
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        copy=False,
        keep_empty_features=keep_empty,
        add_indicator=True,
    ).fit(train)

    result = model.transform(query)

    expected = (
        [[0.25, 0, low, 0, 1], [2, 0, 15.125, 1, 0]]
        if keep_empty else [[0.25, low, 0, 1], [2, 15.125, 1, 0]]
    )
    assert result.dtype == np.float64
    assert_array_equal(result, expected)
    assert_array_equal(query, before)
    assert result.shape[1] == len(model.get_feature_names_out())


@pytest.mark.parametrize("keep_empty", [False, True])
def test_entirely_empty_training_data_preserves_float64(
    policy, keep_empty,
):
    train = np.full((2, 2), np.nan, dtype=np.float64)
    query = np.array([[1, np.nan]], dtype=np.float64)
    model = FaissImputer(
        donor_policy=policy,
        keep_empty_features=keep_empty,
        add_indicator=True,
    ).fit(train)

    result = model.transform(query)

    assert model.donors_.dtype == np.float64
    assert model.statistics_.dtype == np.float64
    assert result.dtype == np.float64
    assert_array_equal(
        result, [[0, 0, 0, 1]] if keep_empty else [[0, 1]],
    )


def test_pandas_pipeline_preserves_float64(policy):
    pd = pytest.importorskip("pandas")
    columns = ["age", "income"]
    train = pd.DataFrame(
        [[0, 10 + 2**-30], [2, 20 + 2**-30]],
        columns=columns,
        dtype=np.float64,
    )
    query = pd.DataFrame(
        [[0.25, np.nan]], columns=columns, index=["row"],
    )
    pipeline = make_pipeline(
        FaissImputer(n_neighbors=1, donor_policy=policy)
    ).set_output(transform="pandas")

    result = pipeline.fit(train).transform(query)

    expected = pd.DataFrame(
        [[0.25, 10 + 2**-30]],
        columns=columns,
        index=["row"],
        dtype=np.float64,
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_complete_search_rejects_values_outside_float32_range():
    model = FaissImputer(n_neighbors=1).fit(
        np.array([[0, 10], [2, 20]], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="FAISS search values"):
        model.transform(
            np.array([[1e100, np.nan]], dtype=np.float64)
        )


def test_non_flat_range_failure_clears_fitted_state():
    model = FaissImputer(n_neighbors=1).fit([[0, 10], [2, 20]])
    model.set_params(index_factory="IVF2,Flat")

    with pytest.raises(ValueError, match="FAISS search values"):
        model.fit(
            np.array([[0, 1e100], [2, 1e100]], dtype=np.float64)
        )

    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "donors_")
    assert not hasattr(model, "index_")


def test_available_search_repairs_overflowing_squared_norms():
    origin = 1e155
    step = np.spacing(origin)
    high = 20 + 2**-30
    train = np.array(
        [[origin + 4 * step, 10], [origin + step, high]],
        dtype=np.float64,
    )
    model = FaissImputer(
        n_neighbors=1, donor_policy="available",
    ).fit(train)

    result = model.transform(
        np.array([[origin, np.nan]], dtype=np.float64)
    )

    assert_array_equal(result, [[origin, high]])


@pytest.mark.parametrize("value", [1e200, 1e-200])
def test_available_distance_range_failure_clears_query_cache(value):
    model = FaissImputer(
        n_neighbors=1, donor_policy="available",
    ).fit(np.array([[value, 10]], dtype=np.float64))

    with pytest.raises(ValueError, match="Squared distances"):
        model.transform(
            np.array([[0, np.nan]], dtype=np.float64)
        )

    assert model.available_index_.query_ref is None
    assert model.available_index_.matrix is None
    assert model.available_index_.precise_rows == {}
    assert_array_equal(
        model.transform(
            np.array([[value, np.nan]], dtype=np.float64)
        ),
        [[value, 10]],
    )