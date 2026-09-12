"""Numeric missing markers must be identified before float32 conversion."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from faiss_imputer import FaissImputer


POLICIES = ["complete", "available"]


def marker_data(marker):
    train = np.array(
        [[np.nan, 2, 10], [np.nan, 6, 30], [np.nan, 12, np.nan],
         [np.nan, np.nan, np.nan]],
        dtype=np.float64,
    )
    query = np.array(
        [[77, 3, np.nan], [np.nan, np.nan, 42],
         [np.nan, np.nan, np.nan], [1, 8, 9]],
        dtype=np.float64,
    )
    return train, query, np.where(np.isnan(train), marker, train), np.where(
        np.isnan(query), marker, query
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("marker", [-1, 0, np.float64(9999)])
@pytest.mark.parametrize("keep", [False, True])
def test_numeric_marker_matches_nan_baseline_and_preserves_inputs(
    policy, marker, keep
):
    train_nan, query_nan, train, query = marker_data(marker)
    before_train, before_query = train.copy(), query.copy()
    train.setflags(write=False)
    query.setflags(write=False)
    params = dict(
        n_neighbors=1, donor_policy=policy,
        add_indicator=True, keep_empty_features=keep,
    )
    reference = FaissImputer(**params).fit(train_nan)
    model = FaissImputer(**params, missing_values=marker).fit(train)

    actual = model.transform(query)
    assert_allclose(actual, reference.transform(query_nan), rtol=2e-6)
    assert actual.dtype == np.float32
    assert not np.shares_memory(actual, query)
    assert_array_equal(model.indicator_.features_, [0, 1, 2])
    assert_array_equal(model.get_feature_names_out(), reference.get_feature_names_out())
    assert_array_equal(model.valid_features_, [False, True, True])
    assert_array_equal(train, before_train)
    assert_array_equal(query, before_query)
    assert_array_equal(model.transform(query[::-1]), actual[::-1])
    assert_array_equal(
        np.vstack([model.transform(row[None, :]) for row in query]), actual
    )


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_available_numeric_marker_matches_knn(weights):
    train = np.array([[2, 10, 100], [4, -1, 200], [6, 50, 300], [-1, 70, 400]])
    query = np.array([[3, -1, 150], [-1, 30, 250], [-1, -1, -1]])
    params = dict(n_neighbors=2, weights=weights, missing_values=-1, add_indicator=True)
    reference = KNNImputer(**params).fit(train)
    model = FaissImputer(**params, donor_policy="available").fit(train)

    assert_allclose(model.transform(query), reference.transform(query), rtol=3e-6)
    assert_array_equal(model.get_feature_names_out(), reference.get_feature_names_out())


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "dtype,marker,observed",
    [
        (np.float64, 16777217.0, 16777216.0),
        (np.int64, 2**53 + 1, 2**53),
        (np.uint64, np.uint64(2**64 - 1), 2**64 - 2),
        (np.int64, float(2**53), 2**53 + 1),
    ],
    ids=["float64-before-float32", "int64-before-float64", "uint64-limit", "float-marker-integer-input"],
)
def test_marker_equality_precedes_lossy_conversion(policy, dtype, marker, observed):
    train = np.array([[observed, 10], [marker, 20], [observed, 30]], dtype=dtype)
    query = np.array([[marker, 25], [observed, marker]], dtype=dtype)
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, missing_values=marker,
        add_indicator=True,
    ).fit(train)

    assert_array_equal(model.indicator_.features_, [0])
    assert_array_equal(model.valid_features_, [True, True])
    assert_array_equal(
        model.transform(query),
        np.array([[observed, 25, 1], [observed, 20, 0]], dtype=np.float32),
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_missing_marker_may_exceed_float32_range(policy):
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, missing_values=1e100,
        add_indicator=True,
    ).fit([[2, 10], [4, 1e100], [6, 30]])

    assert_array_equal(model.transform([[2.1, 1e100]]), np.array([[2.1, 10, 1]], dtype=np.float32))
    with pytest.raises(ValueError):
        model.transform([[1e101, 10]])
    with pytest.raises(ValueError):
        model.fit([[2, 10], [1e101, 20]])
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
def test_all_marker_features_keep_fixed_shape_and_original_indicators(policy, keep):
    model = FaissImputer(
        missing_values=-1, n_neighbors=10, donor_policy=policy,
        keep_empty_features=keep, add_indicator=True,
    ).fit([[-1, -1], [-1, -1]])
    result = model.transform([[-1, 9], [8, -1]])
    indicators = np.array([[1, 0], [0, 1]], dtype=np.float32)
    expected = np.column_stack([np.zeros((2, 2)), indicators]) if keep else indicators

    assert_array_equal(result, expected)
    assert result.dtype == np.float32
    assert_array_equal(
        model.get_feature_names_out(),
        (["x0", "x1"] if keep else []) + ["missingindicator_x0", "missingindicator_x1"],
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_dataframe_schema_and_pandas_output_use_normalized_mask(policy):
    pd = pytest.importorskip("pandas")
    columns = ["empty", "age", "score"]
    train = pd.DataFrame([[-1, 2, 10], [-1, 6, 30], [-1, 12, -1]], columns=columns)
    query = pd.DataFrame([[99, 3, -1], [-1, 7, 42]], columns=columns, index=["a", "b"])
    original_train, original_query = train.copy(deep=True), query.copy(deep=True)
    model = FaissImputer(
        missing_values=np.int64(-1), n_neighbors=1,
        donor_policy=policy, add_indicator=True,
    ).set_output(transform="pandas").fit(train)
    result = model.transform(query)

    assert result.columns.tolist() == ["age", "score", "missingindicator_empty", "missingindicator_score"]
    assert result.index.tolist() == ["a", "b"]
    assert_array_equal(result.to_numpy(), [[3, 10, 0, 1], [7, 42, 1, 0]])
    assert all(dtype == np.float32 for dtype in result.dtypes)
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(query, original_query)
    with pytest.raises(ValueError):
        model.transform(query[columns[::-1]])


def test_default_nan_and_estimator_parameter_apis():
    assert np.isnan(FaissImputer().get_params()["missing_values"])
    template = FaissImputer(n_neighbors=1, missing_values=np.int64(-1))
    model = clone(template)
    assert model.get_params()["missing_values"] == -1
    model.set_params(missing_values=0)
    train = [[2, 10], [4, 0], [6, 30]]
    pipeline = make_pipeline(model, StandardScaler()).fit(train)
    expected = make_pipeline(FaissImputer(n_neighbors=1), StandardScaler()).fit(
        [[2, 10], [4, np.nan], [6, 30]]
    )

    assert_allclose(pipeline.transform([[2.1, 0]]), expected.transform([[2.1, np.nan]]))
    assert_array_equal(pipeline.get_feature_names_out(), ["x0", "x1"])
    assert not hasattr(template, "n_features_in_")


@pytest.mark.parametrize(
    "marker",
    [True, np.bool_(False), None, "NaN", complex(-1), [0], np.array(-1),
     np.inf, -np.inf],
    ids=["bool", "numpy-bool", "none", "string", "complex", "list", "array", "inf", "negative-inf"],
)
def test_invalid_marker_refit_clears_fitted_state(marker):
    model = FaissImputer(n_neighbors=1, add_indicator=True).fit([[2, 10], [4, np.nan]])
    model.set_params(missing_values=marker)
    with pytest.raises(ValueError):
        model.fit([[2, 10], [4, 20]])
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()
    assert not hasattr(model, "indicator_")
    assert not hasattr(model, "valid_features_")


@pytest.mark.parametrize("policy", POLICIES)
def test_numeric_marker_rejects_nan_and_infinity_without_changing_fitted_model(policy):
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, missing_values=-1,
        add_indicator=True,
    ).fit([[2, 10], [4, -1], [6, 30]])
    expected = model.transform([[2.1, -1]])
    for bad in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            model.transform([[bad, -1]])
        assert_array_equal(model.transform([[2.1, -1]]), expected)

    with pytest.raises(ValueError):
        model.fit([[2, 10], [4, np.nan]])
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, "indicator_")
    assert not hasattr(model, "valid_features_")


def test_marker_absent_during_fit_does_not_create_new_indicator():
    model = FaissImputer(n_neighbors=1, missing_values=-1, add_indicator=True).fit(
        [[2, 10], [6, 30]]
    )
    assert_array_equal(model.indicator_.features_, [])
    assert_array_equal(model.transform([[2.1, -1]]), np.array([[2.1, 10]], dtype=np.float32))
    assert_array_equal(model.get_feature_names_out(), ["x0", "x1"])


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("container", ["list", "object-array", "mixed-dataframe"])
@pytest.mark.parametrize("marker", [float(2**53), np.longdouble(2**53)])
def test_mixed_input_preserves_exact_large_numpy_integer_values(policy, container, marker):
    observed = np.int64(2**53 + 1)
    train = [[observed, 10.0], [np.int64(2**53), 20.0]]
    query = [[observed, 10.0], [np.int64(2**53), 10.0]]
    if container == "object-array":
        train, query = np.array(train, dtype=object), np.array(query, dtype=object)
    elif container == "mixed-dataframe":
        pd = pytest.importorskip("pandas")
        train, query = pd.DataFrame(train), pd.DataFrame(query)

    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, missing_values=marker,
        add_indicator=True,
    ).fit(train)
    assert_array_equal(model.valid_features_, [True, True])
    assert_array_equal(model.indicator_.features_, [0])
    assert_array_equal(
        model.transform(query),
        np.array([[observed, 10, 0], [observed, 10, 1]], dtype=np.float32),
    )


@pytest.mark.parametrize("dtype,marker", [(np.float32, 0.1), (np.float64, 2**53 + 1)])
def test_marker_unrepresentable_in_input_dtype_does_not_match_rounded_value(dtype, marker):
    train = np.array([[marker, 10], [marker, 20]], dtype=dtype)
    model = FaissImputer(
        n_neighbors=1, missing_values=marker, add_indicator=True,
    ).fit(train)
    assert_array_equal(model.valid_features_, [True, True])
    assert_array_equal(model.indicator_.features_, [])
    assert_array_equal(model.transform(train), train.astype(np.float32))
