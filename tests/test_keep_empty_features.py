import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from faiss_imputer import FaissImputer


POLICIES = ["complete", "available"]


def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)


def unexpected_weights(distances):
    raise AssertionError("No neighbor weights should be needed")


def mixed_data():
    train = np.array(
        [
            [np.nan, 0, 10, np.nan],
            [np.nan, 4, 30, np.nan],
            [np.nan, 10, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [77, 1, np.nan, 88],
            [np.nan, np.nan, 42, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [1, 8, 9, np.nan],
        ],
        dtype=np.float32,
    )
    return train, queries


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("indicator", [False, True])
def test_empty_columns_have_fixed_output_and_original_indicators(
    policy, keep, indicator
):
    train, queries = mixed_data()
    original_train = train.copy()
    original_queries = queries.copy()
    train.setflags(write=False)
    queries.setflags(write=False)
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        keep_empty_features=keep,
        add_indicator=indicator,
    ).fit(train)

    # Both policies learn fallback statistics from all training rows before
    # donor filtering, so the partial row's value 10 contributes in each.
    fallback_x1 = 14 / 3
    expected = np.array(
        [[1, 10], [4, 42], [fallback_x1, 20], [8, 9]],
        dtype=np.float32,
    )
    names = ["x1", "x2"]
    if keep:
        expected = np.column_stack(
            [np.zeros(4), expected, np.zeros(4)]
        )
        names = ["x0", "x1", "x2", "x3"]
    if indicator:
        expected = np.column_stack([expected, np.isnan(original_queries)])
        names += [f"missingindicator_x{i}" for i in range(4)]
        assert_array_equal(model.indicator_.features_, [0, 1, 2, 3])

    actual = model.transform(queries)
    assert actual.dtype == np.float32
    assert_allclose(actual, expected, rtol=2e-6)
    assert_array_equal(model.get_feature_names_out(), names)
    assert model.n_features_in_ == 4
    assert_array_equal(train, original_train)
    assert_array_equal(queries, original_queries)
    assert_array_equal(model.transform(queries[::-1]), actual[::-1])
    assert_array_equal(
        np.vstack([model.transform(row[None, :]) for row in queries]), actual
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_default_drops_fit_empty_columns_in_original_order(policy):
    model = FaissImputer(n_neighbors=1, donor_policy=policy)
    assert model.get_params()["keep_empty_features"] is False
    model.fit([[np.nan, 0, np.nan, 10], [np.nan, 4, np.nan, 30]])

    assert_array_equal(model.transform([[100, 1, 200, np.nan]]), [[1, 10]])
    assert_array_equal(model.get_feature_names_out(), ["x1", "x3"])
    assert_array_equal(
        model.get_feature_names_out(["empty_a", "x", "empty_b", "y"]),
        ["x", "y"],
    )
    with pytest.raises(ValueError):
        model.get_feature_names_out(["x", "y"])
    with pytest.raises(ValueError):
        model.transform([[1, np.nan]])


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
def test_new_query_missingness_does_not_create_extra_indicators(policy, keep):
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        keep_empty_features=keep,
        add_indicator=True,
    ).fit([[np.nan, 0, 10, np.nan], [np.nan, 4, 30, np.nan]])

    # x2 is first missing at transform time. Only x0 and x3 have indicators.
    query = [[7, 1, np.nan, np.nan], [np.nan, 8, 9, 4]]
    base = [[0, 1, 10, 0], [0, 8, 9, 0]] if keep else [[1, 10], [8, 9]]
    expected = np.column_stack([base, [[0, 1], [1, 0]]])
    assert_array_equal(model.transform(query), expected)
    assert_array_equal(model.indicator_.features_, [0, 3])


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("weights,expected_y", [("distance", 15), (shifted_inverse, 16)])
def test_weighting_retains_original_dimension_after_dropping_empty_columns(
    policy, keep, weights, expected_y
):
    train = [[np.nan, 0, 10, np.nan], [np.nan, 4, 30, np.nan], [np.nan, 20, 90, np.nan]]
    model = FaissImputer(
        n_neighbors=2,
        donor_policy=policy,
        weights=weights,
        keep_empty_features=keep,
        add_indicator=True,
    ).fit(train)

    # Four original dimensions and one shared feature give distances 2, 6.
    # shifted_inverse weights 1/3 and 1/7 produce (70 + 90) / 10 = 16.
    # Scaling by the two retained dimensions would incorrectly give 16.306...
    expected = [0, 1, expected_y, 0, 0, 1] if keep else [1, expected_y, 0, 1]
    assert_allclose(
        model.transform([[99, 1, np.nan, np.nan]]), [expected], rtol=2e-6
    )


@pytest.mark.parametrize("keep", [False, True])
def test_available_callable_weights_scale_each_donor_overlap(keep):
    model = FaissImputer(
        n_neighbors=2,
        donor_policy="available",
        weights=shifted_inverse,
        keep_empty_features=keep,
    ).fit([[np.nan, 0, np.nan, 10], [np.nan, 4, 5, 30]])

    # The donors share one and two query features respectively. Full input
    # dimension four gives distances sqrt(4) and sqrt((9 + 16) * 4 / 2).
    w1, w2 = 1 / 3, 1 / (1 + np.sqrt(50))
    value = (10 * w1 + 30 * w2) / (w1 + w2)
    expected = [0, 1, 1, value] if keep else [1, 1, value]
    assert_allclose(model.transform([[123, 1, 1, np.nan]]), [expected], rtol=2e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_values_only_in_fit_empty_columns_use_fitted_fallback(policy, keep, strategy):
    model = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        keep_empty_features=keep,
        strategy=strategy,
    ).fit([[np.nan, 0, 10], [np.nan, 4, 30], [np.nan, 20, 90]])

    base = [8, 130 / 3] if strategy == "mean" else [4, 30]
    expected = [0] + base if keep else base
    assert_allclose(model.transform([[999, np.nan, np.nan]]), [expected], rtol=2e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("indicator", [False, True])
def test_all_empty_training_needs_no_donors_and_keeps_original_mask(
    policy, keep, indicator
):
    train = np.full((2, 3), np.nan, dtype=np.float32)
    queries = np.array(
        [[5, np.nan, 7], [np.nan, np.nan, np.nan], [1, 2, 3]],
        dtype=np.float32,
    )
    original_queries = queries.copy()
    train.setflags(write=False)
    queries.setflags(write=False)
    model = FaissImputer(
        n_neighbors=99,
        donor_policy=policy,
        weights=unexpected_weights,
        keep_empty_features=keep,
        add_indicator=indicator,
    ).fit(train)

    expected = np.zeros((3, 3 if keep else 0), dtype=np.float32)
    names = ["x0", "x1", "x2"] if keep else []
    if indicator:
        expected = np.column_stack([expected, np.isnan(original_queries)])
        names += [f"missingindicator_x{i}" for i in range(3)]
    actual = model.transform(queries)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert_array_equal(actual, expected)
    assert_array_equal(model.get_feature_names_out(), names)
    assert_array_equal(queries, original_queries)
    assert model.n_features_in_ == 3

    fitted_output = model.fit_transform(train)
    expected_fit = np.zeros((2, 3 if keep else 0), dtype=np.float32)
    if indicator:
        expected_fit = np.column_stack([expected_fit, np.ones((2, 3))])
    assert_array_equal(fitted_output, expected_fit)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [np.bool_(False), np.bool_(True)])
def test_numpy_boolean_option_clone_and_fit_transform(policy, keep):
    train = np.array([[np.nan, 2], [np.nan, 5]], dtype=np.float32)
    original = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        keep_empty_features=keep,
        add_indicator=True,
    )
    model = clone(original)
    assert model.get_params()["keep_empty_features"] == keep
    expected = [[0, 2, 1], [0, 5, 1]] if keep else [[2, 1], [5, 1]]
    assert_array_equal(model.fit_transform(train), expected)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "flag", [None, 0, 1, "true", np.array([True])],
    ids=["none", "zero", "one", "string", "array"],
)
def test_invalid_keep_option_clears_failed_refit_state(policy, flag):
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, add_indicator=True
    ).fit([[np.nan, 2], [np.nan, 5]])
    model.set_params(keep_empty_features=flag)
    with pytest.raises(ValueError, match="keep_empty_features"):
        model.fit([[np.nan, np.nan]])
    with pytest.raises(NotFittedError):
        model.transform([[np.nan, 3]])
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "parameters",
    [{"n_neighbors": 0}, {"weights": "invalid"}, {"metric": "invalid"}, {"strategy": "invalid"}],
)
def test_all_empty_fit_still_validates_parameters(policy, parameters):
    with pytest.raises(ValueError):
        FaissImputer(donor_policy=policy, **parameters).fit([[np.nan, np.nan]])


@pytest.mark.parametrize("factory", [None, "not-a-factory"])
def test_all_empty_complete_fit_validates_factory_and_clears_state(factory):
    model = FaissImputer(n_neighbors=1, add_indicator=True).fit(
        [[np.nan, 1], [np.nan, 2]]
    )
    model.set_params(index_factory=factory)
    with pytest.raises((ValueError, RuntimeError)):
        model.fit([[np.nan, np.nan]])
    assert not hasattr(model, "valid_features_")
    assert not hasattr(model, "indicator_")
    with pytest.raises(NotFittedError):
        model.transform([[np.nan, 2]])


@pytest.mark.parametrize("parameters", [{"metric": "ip"}, {"index_factory": "IVF2,Flat"}])
def test_all_empty_available_fit_still_rejects_unsupported_configuration(parameters):
    with pytest.raises(ValueError, match="donor_policy"):
        FaissImputer(donor_policy="available", **parameters).fit([[np.nan, np.nan]])


@pytest.mark.parametrize("policy", POLICIES)
def test_all_empty_model_still_validates_original_input_dimensions_and_finiteness(policy):
    model = FaissImputer(n_neighbors=1, donor_policy=policy).fit([[np.nan, np.nan]])
    for invalid_query in ([[np.nan]], [[np.nan, np.inf]], [[np.nan, -np.inf]]):
        with pytest.raises(ValueError):
            model.transform(invalid_query)
    for invalid_train in (np.empty((0, 2)), np.empty((2, 0)), [[np.nan, np.inf]]):
        with pytest.raises(ValueError):
            model.fit(invalid_train)
        with pytest.raises(NotFittedError):
            model.transform([[np.nan, np.nan]])


@pytest.mark.parametrize("policy", POLICIES)
def test_refit_replaces_empty_features_and_recovers_from_failure(policy):
    model = FaissImputer(
        n_neighbors=1, donor_policy=policy, add_indicator=True
    ).fit([[np.nan, 2, 10], [np.nan, 5, 30]])
    assert_array_equal(model.get_feature_names_out(), ["x1", "x2", "missingindicator_x0"])

    model.fit([[1, np.nan, 10], [4, np.nan, 30]])
    assert_array_equal(model.get_feature_names_out(), ["x0", "x2", "missingindicator_x1"])
    assert_array_equal(model.transform([[2, 99, np.nan]]), [[2, 10, 0]])

    model.fit([[np.nan, np.nan, np.nan]])
    assert_array_equal(model.transform([[1, np.nan, 3]]), [[0, 1, 0]])
    with pytest.raises(ValueError):
        model.fit([[0, np.inf, 2]])
    with pytest.raises(NotFittedError):
        model.transform([[1, 2, 3]])
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()

    model.set_params(keep_empty_features=True).fit([[1, np.nan, 10], [4, np.nan, 30]])
    assert_array_equal(model.transform([[2, 99, np.nan]]), [[2, 0, 10, 0]])
    assert_array_equal(model.get_feature_names_out(), ["x0", "x1", "x2", "missingindicator_x1"])


def test_complete_policy_still_requires_complete_donors_after_empty_projection():
    model = FaissImputer(n_neighbors=1, keep_empty_features=True, add_indicator=True)
    model.fit([[np.nan, 0, 10], [np.nan, 4, 30]])
    with pytest.raises(ValueError):
        model.fit([[np.nan, 0, np.nan], [np.nan, np.nan, 10]])
    with pytest.raises(NotFittedError):
        model.transform([[np.nan, 1, np.nan]])
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
def test_pandas_output_names_indices_and_column_transformer(policy, keep):
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame(
        [[np.nan, 0, 10, 7], [np.nan, 4, 30, 9]],
        columns=["empty", "x", "y", "passthrough"],
        dtype=np.float32,
    )
    queries = pd.DataFrame(
        [[100, 1, np.nan, 11], [np.nan, 8, 9, 12]],
        columns=train.columns,
        index=["query-a", "query-b"],
        dtype=np.float32,
    )
    original_train = train.copy(deep=True)
    original_queries = queries.copy(deep=True)
    transformer = ColumnTransformer(
        [("imputer", FaissImputer(
            n_neighbors=1,
            donor_policy=policy,
            keep_empty_features=keep,
            add_indicator=True,
        ), ["empty", "x", "y"])],
        remainder="passthrough",
    ).set_output(transform="pandas")
    transformer.fit(train)
    actual = transformer.transform(queries)

    names = ["imputer__x", "imputer__y", "imputer__missingindicator_empty", "remainder__passthrough"]
    expected = [[1, 10, 0, 11], [8, 9, 1, 12]]
    if keep:
        names.insert(0, "imputer__empty")
        expected = [[0] + row for row in expected]
    assert isinstance(actual, pd.DataFrame)
    assert_array_equal(actual.columns, names)
    assert_array_equal(transformer.get_feature_names_out(), names)
    assert_array_equal(actual.index, queries.index)
    assert_array_equal(actual.to_numpy(), expected)
    assert all(dtype == np.float32 for dtype in actual.dtypes)
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(queries, original_queries)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("keep", [False, True])
def test_column_transformer_supports_an_all_empty_feature_group(policy, keep):
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame({"empty_a": [np.nan, np.nan], "empty_b": [np.nan, np.nan], "value": [2, 5]})
    queries = pd.DataFrame({"empty_a": [9], "empty_b": [np.nan], "value": [7]}, index=["q"])
    transformer = ColumnTransformer(
        [("empty", FaissImputer(donor_policy=policy, keep_empty_features=keep), ["empty_a", "empty_b"])],
        remainder="passthrough",
    ).set_output(transform="pandas").fit(train)

    actual = transformer.transform(queries)
    names = ["empty__empty_a", "empty__empty_b", "remainder__value"] if keep else ["remainder__value"]
    assert_array_equal(actual.columns, names)
    assert_array_equal(actual.index, queries.index)
    assert_array_equal(actual.to_numpy(), [[0, 0, 7]] if keep else [[7]])
    assert_array_equal(transformer.get_feature_names_out(), names)
