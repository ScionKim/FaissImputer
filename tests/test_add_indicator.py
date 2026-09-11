import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faiss_imputer import FaissImputer


POLICIES = ["complete", "available"]
BASE_NAMES = ["x0", "x1", "x2"]
INDICATOR_NAMES = ["missingindicator_x0", "missingindicator_x1"]

EXPECTED_INDICATORS = np.array(
    [[0, 1], [1, 0], [1, 1], [0, 0], [0, 0]],
    dtype=np.float32,
)


def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)


def example_data():
    train = np.array(
        [
            [0, 10, 100],
            [2, np.nan, 200],
            [4, 50, 300],
            [np.nan, 70, 400],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [1, np.nan, 150],
            [np.nan, 30, 250],
            [np.nan, np.nan, np.nan],
            [3, 40, 350],
            [1, 20, np.nan],
        ],
        dtype=np.float32,
    )
    return train, queries


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "strategy,weights",
    [
        ("mean", "uniform"),
        ("median", "uniform"),
        ("mean", "distance"),
        ("mean", shifted_inverse),
    ],
    ids=["mean", "median", "distance", "callable"],
)
def test_indicators_preserve_imputation_and_original_inputs(
    policy, strategy, weights
):
    train, queries = example_data()
    original_train = train.copy()
    original_queries = queries.copy()
    train.setflags(write=False)
    queries.setflags(write=False)

    parameters = dict(
        n_neighbors=2,
        donor_policy=policy,
        strategy=strategy,
        weights=weights,
    )
    baseline = FaissImputer(**parameters).fit(train).transform(queries)
    model = FaissImputer(**parameters, add_indicator=True).fit(train)
    actual = model.transform(queries)

    assert actual.shape == (len(queries), 5)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert_array_equal(actual[:, :3], baseline)
    assert_array_equal(actual[:, 3:], EXPECTED_INDICATORS)

    # Partial training rows affect the indicator even in complete mode.
    assert_array_equal(model.indicator_.features_, [0, 1])
    assert_array_equal(
        model.get_feature_names_out(), BASE_NAMES + INDICATOR_NAMES
    )

    observed = ~np.isnan(queries)
    assert_array_equal(actual[:, :3][observed], queries[observed])
    assert_array_equal(train, original_train)
    assert_array_equal(queries, original_queries)


@pytest.mark.parametrize("weights", ["uniform", "distance", shifted_inverse])
def test_available_indicator_output_matches_knn(weights):
    train, queries = example_data()
    expected_model = KNNImputer(
        n_neighbors=2, weights=weights, add_indicator=True
    ).fit(train.copy())
    model = FaissImputer(
        n_neighbors=2,
        donor_policy="available",
        weights=weights,
        add_indicator=True,
    ).fit(train)

    expected = expected_model.transform(queries.copy())
    actual = model.transform(queries)

    assert_allclose(actual[:, :3], expected[:, :3], rtol=3e-6, atol=3e-6)
    assert_array_equal(actual[:, 3:], expected[:, 3:])
    assert_array_equal(
        model.get_feature_names_out(),
        expected_model.get_feature_names_out(),
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("flag", [np.bool_(False), np.bool_(True)])
def test_numpy_boolean_parameter_clone_and_fit_transform(policy, flag):
    train, _ = example_data()
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, add_indicator=flag
    )
    copied = clone(model)
    assert copied.get_params()["add_indicator"] == flag

    actual = copied.fit_transform(train)

    assert actual.dtype == np.float32
    assert actual.shape == (len(train), 5 if flag else 3)
    if flag:
        assert_array_equal(actual[:, 3:], np.isnan(train[:, [0, 1]]))
    else:
        assert copied.indicator_ is None
        assert_array_equal(copied.get_feature_names_out(), BASE_NAMES)


@pytest.mark.parametrize("policy", POLICIES)
def test_refit_replaces_disables_and_clears_indicator_features(policy):
    train, queries = example_data()
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, add_indicator=True
    ).fit(train)
    assert_array_equal(model.indicator_.features_, [0, 1])

    replacement = np.array(
        [[0, 10, 100], [2, 30, np.nan], [4, 50, 300]],
        dtype=np.float32,
    )
    model.fit(replacement)
    assert_array_equal(model.indicator_.features_, [2])
    assert_array_equal(
        model.get_feature_names_out(),
        BASE_NAMES + ["missingindicator_x2"],
    )
    actual = model.transform(queries)
    assert actual.shape == (len(queries), 4)
    assert_array_equal(actual[:, 3], np.isnan(queries[:, 2]))

    model.set_params(add_indicator=False).fit(replacement)
    assert model.indicator_ is None
    assert model.transform(queries).shape == queries.shape
    assert_array_equal(model.get_feature_names_out(), BASE_NAMES)

    # Re-enable on fully observed training data. New query missingness
    # must not introduce indicator columns or raise an error.
    model.set_params(add_indicator=True).fit(replacement[[0, 2]])
    assert model.indicator_.features_.size == 0
    actual = model.transform(queries)
    assert actual.shape == queries.shape
    assert np.isfinite(actual).all()
    assert_array_equal(model.get_feature_names_out(), BASE_NAMES)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "flag",
    [None, 0, 1, "true", np.array([True])],
    ids=["none", "zero", "one", "string", "array"],
)
def test_invalid_indicator_parameter_clears_failed_refit_state(policy, flag):
    train, queries = example_data()
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, add_indicator=True
    ).fit(train)
    model.set_params(add_indicator=flag)

    with pytest.raises(ValueError, match="add_indicator"):
        model.fit(train)

    assert not hasattr(model, "indicator_")
    with pytest.raises(NotFittedError):
        model.transform(queries)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()


@pytest.mark.parametrize("policy", POLICIES)
def test_failure_after_indicator_fitting_leaves_model_unfitted(policy, monkeypatch):
    train, queries = example_data()
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, add_indicator=True
    ).fit(train)

    if policy == "complete":
        # Every column has observations, but no row is complete.
        invalid_train = np.array(
            [[0, np.nan, 100], [np.nan, 20, 200]], dtype=np.float32
        )
    else:
        invalid_train = train

    with monkeypatch.context() as patch:
        if policy == "available":
            # Empty columns are now supported. A backend failure still
            # exercises cleanup after fitting the missingness indicator.
            import faiss_imputer.faiss_imputer as implementation

            def fail_backend(*args, **kwargs):
                raise ValueError("injected backend failure")

            patch.setattr(implementation, "MatrixNaNIndex", fail_backend)
        with pytest.raises(ValueError):
            model.fit(invalid_train)

    assert not hasattr(model, "indicator_")
    assert not hasattr(model, "n_features_in_")
    with pytest.raises(NotFittedError):
        model.transform(queries)
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()

    model.fit(train)
    assert_array_equal(model.indicator_.features_, [0, 1])


def test_available_fallback_keeps_original_missingness_indicators():
    train = np.array(
        [[0, np.nan], [np.nan, 10], [np.nan, 30]], dtype=np.float32
    )
    queries = np.array(
        [[1, np.nan], [np.nan, np.nan], [2, 25]], dtype=np.float32
    )
    model = FaissImputer(
        n_neighbors=2,
        donor_policy="available",
        weights="distance",
        add_indicator=True,
    ).fit(train)

    assert_array_equal(
        model.transform(queries),
        [[1, 20, 0, 1], [0, 20, 1, 1], [2, 25, 0, 0]],
    )


def test_feature_names_before_fit_and_with_explicit_names():
    train, _ = example_data()
    model = FaissImputer(
        n_neighbors=2, donor_policy="available", add_indicator=True
    )

    with pytest.raises(NotFittedError):
        model.get_feature_names_out()

    model.fit(train)
    assert_array_equal(
        model.get_feature_names_out(["a", "b", "c"]),
        ["a", "b", "c", "missingindicator_a", "missingindicator_b"],
    )
    with pytest.raises(ValueError):
        model.get_feature_names_out(["a", "b"])


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("global_output", [False, True])
def test_pandas_pipeline_preserves_names_index_and_indicator_values(
    policy, global_output
):
    pd = pytest.importorskip("pandas")
    train, queries = example_data()
    columns = ["age", "income", "score"]
    train_frame = pd.DataFrame(train, columns=columns)
    query_frame = pd.DataFrame(
        queries,
        columns=columns,
        index=[f"query-{i}" for i in range(len(queries))],
    )
    original_train = train_frame.copy(deep=True)
    original_queries = query_frame.copy(deep=True)
    expected_names = columns + [
        "missingindicator_age",
        "missingindicator_income",
    ]

    with config_context(
        transform_output="pandas" if global_output else "default"
    ):
        pipe = make_pipeline(
            FaissImputer(
                n_neighbors=2,
                donor_policy=policy,
                weights="distance",
                add_indicator=True,
            ),
            StandardScaler(),
        )
        if not global_output:
            pipe.set_output(transform="pandas")

        pipe.fit(train_frame)
        indicated = pipe[0].transform(query_frame)
        actual = pipe.transform(query_frame)

    assert isinstance(indicated, pd.DataFrame)
    assert isinstance(actual, pd.DataFrame)
    assert_array_equal(indicated.columns, expected_names)
    assert_array_equal(actual.columns, expected_names)
    assert_array_equal(indicated.index, query_frame.index)
    assert_array_equal(actual.index, query_frame.index)
    assert_array_equal(pipe.get_feature_names_out(), expected_names)
    assert all(dtype == np.float32 for dtype in indicated.dtypes)
    assert_array_equal(indicated.iloc[:, 3:].to_numpy(), EXPECTED_INDICATORS)
    assert np.isfinite(actual.to_numpy()).all()

    pd.testing.assert_frame_equal(train_frame, original_train)
    pd.testing.assert_frame_equal(query_frame, original_queries)

    with pytest.raises(ValueError):
        pipe[0].get_feature_names_out(["age", "income", "different"])
