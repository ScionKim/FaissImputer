import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer

from faiss_imputer import FaissImputer


POLICIES = ["complete", "available"]


def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)


def large_weights(distances):
    return np.full(distances.shape, 1e308)


def failing_weights(distances):
    raise RuntimeError("weight callback failed")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("weights", ["uniform", "distance", shifted_inverse])
def test_weights_match_hand_calculated_mean_and_preserve_parameters(
    policy, weights
):
    train = np.array(
        [[0, 10, 100], [4, 30, 300], [20, 90, 900]],
        dtype=np.float32,
    )
    queries = np.array([[1, np.nan, np.nan]], dtype=np.float32)

    if weights == "uniform":
        expected_value = 20.0
    elif weights == "distance":
        # Distances are sqrt(3) and 3 * sqrt(3), not 3 and 27.
        expected_value = 15.0
    else:
        # The additive constant also detects missing-feature scaling errors.
        first = 1.0 / (1.0 + np.sqrt(3.0))
        second = 1.0 / (1.0 + 3.0 * np.sqrt(3.0))
        expected_value = (10 * first + 30 * second) / (first + second)

    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights=weights
    ).fit(train)
    actual = model.transform(queries)

    assert actual.dtype == np.float32
    assert_allclose(
        actual,
        [[1, expected_value, 10 * expected_value]],
        rtol=2e-6,
        atol=2e-6,
    )
    assert model.get_params()["weights"] == weights

    copied = clone(model)
    assert copied.get_params()["weights"] == weights
    assert_array_equal(copied.fit(train).transform(queries), actual)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("weights", ["distance", shifted_inverse])
def test_weighted_predictions_match_knn_without_mutating_inputs(
    policy, weights
):
    train = np.array(
        [
            [0, 2, 10, 100],
            [2, 3, 20, 200],
            [5, 6, 40, 400],
            [9, 8, 80, 800],
            [14, 13, 160, 1600],
        ],
        dtype=np.float32,
    )
    if policy == "available":
        train[1, 3] = np.nan
        train[2, 1] = np.nan
        train[3, 2] = np.nan

    queries = np.array(
        [
            [1, 2.5, np.nan, np.nan],
            [7, np.nan, 50, np.nan],
            [np.nan, 10, np.nan, 900],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    original_train = train.copy()
    original_queries = queries.copy()
    train.setflags(write=False)
    queries.setflags(write=False)

    expected = KNNImputer(
        n_neighbors=2, weights=weights
    ).fit(original_train.copy()).transform(original_queries.copy())
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights=weights
    ).fit(train)
    actual = model.transform(queries)

    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    observed = ~np.isnan(queries)
    assert_array_equal(actual[observed], queries[observed])
    assert_array_equal(train, original_train)
    assert_array_equal(queries, original_queries)
    assert_array_equal(model.transform(queries), actual)

    individually = np.vstack(
        [model.transform(row[None, :]) for row in queries]
    )
    assert_array_equal(individually, actual)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_explicit_uniform_and_none_preserve_default_results(policy, strategy):
    train = np.array([[0, 10], [4, 30], [9, 80]], dtype=np.float32)
    queries = np.array(
        [[1, np.nan], [np.nan, np.nan], [7, 8]], dtype=np.float32
    )
    parameters = dict(
        n_neighbors=3, donor_policy=policy, strategy=strategy
    )
    baseline = FaissImputer(**parameters).fit(train)
    expected = baseline.transform(queries)

    for weights in ("uniform", None):
        model = FaissImputer(**parameters, weights=weights).fit(train)
        assert_array_equal(model.statistics_, baseline.statistics_)
        assert_array_equal(model.transform(queries), expected)


@pytest.mark.parametrize("policy", POLICIES)
def test_distance_weights_use_only_zero_distance_neighbors(policy):
    train = np.array([[0, 10], [0, 30], [4, 90]], dtype=np.float32)
    queries = np.array([[0, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=3, donor_policy=policy, weights="distance"
    ).fit(train)

    assert_array_equal(model.transform(queries), [[0, 20]])


def test_available_weights_select_donors_separately_for_each_feature():
    train = np.array(
        [
            [0, 10, np.nan],
            [0, 30, np.nan],
            [2, 90, 100],
            [4, np.nan, 400],
            [np.nan, 1000, 900],
        ],
        dtype=np.float32,
    )
    queries = np.array([[0, np.nan, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=3, donor_policy="available", weights="distance"
    ).fit(train)

    # Feature 1 uses the two exact matches. Feature 2 uses distances 2:4.
    # The donor with no shared observed feature contributes to neither.
    assert_allclose(
        model.transform(queries), [[0, 20, 200]], rtol=2e-6
    )


def test_available_weights_remain_aligned_after_candidate_expansion():
    train = np.column_stack(
        [np.arange(34), np.full(34, np.nan)]
    ).astype(np.float32)
    train[20:, 1] = 10 * np.arange(1, 15)
    queries = np.array(
        [[32.25, np.nan], [0, np.nan]], dtype=np.float32
    )
    model = FaissImputer(
        n_neighbors=2, donor_policy="available", weights="distance"
    ).fit(train)

    # The first query finishes early; the second needs more candidates.
    actual = model.transform(queries)
    assert_allclose(
        actual,
        [[32.25, 132.5], [0, 610 / 41]],
        rtol=2e-6,
    )
    individually = np.vstack(
        [model.transform(row[None, :]) for row in queries]
    )
    assert_array_equal(individually, actual)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("weights", ["distance", failing_weights])
def test_all_missing_queries_use_unweighted_fitted_statistics(policy, weights):
    train = np.array([[0, 10], [8, 50]], dtype=np.float32)
    queries = np.array([[np.nan, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights=weights
    ).fit(train)

    assert_array_equal(model.transform(queries), [[4, 30]])


@pytest.mark.parametrize("weights", ["distance", failing_weights])
def test_available_no_overlap_uses_fitted_statistics(weights):
    train = np.array(
        [[0, np.nan], [np.nan, 10], [np.nan, 30]], dtype=np.float32
    )
    queries = np.array([[1, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2, donor_policy="available", weights=weights
    ).fit(train)

    assert_array_equal(model.transform(queries), [[1, 20]])


@pytest.mark.parametrize("policy", POLICIES)
def test_tiny_distances_are_not_treated_as_exact_matches(policy):
    train = np.array(
        [[0, 10], [2e-23, 30], [1, 90]], dtype=np.float32
    )
    queries = np.array([[0.5e-23, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights="distance"
    ).fit(train)

    # Squared distances underflow in float32, but their ratio is 1:9.
    assert_allclose(model.transform(queries)[0, 1], 15, rtol=2e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("weights", ["distance", large_weights])
@pytest.mark.parametrize("sign", [-1, 1])
def test_large_values_and_weights_produce_finite_means(policy, weights, sign):
    value = np.float32(sign * 0.75 * np.finfo(np.float32).max)
    train = np.array([[0, value], [4, value]], dtype=np.float32)
    queries = np.array([[1, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights=weights
    ).fit(train)

    actual = model.transform(queries)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert actual[0, 1] == value


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "weights",
    [True, 1, "invalid", {}, np.array(["uniform", "distance"])],
    ids=["bool", "integer", "string", "mapping", "array"],
)
def test_invalid_weights_clear_failed_refit_state(policy, weights):
    train = np.array([[0, 10], [4, 30]], dtype=np.float32)
    model = FaissImputer(n_neighbors=2, donor_policy=policy).fit(train)
    model.set_params(weights=weights)

    with pytest.raises(ValueError, match="weights"):
        model.fit(train)
    with pytest.raises(NotFittedError):
        model.transform([[1, np.nan]])


@pytest.mark.parametrize("weights", ["distance", shifted_inverse])
@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"strategy": "median"}, "strategy"),
        ({"metric": "ip"}, "metric"),
    ],
)
def test_nonuniform_weights_reject_unsupported_combinations(
    weights, parameters, message
):
    train = np.array([[0, 10], [4, 30]], dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        FaissImputer(
            n_neighbors=2, weights=weights, **parameters
        ).fit(train)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "weights",
    [
        lambda distances: 1.0,
        lambda distances: np.ones_like(distances, dtype=complex) * 1j,
        lambda distances: np.full_like(distances, np.inf),
        lambda distances: np.zeros_like(distances),
        lambda distances: np.tile([1.0, -1.0], (distances.shape[0], 1)),
    ],
    ids=["wrong-shape", "complex", "infinite", "all-zero", "zero-sum"],
)
def test_invalid_callable_outputs_raise_clear_errors(policy, weights):
    train = np.array([[0, 10], [4, 30]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2, donor_policy=policy, weights=weights
    ).fit(train)

    with pytest.raises(ValueError, match="weights"):
        model.transform([[1, np.nan]])


def test_available_cache_is_cleared_when_callable_raises():
    train = np.array([[0, 10], [4, 30]], dtype=np.float32)
    queries = np.array([[0, np.nan]], dtype=np.float32)
    model = FaissImputer(
        n_neighbors=2,
        donor_policy="available",
        weights=failing_weights,
    ).fit(train)

    with pytest.raises(RuntimeError, match="weight callback failed"):
        model.transform(queries)

    assert model.available_index_.query_ref is None
    assert model.available_index_.matrix is None
    assert model.available_index_.precise_rows == {}

    model.set_params(weights="distance")
    assert_array_equal(model.transform(queries), [[0, 10]])
    assert model.available_index_.query_ref is None
    assert model.available_index_.matrix is None
    assert model.available_index_.precise_rows == {}
