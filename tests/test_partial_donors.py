import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from faiss_imputer import FaissImputer


def _partial_train():
    return np.array(
        [
            [0.0, 10.0, np.nan],
            [2.0, np.nan, 20.0],
        ],
        dtype=np.float32,
    )

def test_available_mode_uses_partially_observed_donors():
    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
    )
    imputer.fit(_partial_train())

    result = imputer.transform([[0.1, np.nan, np.nan]])

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.1, 10.0, 20.0]])

@pytest.mark.parametrize("previously_fitted", [False, True])
def test_available_mode_failed_fit_clears_state(previously_fitted):
    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
    )

    if previously_fitted:
        imputer.fit(_partial_train())

    with pytest.raises(ValueError, match="all-missing"):
        imputer.fit(
            [
                [1.0, 10.0, np.nan],
                [2.0, 20.0, np.nan],
            ]
        )

    with pytest.raises(NotFittedError):
        check_is_fitted(imputer)

    with pytest.raises(NotFittedError):
        imputer.transform([[0.1, np.nan, np.nan]])

@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize(
    "train, query, k, expected_mean, expected_median",
    [
        pytest.param(
            [[2, np.nan, 10], [1, 2, 20]],
            [[0, 0, np.nan]],
            1,
            [[0, 0, 20]],
            [[0, 0, 20]],
            id="scaled-distance",
        ),
        pytest.param(
            [[0, np.nan, 100], [1, 20, np.nan], [2, 40, np.nan]],
            [[0, np.nan, np.nan]],
            1,
            [[0, 20, 100]],
            [[0, 20, 100]],
            id="neighbors-per-column",
        ),
        pytest.param(
            [[0, np.nan], [np.nan, 10], [np.nan, 30], [np.nan, 110]],
            [[5, np.nan]],
            1,
            [[5, 50]],
            [[5, 30]],
            id="no-overlap-fallback",
        ),
        pytest.param(
            [[0, 10], [2, 30], [3, 110], [np.nan, 1000]],
            [[1, np.nan]],
            5,
            [[1, 50]],
            [[1, 30]],
            id="fewer-than-k-neighbors",
        ),
        pytest.param(
            [[0, 10, np.nan], [100, 20, 5]],
            [[0, np.nan, 5]],
            1,
            [[0, 10, 5]],
            [[0, 10, 5]],
            id="partial-donor-can-be-closest",
        ),
    ],
)

def test_available_mode_neighbor_rules(
    train, query, k, expected_mean, expected_median, strategy
):
    imputer = FaissImputer(
        n_neighbors=k,
        strategy=strategy,
        donor_policy="available",
    )
    result = imputer.fit(train).transform(query)
    expected = expected_mean if strategy == "mean" else expected_median

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize(
    "policy, metric, factory",
    [
        ("unknown", "l2", "Flat"),
        ("available", "ip", "Flat"),
        ("available", "l2", "IVF1,Flat"),
    ],
)
def test_partial_donor_configuration_validation(policy, metric, factory):
    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy=policy,
        metric=metric,
        index_factory=factory,
    )

    with pytest.raises(ValueError, match="donor_policy"):
        imputer.fit([[0, 10], [2, 20]])

    with pytest.raises(NotFittedError):
        check_is_fitted(imputer)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_available_mode_matches_default_on_complete_training(strategy):
    train = [[0, 10, 100], [2, 30, 300], [6, 110, 1100]]
    query = [
        [1, np.nan, np.nan],
        [np.nan, 35, 400],
        [np.nan, np.nan, np.nan],
    ]
    params = {"n_neighbors": 2, "strategy": strategy}

    expected = FaissImputer(**params).fit(train).transform(query)
    result = (
        FaissImputer(donor_policy="available", **params)
        .fit(train)
        .transform(query)
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_available_mode_preserves_inputs_and_batch_behavior(strategy):
    train = np.array(
        [
            [0, 10, np.nan],
            [2, 30, np.nan],
            [4, np.nan, 100],
            [6, np.nan, 300],
            [np.nan, 110, 1100],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    query = np.full((302, 3), np.nan, dtype=np.float32)
    query[:300, 0] = np.linspace(0.13, 5.73, 300, dtype=np.float32)
    query[300, 1] = 12

    train_before = train.copy()
    query_before = query.copy()

    imputer = FaissImputer(
        n_neighbors=1,
        strategy=strategy,
        donor_policy="available",
    ).fit(train)

    batch = imputer.transform(query)
    individual = np.vstack(
        [imputer.transform(query[i:i + 1]) for i in range(len(query))]
    )

    np.testing.assert_allclose(batch, individual, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_allclose(batch[300], [0, 12, 1100])

    expected_fallback = (
        [3, 50, 500] if strategy == "mean" else [3, 30, 300]
    )
    np.testing.assert_allclose(batch[301], expected_fallback)

    train[:] = 999
    np.testing.assert_allclose(
        imputer.transform(query), batch, rtol=1e-6, atol=1e-6
    )


def test_available_mode_ignores_invalid_faiss_neighbor_ids(monkeypatch):
    class NoNeighbors:
        def __init__(self, dimension):
            pass

        def add(self, vectors):
            pass

        def search(self, queries, k):
            shape = (queries.shape[0], k)
            return (
                np.zeros(shape, dtype=np.float32),
                np.full(shape, -1, dtype=np.int64),
            )

    monkeypatch.setattr("faiss.IndexFlatL2", NoNeighbors)

    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
    ).fit(
        [
            [0, 10, np.nan],
            [1, 30, np.nan],
            [2, np.nan, 100],
        ]
    )

    result = imputer.transform([[0, np.nan, np.nan]])
    np.testing.assert_allclose(result, [[0, 20, 100]])

def test_available_mode_builds_one_index_per_query(monkeypatch):
    import faiss

    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
    ).fit(
        [
            [0, 10, np.nan, np.nan],
            [1, 20, 30, np.nan],
            [2, 40, np.nan, 50],
            [3, 60, 70, 80],
        ]
    )

    original_index = faiss.IndexFlatL2
    created = []

    def counted_index(*args, **kwargs):
        created.append(1)
        return original_index(*args, **kwargs)

    monkeypatch.setattr(faiss, "IndexFlatL2", counted_index)

    result = imputer.transform([[0.1, np.nan, np.nan, np.nan]])

    np.testing.assert_allclose(result, [[0.1, 10, 30, 50]])
    assert len(created) == 1

def test_available_mode_expands_search_for_each_target():
    train = np.column_stack(
        [
            np.arange(64),
            np.arange(64) + 100,
            np.arange(64) + 1000,
        ]
    ).astype(np.float32)
    train[:60, 2] = np.nan

    imputer = FaissImputer(
        n_neighbors=2,
        donor_policy="available",
    ).fit(train)

    result = imputer.transform([[0.1, np.nan, np.nan]])

    np.testing.assert_allclose(result, [[0.1, 100.5, 1060.5]])


def test_available_mode_handles_large_finite_distances():
    imputer = FaissImputer(
        n_neighbors=1,
        donor_policy="available",
    ).fit(
        [
            [1.1e19, np.nan, 10],
            [np.nan, 1.2e19, 20],
        ]
    )

    result = imputer.transform([[0, 0, np.nan]])

    np.testing.assert_allclose(result, [[0, 0, 10]])
