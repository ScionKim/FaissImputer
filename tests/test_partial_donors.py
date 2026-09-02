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
