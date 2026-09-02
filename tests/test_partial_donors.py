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
