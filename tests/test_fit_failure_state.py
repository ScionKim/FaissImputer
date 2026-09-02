import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from faiss_imputer import FaissImputer


@pytest.mark.parametrize("previously_fitted", [False, True])
def test_failed_fit_leaves_imputer_unfitted(previously_fitted):
    imputer = FaissImputer(n_neighbors=1)

    if previously_fitted:
        imputer.fit(
            np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        )

    # Neither row is complete, so fitting must fail.
    invalid_train = np.array(
        [[1.0, np.nan], [np.nan, 20.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="at least one complete row"):
        imputer.fit(invalid_train)

    with pytest.raises(NotFittedError):
        check_is_fitted(imputer)

    with pytest.raises(NotFittedError):
        imputer.transform([[np.nan, np.nan]])
