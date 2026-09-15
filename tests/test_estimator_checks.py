"""Common scikit-learn estimator checks for both donor policies."""

from sklearn.utils.estimator_checks import parametrize_with_checks
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


@parametrize_with_checks(
    [
        FaissImputer(n_neighbors=1, donor_policy="complete"),
        FaissImputer(n_neighbors=1, donor_policy="available"),
    ]
)
def test_estimator_checks(estimator, check):
    with threadpool_limits(limits=1):
        check(estimator)