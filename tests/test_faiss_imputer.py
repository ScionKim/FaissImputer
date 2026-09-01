import faiss
import numpy as np

from faiss_imputer import FaissImputer


def test_fit_accepts_missing_values():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
        ],
        dtype=np.float32,
    )

    imputer = FaissImputer(n_neighbors=1)

    assert imputer.fit(X) is imputer

def test_transform_uses_fitted_donors():
    train = np.array(
        [
            [0.0, 200.0],
            [10.0, 200.0],
            [20.0, 200.0],
        ],
        dtype=np.float32,
    )
    test = np.array(
        [
            [19.0, np.nan],
            [0.0, 200.0],
        ],
        dtype=np.float32,
    )

    imputer = FaissImputer(n_neighbors=1).fit(train)
    transformed = imputer.transform(test)

    assert transformed[0, 1] == 200.0

def test_transform_does_not_modify_input():
    train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    test = np.array(
        [
            [2.0, np.nan],
            [3.0, 0.5],
        ],
        dtype=np.float32,
    )
    original = test.copy()

    imputer = FaissImputer(n_neighbors=1).fit(train)
    imputer.transform(test)

    np.testing.assert_array_equal(test, original)

def test_transform_is_independent_of_batch_values():
    train = np.array(
        [
            [0.0, 0.0],
            [10.0, 100.0],
        ],
        dtype=np.float32,
    )
    batch_a = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [9.0, np.nan],
        ],
        dtype=np.float32,
    )
    batch_b = np.array(
        [
            [1.0, 100.0],
            [2.0, 100.0],
            [9.0, np.nan],
        ],
        dtype=np.float32,
    )

    imputer = FaissImputer(n_neighbors=1).fit(train)

    result_a = imputer.transform(batch_a)[-1, 1]
    result_b = imputer.transform(batch_b)[-1, 1]

    assert result_a == result_b == 100.0

def test_fit_requires_at_least_one_complete_donor():
    X = np.array(
        [
            [1.0, np.nan],
            [np.nan, 2.0],
        ],
        dtype=np.float32,
    )

    with np.testing.assert_raises(ValueError):
        FaissImputer(n_neighbors=1).fit(X)

def test_fit_rejects_more_neighbors_than_donors():
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
        ],
        dtype=np.float32,
    )

    with np.testing.assert_raises(ValueError):
        FaissImputer(n_neighbors=3).fit(X)

def test_transform_rejects_different_feature_count():
    train = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
        ],
        dtype=np.float32,
    )
    invalid_test = np.array(
        [
            [0.0, 10.0, 20.0],
        ],
        dtype=np.float32,
    )

    imputer = FaissImputer(n_neighbors=1).fit(train)

    with np.testing.assert_raises(ValueError):
        imputer.transform(invalid_test)

def test_inner_product_metric_reaches_ivf_quantizer():
    X = (
        np.random.default_rng(0)
        .normal(size=(100, 4))
        .astype(np.float32)
    )

    imputer = FaissImputer(
        n_neighbors=1,
        metric='ip',
        index_factory='IVF2,Flat',
    ).fit(X)

    quantizer = faiss.downcast_index(imputer.index_.quantizer)

    assert quantizer.metric_type == faiss.METRIC_INNER_PRODUCT
