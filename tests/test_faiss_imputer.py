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
