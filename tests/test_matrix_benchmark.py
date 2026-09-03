import numpy as np
import pytest

from benchmarks.benchmark_partial_donors import MatrixNaNBenchmark


@pytest.mark.parametrize(
    "scale", [1.0, 1e-30], ids=["normal-scale", "tiny-scale"]
)
@pytest.mark.parametrize(
    "reverse_rows", [False, True], ids=["farther-first", "nearer-first"]
)
def test_matrix_benchmark_keeps_nearest_donor(scale, reverse_rows):
    train = np.array(
        [[2 * scale, 10.0], [scale, 20.0]],
        dtype=np.float32,
    )
    if reverse_rows:
        train = train[::-1].copy()
    query = np.array([[0.0, np.nan]], dtype=np.float32)

    delta = train[:, 0].astype(np.float64) - float(query[0, 0])
    distances = delta * delta
    assert np.all(distances > 0)
    assert distances[0] != distances[1]
    assert train[np.argmin(distances), 1] == 20.0

    result = MatrixNaNBenchmark(
        n_neighbors=1,
        donor_policy="available",
    ).fit(train).transform(query)

    np.testing.assert_array_equal(result, [[0.0, 20.0]])
