from itertools import combinations

import numpy as np
import pytest

from faiss_imputer import FaissImputer


def _reference_complete_l2(donors, queries, n_neighbors, strategy):
    result = queries.copy()
    if strategy == "mean":
        statistics = np.mean(donors, axis=0)
    else:
        statistics = np.median(donors, axis=0)

    for row_index, row in enumerate(queries):
        missing = np.isnan(row)
        if not missing.any():
            continue

        observed = ~missing
        if not observed.any():
            result[row_index, missing] = statistics[missing]
            continue

        offsets = donors[:, observed] - row[observed]
        distances = np.sum(offsets * offsets, axis=1)
        neighbors = np.argsort(distances, kind="stable")[:n_neighbors]
        values = donors[neighbors][:, missing]

        if strategy == "mean":
            result[row_index, missing] = np.mean(values, axis=0)
        else:
            result[row_index, missing] = np.median(values, axis=0)

    return result


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_complete_flat_l2_mixed_masks_match_reference(strategy):
    rng = np.random.default_rng(20260903)
    donors = rng.normal(size=(128, 8)).astype(np.float32)
    queries = rng.normal(size=(26, 8)).astype(np.float32)

    for row, missing_columns in zip(
        queries[:24],
        list(combinations(range(8), 2))[:24],
    ):
        row[list(missing_columns)] = np.nan

    queries[-2] = np.nan
    donors_before = donors.copy()
    queries_before = queries.copy()
    expected = _reference_complete_l2(
        donors,
        queries,
        n_neighbors=3,
        strategy=strategy,
    )

    result = FaissImputer(
        n_neighbors=3,
        metric="l2",
        strategy=strategy,
        index_factory="Flat",
        donor_policy="complete",
    ).fit(donors).transform(queries)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(donors, donors_before)
    np.testing.assert_array_equal(queries, queries_before)
    np.testing.assert_array_equal(
        result[~np.isnan(queries)],
        queries[~np.isnan(queries)],
    )
    assert result.dtype == np.float32
    assert not np.shares_memory(result, queries)
