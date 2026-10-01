"""Compare bounded selected-distance batches with the old rowwise loop."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
import faiss_imputer._matrix as matrix
from faiss_imputer._matrix import MatrixNaNIndex
from benchmarks import profile_available_transform as profile


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


class RowwiseIndex(MatrixNaNIndex):
    def _refine_selected(self, queries, values, ids):
        # The final float64-selected block from the supplied baseline.
        for row in range(len(queries)):
            if row in self.precise_rows:
                continue
            valid = (ids[row] >= 0) & (ids[row] < len(self.donors64))
            values[row, ~valid] = np.inf
            if valid.any():
                selected = ids[row, valid]
                values[row, valid] = self._distances_to(
                    np.asarray(queries[row], dtype=np.float64),
                    self.donors64[selected], self.present[selected],
                )


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_empty_selected_blocks_do_not_invoke_the_kernel(monkeypatch, shape):
    index = MatrixNaNIndex(np.ones((3, 2)))
    queries = np.zeros((shape[0], 2))
    values = np.empty(shape)
    ids = np.empty(shape, dtype=np.int64)

    def unexpected(*args):
        pytest.fail("Empty selected pairs should not call the kernel")

    monkeypatch.setattr(index, "_distances_to", unexpected)
    index._refine_selected(queries, values, ids)


def test_source_contract_identifies_batched_selected_refinement():
    index = MatrixNaNIndex(np.array([[1, 2], [3, 4]], dtype=np.float64))
    assert profile.guard_search_contract(index)["name"] == profile.BATCHED_SEARCH_CONTRACT


@pytest.mark.parametrize("features", [2, 7, 11, 20, 100])
@pytest.mark.parametrize("max_pairs", [1, 13, 4096])
def test_selected_batches_preserve_pairs_and_bound_kernel_calls(
    monkeypatch, features, max_pairs,
):
    rng = np.random.default_rng(618)
    donors = rng.normal(size=(21, features))
    queries = rng.normal(size=(9, features))
    donors[rng.random(donors.shape) < 0.35] = np.nan
    queries[rng.random(queries.shape) < 0.25] = np.nan
    donors[0] = np.nan
    donors[2] = donors[1]  # Duplicates and genuine equal distances.
    queries[0] = np.nan  # No shared features, even for otherwise valid ids.
    ids = rng.integers(-3, len(donors) + 4, size=(len(queries), 17))
    ids[1, :5] = [2, 1, 2, 0, -1]  # Reordered and repeated candidates.
    ids[3] = -1
    original_ids = ids.copy()
    initial = rng.normal(size=ids.shape)
    actual, expected = initial.copy(), initial.copy()
    # Schema width differs from stored width: memory uses actual features.
    index = MatrixNaNIndex(donors, n_features=features + 5)
    reference = RowwiseIndex(donors, n_features=features + 5)
    index.precise_rows[2] = reference.precise_rows[2] = np.zeros(len(donors))
    budget = max_pairs * (64 * features + 128)
    monkeypatch.setattr(matrix, "SELECTED_DISTANCE_WORKSPACE_BYTES", budget)
    calls = []
    original_kernel = index._distances_to

    def counted(query, selected, present):
        assert query.ndim == 2 and query.shape == selected.shape
        assert query.dtype == selected.dtype == np.dtype(np.float64)
        assert 0 < len(selected) <= max_pairs
        calls.append(len(selected))
        return original_kernel(query, selected, present)

    monkeypatch.setattr(index, "_distances_to", counted)
    reference._refine_selected(queries, expected, ids.copy())
    index._refine_selected(queries, actual, ids)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(ids, original_ids)
    np.testing.assert_array_equal(actual[2], initial[2])
    assert np.isinf(actual[0]).all() and np.isinf(actual[3]).all()
    assert calls
    if max_pairs == 4096:
        assert len(calls) == 1  # Ordinary rows share one kernel invocation.


@pytest.mark.parametrize("donor_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("query_dtype", [np.float32, np.float64])
def test_search_preserves_mixed_input_dtypes(donor_dtype, query_dtype):
    rng = np.random.default_rng(81)
    donors = rng.normal(size=(23, 7)).astype(donor_dtype)
    queries = (rng.normal(size=(8, 7)) + 5).astype(query_dtype)
    index, reference = MatrixNaNIndex(donors), RowwiseIndex(donors)
    actual_values, actual_ids = index.search(queries, 9)
    expected_values, expected_ids = reference.search(queries, 9)
    expected_dtype = (
        np.float64 if np.float64 in (donor_dtype, query_dtype) else np.float32
    )
    assert actual_values.dtype == expected_dtype
    assert not index.precise_rows
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize("scale", [1e153, 1e-162])
def test_selected_kernel_keeps_overflow_and_underflow_repairs(scale):
    donors = np.array([[scale] * 20, [scale / 2] * 20])
    queries = np.zeros((3, 20))
    ids = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    actual, expected = np.empty(ids.shape), np.empty(ids.shape)
    MatrixNaNIndex(donors)._refine_selected(queries, actual, ids.copy())
    RowwiseIndex(donors)._refine_selected(queries, expected, ids.copy())
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all() and (actual > 0).all()


@pytest.mark.parametrize("scale", [1e155, 1e-200])
def test_selected_kernel_raises_the_same_unrepresentable_distance(scale):
    donors, queries = np.full((2, 2), scale), np.zeros((2, 2))
    errors = []
    for cls in (MatrixNaNIndex, RowwiseIndex):
        with pytest.raises(ValueError, match="finite and representable as float64") as error:
            cls(donors)._refine_selected(
                queries, np.empty((2, 2)), np.array([[0, 1], [1, 0]]),
            )
        errors.append(error.value.args)
    assert errors[0] == errors[1]


def test_search_ties_precise_rows_and_reordered_retained_expansion():
    donors = np.array([
        [0, 10, np.nan], [0, 20, 2], [1, np.nan, 3],
        [2, 30, np.nan], [4, np.nan, 5], [0, 40, 2],
    ])
    queries = np.array([
        [0, np.nan, 2], [1, np.nan, np.nan],
        [4, np.nan, 5], [np.nan, 25, 3],
    ])
    index, reference = MatrixNaNIndex(donors), RowwiseIndex(donors)
    for cls_index in (index, reference):
        cls_index.search(queries, 2)
        assert 0 in cls_index.precise_rows
    rows = np.array([2, 0, 3])
    retained = index.retain_queries(rows)
    retained_reference = reference.retain_queries(rows)
    assert 1 in index.precise_rows
    actual_values, actual_ids = index.search(retained, len(donors))
    expected_values, expected_ids = reference.search(retained_reference, len(donors))
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_array_equal(actual_values, expected_values)
    assert actual_ids[1, :3].tolist() == [0, 1, 5]  # True ties keep training order.
    assert index.query_ref is retained


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strategy,weights", [
    ("mean", "uniform"), ("mean", "distance"), ("median", "uniform"),
])
def test_available_model_outputs_and_expansion_match_rowwise(
    monkeypatch, dtype, strategy, weights,
):
    rng = np.random.default_rng(209)
    train = rng.normal(size=(40, 7)).astype(dtype)
    train[:24, 1] = np.nan
    train[24:] += 30
    queries = train[[0, 39, 38, 37]].copy()
    queries[0, 1], queries[1, 0] = np.nan, np.nan
    queries[2] = np.nan
    options = dict(
        donor_policy="available", n_neighbors=3,
        strategy=strategy, weights=weights,
    )
    model, reference = FaissImputer(**options).fit(train), FaissImputer(**options).fit(train)
    reference.available_index_ = RowwiseIndex(
        reference.donors_, n_features=reference.n_features_in_,
    )
    retained_rows = []
    original_retain = model.available_index_.retain_queries

    def counted_retain(rows):
        retained_rows.append(rows.copy())
        return original_retain(rows)

    monkeypatch.setattr(model.available_index_, "retain_queries", counted_retain)
    actual, expected = model.transform(queries), reference.transform(queries)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype and retained_rows
    permutation = np.array([1, 0, 3, 2])
    np.testing.assert_array_equal(model.transform(queries[permutation]), actual[permutation])
    for fitted in (model, reference):
        index = fitted.available_index_
        assert index.query_ref is None and index.matrix is None
        assert not index.precise_rows
