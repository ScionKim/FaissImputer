"""Regression checks for available-donor batching; no product patching."""

import ast
import inspect
import re
import textwrap

import numpy as np
import pytest
import sklearn.metrics.pairwise as pairwise
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
from benchmarks.benchmark_scaling_threads import make_data


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def variant(budget=128, batch_rows=None):
    original = FaissImputer._transform_available_batched
    source = textwrap.dedent(inspect.getsource(original))
    source, changed = re.subn(
        r"\(\s*\d+\s*\*\s*1024\s*\*\s*1024\s*\)",
        f"({budget} * 1024 * 1024)",
        source,
    )
    assert changed == 1, "Review tests after changing the batch formula"
    tree = ast.parse(source)
    if batch_rows is not None:
        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "batch_size"
                    for t in node.targets)
        ]
        assert len(assignments) == 1
        assignments[0].value = ast.Constant(value=batch_rows)
    namespace = original.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree), "<batch-regression>", "exec"), namespace)
    return type(
        f"Batch{budget}Rows{batch_rows}",
        (FaissImputer,),
        {"_transform_available_batched": namespace[original.__name__]},
    )


def reference(train, query, neighbors, strategy):
    # Distances use the float32 values actually accepted by the product.
    train = np.asarray(train, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    donor64 = train.astype(np.float64)
    present = ~np.isnan(train)
    result = query.copy()
    aggregate = np.mean if strategy == "mean" else np.median
    statistics = np.array([
        aggregate(train[present[:, col], col])
        for col in range(train.shape[1])
    ], dtype=np.float32)

    for row, values in enumerate(query):
        observed = ~np.isnan(values)
        shared = present & observed
        counts = shared.sum(axis=1)
        delta = np.where(shared, donor64 - values.astype(np.float64), 0.0)
        distances = np.full(len(train), np.inf)
        usable = counts > 0
        distances[usable] = (
            np.sum(delta[usable] ** 2, axis=1)
            * train.shape[1] / counts[usable]
        )
        for col in np.flatnonzero(~observed):
            eligible = np.flatnonzero(usable & present[:, col])
            if eligible.size:
                order = np.argsort(distances[eligible], kind="stable")
                selected = eligible[order[:neighbors]]
                result[row, col] = aggregate(train[selected, col])
            else:
                result[row, col] = statistics[col]
    return result


def transform_checked(cls, train, query, neighbors=1, strategy="mean"):
    train_before = train.copy()
    query_before = query.copy()
    model = cls(
        n_neighbors=neighbors, donor_policy="available", strategy=strategy
    ).fit(train)
    result = model.transform(query)
    assert result.shape == query.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert not np.shares_memory(result, query)
    observed = ~np.isnan(query)
    np.testing.assert_array_equal(result[observed], query[observed])
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(model.transform(query), result)
    assert model.available_index_.query_ref is None
    assert model.available_index_.matrix is None
    return result


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("neighbors", [1, 3, 20])
@pytest.mark.parametrize("batch_rows", [1, 2, 3])
def test_mixed_rows_match_direct_reference(strategy, neighbors, batch_rows):
    train = np.array([
        [0, 10, np.nan, 2],
        [1, 20, 30, np.nan],
        [2, 40, np.nan, 5],
        [3, 60, 70, 8],
        [np.nan, 50, 90, 10],
        [4, np.nan, 80, 9],
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    query = np.array([
        [0.1, np.nan, np.nan, np.nan],
        [np.nan, 12, np.nan, 3],
        [0, 10, np.nan, 2],
        [np.nan, np.nan, np.nan, np.nan],
        [2.2, 44, 55, 6],
        [np.nan, np.nan, 80, 9.1],
        [1.4, np.nan, 36, np.nan],
        [np.nan, 50, 91, np.nan],
        [4.2, np.nan, np.nan, 9],
    ], dtype=np.float32)
    expected = reference(train, query, neighbors, strategy)
    for cls in (FaissImputer, variant(batch_rows=batch_rows)):
        result = transform_checked(cls, train, query, neighbors, strategy)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_no_shared_features_use_fitted_statistics(strategy):
    train = np.array([
        [np.nan, 10, 0], [np.nan, 30, 1], [2, np.nan, np.nan]
    ], dtype=np.float32)
    query = np.array([
        [0, np.nan, np.nan],
        [np.nan, 11, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    expected = reference(train, query, 1, strategy)
    for cls in (FaissImputer, variant(batch_rows=2)):
        result = transform_checked(cls, train, query, strategy=strategy)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("far,near,origin", [
    (2e-23, 1e-23, 0.0),
    (1.0001, 1.0, 0.0),
    (1e8 + 16, 1e8 + 8, 1e8),
    (2e20, 1e20, 0.0),
])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("batch_rows", [1, 3])
def test_unique_nearest_donor_at_different_scales(
    far, near, origin, reverse, batch_rows
):
    donors = [[far, 10, np.nan], [near, 20, np.nan]]
    if reverse:
        donors.reverse()
    train = np.array(donors + [[np.nan, np.nan, 0]], dtype=np.float32)
    query = np.array([
        [origin, np.nan, 0],
        [near, np.nan, 0],
        [np.nan, np.nan, np.nan],
        [origin, np.nan, np.nan],
    ], dtype=np.float32)
    expected = reference(train, query, 1, "mean")
    np.testing.assert_array_equal(expected[[0, 1, 3], 1], [20, 20, 20])
    for cls in (FaissImputer, variant(batch_rows=batch_rows)):
        result = transform_checked(cls, train, query)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("batch_rows", [1, 2])
def test_same_row_has_same_correct_donor_when_another_row_triggers_float64(
    reverse, batch_rows
):
    train = np.array([
        [-1.0, 10, np.nan],
        [1.0, 20, np.nan],
        [np.nan, np.nan, 0],
    ], dtype=np.float32)

    target = np.array([
        [1e-8, np.nan, 0],
    ], dtype=np.float32)

    rows = [
        [1e-8, np.nan, 0],
        [1.0, np.nan, 0],
    ]
    if reverse:
        rows.reverse()

    mixed = np.array(rows, dtype=np.float32)
    target_row = 1 if reverse else 0

    expected_target = reference(train, target, 1, "mean")
    expected_mixed = reference(train, mixed, 1, "mean")

    assert expected_target[0, 1] == 20
    assert expected_mixed[target_row, 1] == 20

    cls = variant(batch_rows=batch_rows)

    alone = transform_checked(cls, train, target)
    together = transform_checked(cls, train, mixed)

    assert (alone[0, 1], together[target_row, 1]) == (20, 20)

@pytest.mark.parametrize("batch_rows", [1, 2, 3])
def test_exact_ties_select_an_eligible_donor(batch_rows):
    train = np.array([
        [-1, 10, np.nan], [1, 20, np.nan], [np.nan, np.nan, 0]
    ], dtype=np.float32)
    query = np.array([
        [0, np.nan, 0], [1, np.nan, 0], [np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    for cls in (FaissImputer, variant(batch_rows=batch_rows)):
        result = transform_checked(cls, train, query)
        # No particular ordering is promised between exactly tied donors.
        assert result[0, 1] in (10, 20)
        assert result[1, 1] == 20
        assert result[2, 1] == 15

@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_real_128mib_variant_changes_batch_boundaries(monkeypatch, strategy):
    train, query, _, _ = make_data(50000, 32, 101, "random")
    from faiss_imputer._matrix import MatrixNaNIndex

    original_search = MatrixNaNIndex.search
    calls = []

    def counted(self, queries, k):
        if queries is not self.query_ref and len(self.donors64) == len(train):
            calls.append(len(queries))
        return original_search(self, queries, k)

    monkeypatch.setattr(MatrixNaNIndex, "search", counted)
    results = {}
    for budget, expected_batches in ((16, [27, 5]), (128, [32])):
        calls.clear()
        model = variant(budget=budget)(
            n_neighbors=5, donor_policy="available", strategy=strategy
        ).fit(train)
        results[budget] = model.transform(query)
        assert calls == expected_batches
    np.testing.assert_allclose(results[128], results[16], rtol=1e-6, atol=1e-7)
    expected = reference(train, query, 5, strategy)
    np.testing.assert_allclose(results[16], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(results[128], expected, rtol=1e-6, atol=1e-7)
    calls.clear()
    default_result = FaissImputer(
        n_neighbors=5,
        donor_policy="available",
        strategy=strategy,
    ).fit(train).transform(query)
    assert calls == [32]
    np.testing.assert_allclose(default_result, expected, rtol=1e-6, atol=1e-7)
