import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from benchmarks import benchmark_real_data as benchmark


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_real_data_preparation(mechanism):
    dataset = load_diabetes(scaled=False)
    data = dataset.data.copy()
    before = data.copy()
    names = list(dataset.feature_names)

    first = benchmark.prepare_case(data, names, 101, mechanism)
    second = benchmark.prepare_case(data, names, 101, mechanism)
    for left, right in zip(first[:4], second[:4]):
        np.testing.assert_array_equal(left, right)
    assert first[4] == second[4]
    np.testing.assert_array_equal(data, before)

    train, query, truth, missing, metadata = first
    train_raw, query_raw = train_test_split(
        data, test_size=0.25, random_state=101
    )
    age = names.index("age")
    eligible = [
        i for i, name in enumerate(names)
        if name not in ("age", "sex")
    ]
    cutoff = np.median(train_raw[:, age])
    rng = np.random.default_rng(10101)
    expected_masks = []
    for raw in (train_raw, query_raw):
        probability = np.full(len(raw), 0.25)
        if mechanism == "MAR":
            probability = np.where(raw[:, age] > cutoff, 0.375, 0.125)
        mask = np.zeros(raw.shape, dtype=bool)
        mask[:, eligible] = (
            rng.random((len(raw), len(eligible))) < probability[:, None]
        )
        expected_masks.append(mask)

    np.testing.assert_array_equal(np.isnan(train), expected_masks[0])
    np.testing.assert_array_equal(np.isnan(query), expected_masks[1])
    np.testing.assert_array_equal(missing, expected_masks[1])

    observed_train = train_raw.copy()
    observed_train[expected_masks[0]] = np.nan
    expected_mean = np.nanmean(observed_train, axis=0)
    expected_scale = np.nanstd(observed_train, axis=0)
    np.testing.assert_allclose(metadata["scaler_mean"], expected_mean)
    np.testing.assert_allclose(metadata["scaler_scale"], expected_scale)
    np.testing.assert_allclose(
        truth,
        ((query_raw - expected_mean) / expected_scale).astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(query[~missing], truth[~missing])
    for array in (train, query, truth):
        assert array.dtype == np.float32


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_real_data_methods_smoke(mechanism):
    dataset = load_diabetes(scaled=False)
    with threadpool_limits(limits=1):
        case = benchmark.run_case(
            dataset.data,
            list(dataset.feature_names),
            seed=101,
            mechanism=mechanism,
            repeats=1,
        )

    assert set(case["methods"]) == set(benchmark.METHODS)
    scored_cells = sum(case["query_mask"]["missing_per_feature"])
    assert scored_cells > 0
    for record in case["methods"].values():
        assert record["status"] == "ok"
        assert record["checks_passed"]
        assert record["quality"]["scored_cells"] == scored_cells
        for metric in ("rmse", "mae", "max_abs_difference_from_knn"):
            value = record["quality"][metric]
            assert np.isfinite(value) and value >= 0
        assert len(record["samples"]) == 1
        sample = record["samples"][0]
        for field in ("fit_seconds", "transform_seconds", "total_seconds"):
            assert np.isfinite(sample[field]) and sample[field] >= 0
            assert record["timing"][field]["median"] == sample[field]
