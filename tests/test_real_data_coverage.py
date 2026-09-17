"""Offline regression checks for held-out real-data benchmarks."""

from copy import deepcopy

import numpy as np
import pytest

from benchmarks import benchmark_real_data_cases as cases
from benchmarks import benchmark_real_data_coverage as coverage
from benchmarks import benchmark_real_data_worker as worker_module


NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@pytest.fixture
def source_data():
    rng = np.random.default_rng(71)
    data = rng.normal(size=(192, len(NAMES)))
    data *= np.arange(1, len(NAMES) + 1)
    data[:, 0] = np.linspace(1, 12, len(data))
    data.setflags(write=False)
    return data


@pytest.fixture(autouse=True)
def prevent_dataset_download(monkeypatch):
    def reject_download(*args, **kwargs):
        raise AssertionError("Unit tests must not download datasets")

    monkeypatch.setattr(
        cases, "fetch_california_housing", reject_download
    )


def prepare(data, mechanism="MCAR", dtype="float32", train_size=80):
    return cases.prepare_case(
        data,
        NAMES,
        seed=101,
        mechanism=mechanism,
        train_size=train_size,
        query_size=32,
        dtype=dtype,
        mar_reference_rows=40,
    )


def worker_config(
    method="SimpleImputer[mean]",
    dtype="float32",
    mechanism="MCAR",
):
    return {
        "method": method,
        "train_size": 80,
        "query_size": 32,
        "seed": 101,
        "mechanism": mechanism,
        "dtype": dtype,
        "repeat": 1,
        "mar_reference_rows": 40,
        "data_home": "unused-test-cache",
    }


@pytest.fixture
def cached_data(monkeypatch, source_data):
    dataset = {
        "dataset": "synthetic test fixture",
        "feature_names": NAMES.copy(),
        "dataset_sha256": cases.array_digest(source_data),
    }

    def load_cached(data_home, *, download_if_missing):
        assert data_home == "unused-test-cache"
        assert download_if_missing is False
        return source_data, NAMES.copy(), deepcopy(dataset)

    monkeypatch.setattr(worker_module, "load_dataset", load_cached)
    return source_data


def test_digest_uses_logical_values_independent_of_memory_layout():
    array = np.arange(24, dtype=np.float64).reshape(4, 6)
    fortran = np.asfortranarray(array)
    strided = np.repeat(array, 2, axis=1)[:, ::2]

    assert cases.array_digest(array) == cases.array_digest(fortran)
    assert cases.array_digest(array) == cases.array_digest(strided)
    assert cases.array_digest(array) != cases.array_digest(
        array.astype(np.float32)
    )


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_preparation_is_reproducible_and_preserves_source(
    source_data, mechanism, dtype
):
    before = source_data.copy()
    first = prepare(source_data, mechanism, dtype)
    second = prepare(source_data, mechanism, dtype)

    for left, right in zip(first[:4], second[:4]):
        np.testing.assert_array_equal(left, right)
    assert first[4] == second[4]
    np.testing.assert_array_equal(source_data, before)

    train, query, truth, missing, metadata = first

    assert train.dtype == np.dtype(dtype)
    assert query.dtype == np.dtype(dtype)
    assert truth.dtype == np.float64
    assert missing.dtype == np.bool_
    assert not np.isnan(train[:, 0]).any()
    assert not missing[:, 0].any()
    assert missing.any()
    assert not np.shares_memory(train, source_data)
    assert not np.shares_memory(query, source_data)

    np.testing.assert_array_equal(np.isnan(query), missing)
    np.testing.assert_array_equal(
        query[~missing],
        truth.astype(dtype)[~missing],
    )

    order = np.random.default_rng([101, 0]).permutation(len(source_data))
    query_ids = order[:32]
    train_ids = order[32:112]

    assert set(query_ids).isdisjoint(train_ids)
    assert metadata["fingerprints"]["query_row_ids"] == (
        cases.array_digest(query_ids)
    )
    assert metadata["fingerprints"]["train_row_ids"] == (
        cases.array_digest(train_ids)
    )


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_training_sizes_share_raw_queries_and_nested_training_rows(
    source_data, mechanism
):
    small = prepare(source_data, mechanism, train_size=80)
    large = prepare(source_data, mechanism, train_size=120)

    small_metadata = small[4]
    large_metadata = large[4]

    for name in ("query_row_ids", "raw_query", "query_mask"):
        assert small_metadata["fingerprints"][name] == (
            large_metadata["fingerprints"][name]
        )

    np.testing.assert_array_equal(small[3], large[3])
    np.testing.assert_array_equal(
        np.isnan(small[0]),
        np.isnan(large[0][:80]),
    )
    assert small_metadata["mar_cutoff"] == large_metadata["mar_cutoff"]

    order = np.random.default_rng([101, 0]).permutation(len(source_data))
    assert small_metadata["fingerprints"]["raw_train"] == (
        cases.array_digest(source_data[order[32:112]])
    )
    assert large_metadata["fingerprints"]["raw_train"] == (
        cases.array_digest(source_data[order[32:152]])
    )


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_scoring_truth_is_float64_for_both_input_dtypes(
    source_data, mechanism
):
    single = prepare(source_data, mechanism, "float32")
    double = prepare(source_data, mechanism, "float64")

    np.testing.assert_array_equal(single[2], double[2])
    np.testing.assert_array_equal(single[3], double[3])
    np.testing.assert_array_equal(single[0], double[0].astype(np.float32))
    np.testing.assert_array_equal(single[1], double[1].astype(np.float32))

    assert single[2].dtype == double[2].dtype == np.float64
    assert single[4]["scaler_mean"] == double[4]["scaler_mean"]
    assert single[4]["scaler_scale"] == double[4]["scaler_scale"]


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_query_and_excluded_rows_do_not_change_training_statistics(
    source_data, mechanism
):
    original = prepare(source_data, mechanism, "float64")
    order = np.random.default_rng([101, 0]).permutation(len(source_data))

    changed = source_data.copy()
    changed[order[:32], 1:] += 1_000
    changed[order[112:], :] -= 2_000
    modified = prepare(changed, mechanism, "float64")

    np.testing.assert_array_equal(original[0], modified[0])
    np.testing.assert_array_equal(original[3], modified[3])

    for name in ("scaler_mean", "scaler_scale", "mar_cutoff"):
        assert original[4][name] == modified[4][name]

    assert not np.array_equal(original[2], modified[2])

    observed_train = source_data[order[32:112]].copy()
    observed_train[np.isnan(original[0])] = np.nan

    expected_mean = np.nanmean(observed_train, axis=0)
    expected_scale = np.nanstd(observed_train, axis=0)

    np.testing.assert_allclose(
        original[4]["scaler_mean"], expected_mean, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        original[4]["scaler_scale"], expected_scale, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        original[2],
        (source_data[order[:32]] - expected_mean) / expected_scale,
        rtol=1e-12,
        atol=1e-12,
    )


def test_quality_scores_only_hidden_cells_and_restores_feature_units():
    truth = np.zeros((2, 3), dtype=np.float64)
    output = np.array([[99, 3, 4], [99, -1, 99]], dtype=np.float64)
    missing = np.array([[False, True, True], [False, True, False]])
    case = {
        "feature_names": ["MedInc", "HouseAge", "AveRooms"],
        "scaler_scale": [2, 10, 0.5],
    }

    quality, values = worker_module.quality_summary(
        output, truth, missing, case
    )

    np.testing.assert_array_equal(values, [3, 4, -1])
    assert quality["scored_cells"] == 3
    assert quality["rmse"] == pytest.approx(np.sqrt(26 / 3))
    assert quality["mae"] == pytest.approx(8 / 3)

    income, age, rooms = quality["feature_quality"]
    assert income["scored_cells"] == 0
    for name in (
        "rmse_standardized",
        "mae_standardized",
        "rmse_original_units",
        "mae_original_units",
    ):
        assert income[name] is None

    assert age["scored_cells"] == 2
    assert age["rmse_standardized"] == pytest.approx(np.sqrt(5))
    assert age["mae_standardized"] == pytest.approx(2)
    assert age["rmse_original_units"] == pytest.approx(10 * np.sqrt(5))
    assert age["mae_original_units"] == pytest.approx(20)
    assert rooms["rmse_original_units"] == pytest.approx(2)
    assert rooms["mae_original_units"] == pytest.approx(2)


@pytest.mark.parametrize("method", worker_module.METHODS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_worker_runs_offline_and_scores_float64_truth(
    cached_data, method, dtype, mechanism
):
    config = worker_config(method, dtype, mechanism)
    result = worker_module.worker(config)
    _, _, truth, missing, case = prepare(cached_data, mechanism, dtype)

    assert result["status"] == "ok"
    assert result["checks_passed"] is True
    assert result["input_dtype"] == result["output_dtype"] == dtype
    assert result["case"] == case
    assert result["faiss_omp_threads"] == 1
    assert all(
        pool["num_threads"] == 1
        for pool in result["threadpool_info"]
    ) if "threadpool_info" in result else result["threads"] == 1

    values = np.asarray(result["_values"], dtype=np.float64)
    errors = values - truth[missing]
    assert values.shape == (int(missing.sum()),)
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert result["total_seconds"] == pytest.approx(
        result["fit_seconds"] + result["transform_seconds"]
    )


@pytest.fixture
def successful_record(cached_data):
    config = worker_config()
    payload = worker_module.worker(config)
    values = np.asarray(payload.pop("_values"), dtype=np.float64)
    record = {**payload, **config, "agreement_with_knn": None}
    return record, values


def validate(record, values, references):
    coverage.validate_record(
        record,
        values,
        record["environment"],
        record["dataset"],
        *references,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("checks_passed", False),
        ("output_dtype", "float64"),
        ("total_seconds", -1),
    ],
)
def test_rejected_record_does_not_become_a_reference(
    successful_record, field, value
):
    record, values = successful_record
    record = deepcopy(record)
    record[field] = value
    references = ({}, {}, {})

    with pytest.raises((ValueError, AssertionError)):
        validate(record, values, references)

    assert all(not reference for reference in references)


@pytest.mark.parametrize("change", ["case", "output"])
def test_changed_repetition_preserves_first_valid_reference(
    successful_record, change
):
    record, values = successful_record
    references = ({}, {}, {})
    validate(record, values, references)

    changed = deepcopy(record)
    changed["repeat"] = 2
    changed_values = values.copy()
    if change == "case":
        changed["case"]["scaler_mean"][0] += 1
    else:
        changed_values[0] += 1
        changed["output_sha256"] = cases.array_digest(changed_values)

    with pytest.raises((ValueError, AssertionError)):
        validate(changed, changed_values, references)

    key = coverage.case_key(record)
    assert references[0][key] == record["case"]
    np.testing.assert_array_equal(
        references[2][key + (record["method"],)]["values"],
        values,
    )


def test_different_methods_may_produce_different_outputs(successful_record):
    record, values = successful_record
    references = ({}, {}, {})
    validate(record, values, references)

    other = deepcopy(record)
    other["method"] = "KNNImputer"
    other_values = values + 0.25
    other["output_sha256"] = cases.array_digest(other_values)
    validate(other, other_values, references)

    assert len(references[2]) == 2


def test_late_knn_result_updates_only_successful_records(successful_record):
    record, values = successful_record
    key = coverage.case_key(record)
    references = {key + (record["method"],): {"values": values}}

    failed = deepcopy(record)
    failed.update(status="validation_error", checks_passed=False, repeat=2)
    records = [record, failed]

    coverage.update_agreement(key, records, references)
    assert record["agreement_with_knn"] is None

    knn = deepcopy(record)
    knn["method"] = "KNNImputer"
    records.append(knn)
    references[key + ("KNNImputer",)] = {"values": values + 0.25}

    coverage.update_agreement(key, records, references)

    assert record["agreement_with_knn"]["max_abs_difference"] == (
        pytest.approx(0.25)
    )
    assert knn["agreement_with_knn"]["max_abs_difference"] == 0
    assert failed["agreement_with_knn"] is None


def test_summary_excludes_failed_measurements(successful_record):
    record, _ = successful_record
    failed = deepcopy(record)
    failed.update(status="timeout", checks_passed=False, repeat=2)
    configs = [{**record, "repeat": repeat} for repeat in (1, 2, 3)]

    for name in coverage.MEASURES:
        failed[name] = 1_000_000
    first = coverage.summarize([record, failed], configs)

    for name in coverage.MEASURES:
        failed[name] = 1_000_000_000
    second = coverage.summarize([record, failed], configs)

    assert first == second
    assert len(first) == 1
    assert first[0]["planned_workers"] == 3
    assert first[0]["recorded_workers"] == 2
    assert first[0]["successful_workers"] == 1
    assert first[0]["pending_workers"] == 1
    assert first[0]["status_counts"] == {"ok": 1, "timeout": 1}