"""Offline regressions for released real-data neighbor-weight selection."""

from statistics import median

import numpy as np
import pytest

from benchmarks import benchmark_real_data_worker as worker
from benchmarks import benchmark_released_real_data as benchmark


@pytest.mark.parametrize("dataset_id", benchmark.DATASETS)
@pytest.mark.parametrize("dtype", benchmark.DTYPES)
def test_weighted_configs_keep_inputs_and_worker_order(tmp_path, dataset_id, dtype):
    options = {"dataset_id": dataset_id, "dtype": dtype}
    legacy = benchmark.build_configs("0.3.21", "0.3.22", tmp_path, **options)
    explicit = benchmark.build_configs(
        "0.3.21", "0.3.22", tmp_path, weights="uniform", **options
    )
    distance = benchmark.build_configs(
        "0.3.21", "0.3.22", tmp_path, weights="distance", **options
    )
    assert len(legacy) == len(distance) == 27
    assert explicit == legacy
    assert all("weights" not in config for config in legacy)
    for original, weighted in zip(legacy, distance):
        assert weighted["weights"] == "distance"
        assert {key: value for key, value in weighted.items()
                if key != "weights"} == original


@pytest.mark.parametrize("weights", [None, "inverse_squared", "both"])
def test_unsupported_weights_are_rejected(tmp_path, weights):
    with pytest.raises(ValueError, match="Unsupported weights"):
        benchmark.build_configs("0.3.21", "0.3.22", tmp_path, weights=weights)
    with pytest.raises(ValueError, match="Unsupported weights"):
        worker.make_model("KNNImputer", weights=weights)


@pytest.mark.parametrize(
    "method", ["KNNImputer", "FaissImputer[complete]", "FaissImputer[available]"]
)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_real_estimators_receive_the_selected_weights(method, weights):
    model = worker.make_model(method, weights=weights)
    assert model.get_params(deep=False)["weights"] == weights
    assert worker.make_model(method).get_params(deep=False)["weights"] == "uniform"


@pytest.mark.parametrize("method", ["SimpleImputer[mean]", "SimpleImputer[median]"])
def test_simple_baselines_cannot_be_labeled_distance_weighted(method):
    assert "weights" not in worker.make_model(method).get_params(deep=False)
    with pytest.raises(ValueError, match="do not support neighbor weights"):
        worker.make_model(method, weights="distance")


@pytest.mark.parametrize("method", ["KNNImputer", "FaissImputer[available]"])
@pytest.mark.parametrize("requested_weights", [None, "uniform", "distance"])
def test_worker_uses_selected_weights_for_warmup_and_measurement(
    monkeypatch, method, requested_weights
):
    train = np.arange(12, dtype=np.float64).reshape(6, 2)
    query = np.array([[0.0, np.nan]])
    truth = np.array([[0.0, 3.0]])
    missing = np.isnan(query)
    names = ["Length", "target"]
    case = {"feature_names": names, "scaler_scale": [1.0, 1.0],
            "complete_donors": len(train)}
    seen = []

    class RecordingModel:
        def __init__(self, weights):
            self.weights = weights

        def fit(self, data):
            return self

        def transform(self, data):
            result = data.copy()
            result[np.isnan(result)] = 2.0 if self.weights == "distance" else 1.0
            return result

        def get_params(self, deep=False):
            return {"weights": self.weights}

    def factory(method, *, weights="uniform"):
        seen.append(weights)
        return RecordingModel(weights)

    # A legacy caller may provide a factory accepting only the method argument.
    selected_factory = (
        (lambda method: factory(method)) if requested_weights is None else factory
    )
    monkeypatch.setattr(worker, "make_model", selected_factory)
    monkeypatch.setattr(worker, "check_released_package", lambda version: None)
    monkeypatch.setattr(worker, "metadata", lambda: {})
    monkeypatch.setattr(worker, "peak_rss_mib", lambda: 64.0)
    monkeypatch.setattr(worker, "threadpool_info", lambda: [{"num_threads": 1}])
    monkeypatch.setattr(
        worker, "load_dataset", lambda *args, **kwargs: (train, names, {})
    )
    monkeypatch.setattr(
        worker, "prepare_case",
        lambda *args, **kwargs: (train, query, truth, missing, case),
    )
    config = {"method": method, "dataset_id": "abalone", "data_home": ".",
              "seed": 101, "mechanism": "MCAR", "train_size": 6,
              "query_size": 1, "dtype": "float64"}
    if requested_weights is not None:
        config["weights"] = requested_weights
    result = worker.worker(config)
    expected = requested_weights or "uniform"
    assert result["status"] == "ok" and result["checks_passed"] is True
    assert seen == [expected, expected]
    assert result["model_parameters"]["weights"] == expected
    assert result["_values"] == ([2.0] if expected == "distance" else [1.0])
    if requested_weights is None:
        assert "weights" not in result
    else:
        assert result["weights"] == requested_weights


@pytest.fixture
def records(tmp_path):
    configs = benchmark.build_configs("0.3.21", "0.3.22", tmp_path)
    totals = {"knn": (3.0, 30.0, 60.0), "previous": (2.0, 4.0, 80.0),
              "current": (1.0, 20.0, 40.0)}
    rows = []
    for index, config in enumerate(configs):
        total = totals[config["variant"]][config["repeat"] - 1]
        rows.append({
            **config, "record_index": index, "status": "ok", "checks_passed": True,
            "case": {"seed": config["seed"]}, "imputed_values": [1.0, 2.0],
            "output_sha256": "a" * 64, "fit_seconds": total / 4,
            "transform_seconds": total * 3 / 4, "total_seconds": total,
            "worker_peak_rss_mib": 64.0, "rmse": 0.1, "mae": 0.08,
        })
    return rows


def test_legacy_uniform_aggregation_and_match_dictionaries_are_preserved(records):
    explicit = [{**row, "weights": "uniform"} for row in records]
    legacy_comparisons = benchmark.compare_records(records)
    assert benchmark.compare_records(explicit) == legacy_comparisons
    assert benchmark.summarize(explicit) == benchmark.summarize(records)
    for comparison in legacy_comparisons:
        assert comparison["matched_pairs"] == 9
        for pair in comparison["pairs"]:
            source = records[pair["numerator_record_index"]]
            assert pair["match"] == {
                name: source[name] for name in benchmark.MATCH_FIELDS
            }


def test_distance_results_use_matched_ratios_and_separate_seed_quality(records):
    distance = [{**row, "weights": "distance"} for row in records]
    comparison = next(
        item for item in benchmark.compare_records(distance)
        if item["numerator_variant"] == "previous"
    )
    paired = median(pair["timing_ratios"]["total_seconds"]
                    for pair in comparison["pairs"])
    ratio_of_medians = (
        median(row["total_seconds"] for row in distance if row["variant"] == "previous")
        / median(row["total_seconds"] for row in distance if row["variant"] == "current")
    )
    assert paired == 2.0 and paired != ratio_of_medians
    assert comparison["timing_ratios"]["total_seconds"]["median"] == paired
    assert comparison["matched_pairs"] == 9
    assert all(pair["match"]["weights"] == "distance" for pair in comparison["pairs"])
    for summary in benchmark.summarize(distance):
        assert summary["successful_records"] == 9
        assert summary["quality_seed_count"] == 3


def test_mixed_weights_cannot_be_pooled(records):
    distance = [{**row, "weights": "distance",
                 "record_index": row["record_index"] + len(records)} for row in records]
    with pytest.raises(ValueError, match="Cannot pool"):
        benchmark.summarize(records + distance)
    with pytest.raises(ValueError, match="Cannot pool"):
        benchmark.compare_records(records + distance)


def test_different_weights_are_not_matched(records):
    for row in records:
        if row["variant"] == "previous":
            row["weights"] = "distance"
    comparisons = benchmark.compare_records(records)
    assert [item["matched_pairs"] for item in comparisons] == [0, 9, 0]


@pytest.mark.parametrize("status,checks_passed", [("error", True), ("ok", False)])
def test_failed_distance_records_do_not_contaminate_uniform_results(
    records, status, checks_passed
):
    failed = [{**row, "weights": "distance", "status": status,
               "checks_passed": checks_passed} for row in records]
    assert benchmark.summarize(records + failed) == benchmark.summarize(records)
    assert benchmark.compare_records(records + failed) == benchmark.compare_records(records)


def test_mixed_explicit_and_implicit_uniform_duplicates_are_rejected(records):
    duplicate = {**records[0], "weights": "uniform"}
    with pytest.raises(ValueError, match="Duplicate successful"):
        benchmark.compare_records(records + [duplicate])


def test_missing_worker_weights_cannot_mask_a_distance_request():
    with pytest.raises(ValueError, match="Worker weights differ"):
        benchmark.validate_record(
            {"checks_passed": True}, {"weights": "distance"}, {}, {}
        )


def test_reported_weights_cannot_mask_an_unweighted_model(tmp_path):
    config = benchmark.build_configs(
        "0.3.21", "0.3.22", tmp_path, weights="distance"
    )[0]
    assert config["variant"] == "knn"
    environment = {name: "same" for name in
                   (*benchmark.COMMON_ENVIRONMENT, "platform", "cpu_model", "git_commit")}
    record = {
        "checks_passed": True, "weights": "distance",
        "environment": {**environment, "faiss_imputer": config["expected_version"]},
        "dataset": {}, "input_dtype": config["dtype"], "output_dtype": config["dtype"],
        "threads": 1, "faiss_omp_threads": 1, "threadpools": [{"num_threads": 1}],
        "sklearn_working_memory_mib": benchmark.WORKING_MEMORY_MIB,
        "model_parameters": {
            "n_neighbors": benchmark.N_NEIGHBORS, "weights": "uniform",
            "metric": "nan_euclidean", "copy": True,
        },
    }
    with pytest.raises(ValueError, match="Unexpected model parameters"):
        benchmark.validate_record(record, config, environment, {})
