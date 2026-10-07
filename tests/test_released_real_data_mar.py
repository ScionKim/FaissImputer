"""Offline regressions for released-package MAR benchmark selection."""

from copy import deepcopy
import json

import numpy as np
import pytest

from benchmarks import benchmark_real_data_cases as cases
from benchmarks import benchmark_real_data_worker as worker
from benchmarks import benchmark_released_real_data as benchmark


@pytest.mark.parametrize("dataset_id", benchmark.DATASETS)
@pytest.mark.parametrize("dtype", benchmark.DTYPES)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_mar_selection_preserves_the_grid_and_default_mcar_shape(
    tmp_path, dataset_id, dtype, weights
):
    options = {"dataset_id": dataset_id, "dtype": dtype, "weights": weights}
    legacy = benchmark.build_configs("0.3.21", "0.3.22", tmp_path, **options)
    explicit = benchmark.build_configs(
        "0.3.21", "0.3.22", tmp_path, mechanism="MCAR", **options
    )
    mar = benchmark.build_configs(
        "0.3.21", "0.3.22", tmp_path, mechanism="MAR", **options
    )
    assert explicit == legacy
    assert len(mar) == len(legacy) == 27
    for original, selected in zip(legacy, mar):
        assert original["mechanism"] == "MCAR"
        assert selected == {**original, "mechanism": "MAR"}
        defaults = cases.DATASET_DEFAULTS[dataset_id]
        assert selected["mar_driver"] == defaults["mar_driver"]
        assert selected["mar_reference_rows"] == 1000
        if weights == "uniform":
            assert "weights" not in selected


@pytest.mark.parametrize("mechanism", [None, "mar", "MNAR", "both"])
def test_unsupported_mechanisms_are_rejected(tmp_path, mechanism):
    with pytest.raises(ValueError, match="(?i)mechanism"):
        benchmark.build_configs(
            "0.3.21", "0.3.22", tmp_path, mechanism=mechanism
        )


@pytest.fixture
def run_prepared_worker(monkeypatch, tmp_path):
    """Keep real splitting, masking, scaling and scoring; replace external inputs."""
    def run(mechanism="MAR", dataset_id="abalone", dtype="float64", data=None):
        config = next(
            item for item in benchmark.build_configs(
                "0.3.21", "0.3.22", tmp_path,
                dataset_id=dataset_id, dtype=dtype, weights="distance",
                mechanism=mechanism,
            )
            if item["variant"] == "knn"
        )
        config.update(train_size=160, query_size=64, mar_reference_rows=80)
        names = list(cases.UCI_DATASETS[dataset_id]["feature_names"])
        driver = names.index(config["mar_driver"])
        if data is None:
            data = np.random.default_rng(7).normal(size=(256, len(names)))
            data[:, driver] = np.linspace(-4.0, -1.0, len(data))
        dataset = {"dataset": "offline synthetic features", "feature_names": names,
                   "dataset_sha256": cases.array_digest(data)}
        environment = {
            name: "same" for name in
            (*benchmark.COMMON_ENVIRONMENT, "platform", "cpu_model", "git_commit")
        }
        environment["faiss_imputer"] = config["expected_version"]
        fitted = []

        class RecordingImputer:
            def fit(self, train):
                fitted.append(train.copy())
                return self

            def transform(self, query):
                output = query.copy()
                output[np.isnan(output)] = 0.0
                return output

            def get_params(self, deep=False):
                return {"n_neighbors": benchmark.N_NEIGHBORS, "weights": "distance",
                        "metric": "nan_euclidean", "copy": True}

        def load_dataset(data_home, *, dataset_id, download_if_missing):
            assert dataset_id == config["dataset_id"]
            assert download_if_missing is False
            return data, names, dataset

        def make_model(method, *, weights):
            assert method == "KNNImputer" and weights == "distance"
            return RecordingImputer()

        monkeypatch.setattr(worker, "load_dataset", load_dataset)
        monkeypatch.setattr(worker, "make_model", make_model)
        monkeypatch.setattr(worker, "check_released_package", lambda version: None)
        monkeypatch.setattr(worker, "metadata", lambda: environment)
        monkeypatch.setattr(worker, "peak_rss_mib", lambda: 64.0)
        monkeypatch.setattr(worker, "threadpool_info", lambda: [{"num_threads": 1}])
        record = worker.worker(config)
        record["imputed_values"] = record.pop("_values")
        # Validation should not depend on clock resolution for this tiny stub.
        record.update(fit_seconds=0.25, transform_seconds=0.75, total_seconds=1.0)
        return record, config, environment, dataset, data, fitted[-1]

    return run


@pytest.mark.parametrize("dataset_id", benchmark.DATASETS)
@pytest.mark.parametrize("dtype", benchmark.DTYPES)
def test_real_worker_prepares_mar_and_parent_accepts_both_mechanisms(
    run_prepared_worker, dataset_id, dtype
):
    mar, config, environment, dataset, data, train = run_prepared_worker(
        "MAR", dataset_id, dtype
    )
    mcar, mcar_config, _, _, _, _ = run_prepared_worker("MCAR", dataset_id, dtype)
    benchmark.validate_record(mar, config, environment, dataset)
    benchmark.validate_record(mcar, mcar_config, environment, dataset)
    case = mar["case"]
    driver = dataset["feature_names"].index(config["mar_driver"])
    order = np.random.default_rng([config["seed"], 0]).permutation(len(data))
    reference_ids = order[
        config["query_size"]:config["query_size"] + config["mar_reference_rows"]
    ]
    assert case["mechanism"] == "MAR"
    assert case["mar_reference_rows"] == config["mar_reference_rows"]
    assert case["mar_cutoff"] == np.median(data[reference_ids, driver])
    assert case["mar_cutoff"] < 0
    features = len(dataset["feature_names"])
    base = config["missing_rate"] * features / (features - 1)
    assert case["eligible_base_probability"] == pytest.approx(base)
    assert case["mar_low_probability"] == pytest.approx(0.5 * base)
    assert case["mar_high_probability"] == pytest.approx(1.5 * base)
    assert np.isfinite(train[:, driver]).all()
    assert train.dtype == np.dtype(dtype)
    assert case["train_mask"]["missing_per_feature"][driver] == 0
    assert case["query_mask"]["missing_per_feature"][driver] == 0
    assert mar["scored_cells"] == sum(case["query_mask"]["missing_per_feature"])
    for name in ("train_row_ids", "query_row_ids", "raw_train", "raw_query"):
        assert case["fingerprints"][name] == mcar["case"]["fingerprints"][name]
    for name in ("train_mask", "query_mask"):
        assert case["fingerprints"][name] != mcar["case"]["fingerprints"][name]


def test_mar_cutoff_and_training_preparation_ignore_held_out_values(
    run_prepared_worker,
):
    original, config, _, dataset, data, train = run_prepared_worker()
    order = np.random.default_rng([config["seed"], 0]).permutation(len(data))
    changed = data.copy()
    driver = dataset["feature_names"].index(config["mar_driver"])
    changed[order[:config["query_size"]], driver] += 1000.0
    repeated, _, _, _, _, repeated_train = run_prepared_worker(data=changed)
    assert repeated["case"]["mar_cutoff"] == original["case"]["mar_cutoff"]
    np.testing.assert_array_equal(repeated_train, train)
    assert (
        repeated["case"]["fingerprints"]["raw_query"]
        != original["case"]["fingerprints"]["raw_query"]
    )


@pytest.mark.parametrize(
    "field, value",
    [("eligible_base_probability", 0.9), ("mar_reference_rows", 1),
     ("mar_cutoff", None), ("mar_cutoff", float("nan")),
     ("mar_cutoff", float("inf")), ("mar_cutoff", True),
     ("mar_low_probability", 0.0), ("mar_high_probability", 1.0)],
)
def test_parent_rejects_malformed_mar_evidence(run_prepared_worker, field, value):
    record, config, environment, dataset, _, _ = run_prepared_worker()
    record["case"][field] = value
    with pytest.raises(ValueError, match="(?i)missingness|MAR|probability"):
        benchmark.validate_record(record, config, environment, dataset)


@pytest.mark.parametrize("mask", ["train_mask", "query_mask"])
def test_parent_rejects_a_missing_mar_driver(run_prepared_worker, mask):
    record, config, environment, dataset, _, _ = run_prepared_worker()
    driver = dataset["feature_names"].index(config["mar_driver"])
    record["case"][mask]["missing_per_feature"][driver] = 1
    with pytest.raises(ValueError, match="(?i)missingness|driver"):
        benchmark.validate_record(record, config, environment, dataset)


@pytest.mark.parametrize("prepared, claimed", [("MCAR", "MAR"), ("MAR", "MCAR")])
def test_relabeling_the_case_cannot_disguise_its_missingness(
    run_prepared_worker, prepared, claimed
):
    record, config, environment, dataset, _, _ = run_prepared_worker(prepared)
    config["mechanism"] = claimed
    record["case"]["mechanism"] = claimed
    with pytest.raises(ValueError, match="(?i)missingness|MAR|mechanism|probability"):
        benchmark.validate_record(record, config, environment, dataset)


def aggregation_records(mechanism):
    return [
        {**config, "record_index": index, "status": "ok", "checks_passed": True,
         "case": {"seed": config["seed"], "mechanism": mechanism},
         "imputed_values": [1.0, 2.0], "output_sha256": "a" * 64,
         "fit_seconds": 0.25, "transform_seconds": 0.75, "total_seconds": 1.0,
         "worker_peak_rss_mib": 64.0, "rmse": 0.1, "mae": 0.08}
        for index, config in enumerate(benchmark.build_configs(
            "0.3.21", "0.3.22", "unused-cache", mechanism=mechanism
        ))
    ]


def test_mar_pairs_retain_mechanism_and_mixed_results_cannot_be_pooled():
    mar = aggregation_records("MAR")
    assert all(summary["complete"] for summary in benchmark.summarize(mar))
    for comparison in benchmark.compare_records(mar):
        assert comparison["matched_pairs"] == 9
        assert all(
            pair["match"]["mechanism"] == "MAR" for pair in comparison["pairs"]
        )
    mixed = aggregation_records("MCAR") + mar
    for aggregate in (benchmark.summarize, benchmark.compare_records):
        with pytest.raises(ValueError, match="Cannot pool"):
            aggregate(mixed)


def test_pairs_never_match_across_mechanisms():
    rows = aggregation_records("MCAR")
    previous = next(row for row in rows if row["variant"] == "previous")
    current = deepcopy(next(row for row in rows if row["variant"] == "current"))
    current["mechanism"] = "MAR"
    assert all(item["matched_pairs"] == 0
               for item in benchmark.compare_records([previous, current]))


@pytest.mark.parametrize("mechanism_arg, expected", [(None, "MCAR"), ("MAR", "MAR")])
def test_cli_forwards_and_records_the_selected_mechanism(
    monkeypatch, tmp_path, mechanism_arg, expected
):
    previous_python = tmp_path / "previous-python"
    previous_python.write_text("offline stub", encoding="utf-8")
    output = tmp_path / "results.json"
    argv = [
        "benchmark_released_real_data", "--previous-python", str(previous_python),
        "--previous-version", "0.3.21", "--current-version", "0.3.22",
        "--data-home", str(tmp_path), "--output", str(output),
        "--dataset", "abalone", "--dtype", "float32", "--weights", "distance",
    ]
    if mechanism_arg is not None:
        argv += ["--mechanism", mechanism_arg]
    seen = []

    def run_worker(python, config, timeout):
        seen.append(config.copy())
        return {"status": "error", "error": "intentional offline worker stub"}

    monkeypatch.setattr(benchmark.sys, "argv", argv)
    monkeypatch.setattr(benchmark, "check_released_package", lambda version: None)
    monkeypatch.setattr(benchmark, "metadata", lambda: {})
    monkeypatch.setattr(
        benchmark, "load_dataset",
        lambda *args, **kwargs: (None, None, {"dataset": "offline synthetic features"}),
    )
    monkeypatch.setattr(benchmark, "run_worker", run_worker)
    assert benchmark.main() == 1  # Stub workers deliberately provide no measurements.
    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["parameters"]["mechanism"] == expected
    assert len(seen) == 27
    assert results["planned_configs"] == seen
    for collection in (seen, results["records"]):
        assert all(row["mechanism"] == expected for row in collection)
        assert all(row["weights"] == "distance" for row in collection)
        assert all(row["dataset_id"] == "abalone" and row["dtype"] == "float32"
                   for row in collection)
