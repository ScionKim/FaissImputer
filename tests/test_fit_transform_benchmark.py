"""Correctness checks for the same-data benchmark; no timing thresholds."""

from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from benchmarks import benchmark_fit_transform as driver
from benchmarks.benchmark_fit_transform_worker import (
    make_training_data,
    worker,
)
from benchmarks.benchmark_scaling_threads import METHODS, NEIGHBORS


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_training_data_is_reproducible_and_preserves_hidden_truth(dtype):
    data, truth, missing = make_training_data(64, 101, dtype)
    repeated = make_training_data(64, 101, dtype)

    for actual, expected in zip((data, truth, missing), repeated):
        np.testing.assert_array_equal(actual, expected)

    assert data.dtype == np.dtype(dtype)
    assert truth.dtype == np.dtype(dtype)
    assert data.shape == truth.shape == missing.shape
    assert missing.any()
    assert not missing[:NEIGHBORS].any()
    assert (~missing.any(axis=1)).sum() >= NEIGHBORS
    assert np.isfinite(truth).all()

    np.testing.assert_array_equal(np.isnan(data), missing)
    np.testing.assert_array_equal(data[~missing], truth[~missing])

    if dtype == "float64":
        rounded = truth.astype(np.float32).astype(np.float64)
        assert np.any(truth != rounded)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_worker_apis_agree_on_the_same_incomplete_data(method, dtype):
    config = {
        "method": method,
        "size": 32,
        "seed": 101,
        "dtype": dtype,
    }

    combined = worker({**config, "api": "fit_transform"})
    split = worker({**config, "api": "fit_then_transform"})

    for result in (combined, split):
        assert result["status"] == "ok"
        assert result["checks_passed"] is True
        assert result["input_dtype"] == dtype
        assert result["output_dtype"] == dtype
        assert result["scored_cells"] > 0
        assert result["complete_donors"] >= NEIGHBORS
        assert result["faiss_omp_threads"] == 1
        assert result["total_seconds"] >= 0
        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mae"])
        assert result["quality_scope"] == "masked training entries"

    assert combined["fingerprints"] == split["fingerprints"]
    assert combined["output_sha256"] == split["output_sha256"]
    np.testing.assert_array_equal(combined["_values"], split["_values"])

    assert combined["fit_seconds"] is None
    assert combined["transform_seconds"] is None
    assert combined["fit_transform_seconds"] == combined["total_seconds"]

    assert split["fit_transform_seconds"] is None
    assert split["fit_seconds"] >= 0
    assert split["transform_seconds"] >= 0
    assert split["total_seconds"] == pytest.approx(
        split["fit_seconds"] + split["transform_seconds"]
    )


def _comparison_record():
    environment = {"python": "test-python"}
    record = {
        "size": 16,
        "dtype": "float64",
        "seed": 101,
        "method": "KNNImputer",
        "api": "fit_transform",
        "input_dtype": "float64",
        "output_dtype": "float64",
        "environment": environment.copy(),
        "checks_passed": True,
        "faiss_omp_threads": 1,
        "threadpools": [{"num_threads": 1}],
        "scored_cells": 2,
        "fingerprints": {
            "input": "input-a",
            "truth": "truth-a",
            "missing": "mask-a",
        },
        "output_sha256": "output-a",
    }
    return record, environment


@pytest.mark.parametrize(
    "values",
    [
        None,
        [1.0, np.nan],
        [[1.0, 2.0]],
        [1.0],
    ],
    ids=["missing", "nonfinite", "wrong-dimension", "wrong-length"],
)
def test_invalid_payload_cannot_become_a_comparison_reference(values):
    record, environment = _comparison_record()
    inputs, outputs = {}, {}

    with pytest.raises(ValueError, match="comparison payload"):
        driver.validate_record(record, values, environment, inputs, outputs)

    assert inputs == {}
    assert outputs == {}


def test_output_equality_is_required_within_each_method():
    record, environment = _comparison_record()
    inputs, outputs = {}, {}
    driver.validate_record(
        record, [1.0, 2.0], environment, inputs, outputs
    )

    changed = deepcopy(record)
    changed["api"] = "fit_then_transform"
    changed["output_sha256"] = "output-b"

    with pytest.raises(ValueError, match="Outputs differ"):
        driver.validate_record(
            changed, [1.0, 3.0], environment, inputs, outputs
        )

    assert len(outputs) == 1
    reference = next(iter(outputs.values()))
    np.testing.assert_array_equal(reference[1], [1.0, 2.0])

    # Different imputation methods may legitimately produce different values.
    changed["method"] = "FaissImputer[available]"
    driver.validate_record(
        changed, [1.0, 3.0], environment, inputs, outputs
    )
    assert len(outputs) == 2


def test_different_methods_must_still_receive_identical_inputs():
    record, environment = _comparison_record()
    inputs, outputs = {}, {}
    driver.validate_record(
        record, [1.0, 2.0], environment, inputs, outputs
    )

    changed = deepcopy(record)
    changed["method"] = "FaissImputer[complete]"
    changed["fingerprints"]["input"] = "different-input"

    with pytest.raises(ValueError, match="different inputs"):
        driver.validate_record(
            changed, [1.0, 2.0], environment, inputs, outputs
        )

    assert len(inputs) == 1
    assert len(outputs) == 1


def test_summary_excludes_failures_and_reports_pending_workers():
    base = {
        "size": 16,
        "dtype": "float64",
        "method": "KNNImputer",
        "api": "fit_transform",
    }
    configs = [{**base, "repeat": repeat} for repeat in (1, 2, 3)]
    records = [
        {**configs[0], "status": "ok", "total_seconds": 2.0},
        {**configs[1], "status": "error", "total_seconds": 999.0},
    ]

    summary = driver.summarize(records, configs)[0]

    assert summary["planned_workers"] == 3
    assert summary["recorded_workers"] == 2
    assert summary["successful_workers"] == 1
    assert summary["pending_workers"] == 1
    assert summary["status_counts"] == {"ok": 1, "error": 1}
    assert summary["total_seconds"] == {
        "median": 2.0,
        "min": 2.0,
        "max": 2.0,
    }


def test_budget_exhaustion_preserves_results_and_stops_launching_workers(
    tmp_path, monkeypatch
):
    commit = "a" * 40
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({
            "version": "1.2.3",
            "source_commit": commit,
            "wheel_sha256": "b" * 64,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "results.json"
    clock = [0.0]
    timeouts = []

    monkeypatch.setattr(
        driver, "check_released_package", lambda expected: None
    )
    monkeypatch.setattr(
        driver, "metadata", lambda: {"git_commit": commit}
    )
    monkeypatch.setattr(
        driver,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0]),
    )

    def timeout_worker(config, timeout):
        timeouts.append(timeout)
        clock[0] += timeout
        return {"status": "timeout", "timeout_seconds": timeout}

    monkeypatch.setattr(driver, "run_worker", timeout_worker)
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [
            "benchmark_fit_transform",
            "--expected-version", "1.2.3",
            "--provenance", str(provenance),
            "--sizes", "16",
            "--seeds", "101",
            "--repeats", "1",
            "--timeout-seconds", "180",
            "--budget-seconds", "1",
            "--output", str(output),
        ],
    )

    assert driver.main() == 1
    assert timeouts == [1.0]

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["parameters"]["expected_workers"] == 12
    assert len(result["records"]) == 12
    assert Counter(row["status"] for row in result["records"]) == {
        "timeout": 1,
        "not_run_budget": 11,
    }
    assert all(
        row["successful_workers"] == 0
        and row["pending_workers"] == 0
        for row in result["summaries"]
    )
    assert not output.with_name(output.name + ".tmp").exists()