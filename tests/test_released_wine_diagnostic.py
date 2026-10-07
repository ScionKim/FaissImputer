"""Offline checks for published Wine float32 output diagnostics."""

from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from benchmarks import diagnose_abalone_output as shared
from benchmarks import diagnose_wine_quality_output as wine


@pytest.fixture(scope="module", params=wine.CASES[1:])
def released_wine_evidence(request, tmp_path_factory):
    profile = wine.diagnostic_profile(request.param)
    suffix = "_distance" if profile["weights"] == "distance" else ""
    archive_path = wine.ROOT / (
        f"benchmarks/results/released_wine_quality_float32{suffix}_0.3.22.zip"
    )
    assert sha256(archive_path.read_bytes()).hexdigest() == profile["archive_sha256"]
    member = f"version_comparison_wine_quality_white_float32{suffix}.json"
    with ZipFile(archive_path) as archive:
        raw = archive.read(member)
    path = tmp_path_factory.mktemp(request.param) / "baseline.json"
    path.write_bytes(raw)
    return profile, shared.read_baseline(path, profile)


def test_published_wine_profiles_select_the_exact_archived_case(released_wine_evidence):
    profile, baseline = released_wine_evidence
    selected = shared.select_records(baseline, profile)
    assert profile["target"]["dataset_id"] == "wine_quality_white"
    assert profile["target"]["dtype"] == "float32"
    assert profile["features"] == 11 and profile["mar_driver"] == "alcohol"
    assert profile["trace_current_on_mismatch"] is True
    assert profile["source"] == baseline["metadata"]["git_commit"]
    assert profile["run"] == baseline["metadata"]["github_run_id"]
    for label, (index, record) in selected.items():
        assert record is baseline["records"][index]
        assert record["variant"] == ("current" if label == "faiss" else "knn")
        assert record["seed"] == profile["target"]["seed"]
        assert record["repeat"] == 1
        assert record["expected_version"] == "0.3.22"
        assert record.get("weights", "uniform") == profile["weights"]
        assert record["model_parameters"]["weights"] == profile["weights"]
    assert selected["knn"][1]["case"] == selected["faiss"][1]["case"]
    left = selected["knn"][1]["imputed_values"]
    right = selected["faiss"][1]["imputed_values"]
    expected_counts = {101: 47, 202: 35, 303: 38}
    expected = 2 if profile["weights"] == "uniform" else expected_counts[profile["target"]["seed"]]
    assert sum(abs(a - b) > 1e-5 for a, b in zip(left, right)) == expected


@pytest.mark.parametrize("change", [
    "dataset", "features", "driver", "weights", "model_weights", "variant",
    "seed", "duplicate", "failed", "inputs", "provenance",
])
def test_wine_profile_rejects_wrong_or_inconsistent_records(released_wine_evidence, change):
    profile, original = released_wine_evidence
    baseline = deepcopy(original)
    record = shared.select_records(baseline, profile)["faiss"][1]
    if change == "dataset":
        baseline["parameters"]["dataset_id"] = "abalone"
    elif change == "features":
        record["features"] = 7
    elif change == "driver":
        record["mar_driver"] = "Length"
    elif change == "weights":
        record["weights"] = "uniform" if profile["weights"] == "distance" else "distance"
    elif change == "model_weights":
        record["model_parameters"]["weights"] = "uniform" if profile["weights"] == "distance" else "distance"
    elif change == "variant":
        record["variant"] = "previous"
    elif change == "seed":
        record["seed"] = 999
    elif change == "duplicate":
        baseline["records"].append(deepcopy(record))
    elif change == "failed":
        record["status"] = "error"
    elif change == "inputs":
        record["case"]["fingerprints"]["query"] = "f" * 64
    else:
        baseline["metadata"]["github_run_id"] = "wrong-run"
    with pytest.raises(ValueError):
        shared.select_records(baseline, profile)


def test_changed_wine_json_is_rejected_before_parsing(tmp_path, monkeypatch):
    path = tmp_path / "changed.json"
    path.write_bytes(b'{"records": []}')

    def unexpected_parse(*args, **kwargs):
        pytest.fail("Unverified bytes reached the JSON parser")

    monkeypatch.setattr(shared.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="Archived JSON checksum mismatch"):
        shared.read_baseline(path, wine.diagnostic_profile("released_distance_float32_seed101"))


def test_wine_wheel_rejects_abalone_archive_provenance(tmp_path):
    profile = wine.diagnostic_profile("released_uniform_float32_seed101")
    provenance = {
        "kind": "published-wheel", "version": "0.3.22",
        "archive_sha256": shared.RELEASE_ARCHIVE_SHA256,
    }
    with pytest.raises(ValueError, match="published-wheel provenance"):
        shared.verify_published_wheel(tmp_path / "wheel.json", provenance, tmp_path, profile)


def test_historical_wine_dispatch_keeps_the_existing_diagnostic(monkeypatch):
    calls = []
    args, report = SimpleNamespace(), {}
    monkeypatch.setattr(wine, "diagnose_historical", lambda a, r: calls.append((a, r)))

    def unexpected_published(*args, **kwargs):
        pytest.fail("Historical Wine case reached the published diagnostic")

    monkeypatch.setattr(shared, "diagnose", unexpected_published)
    wine.diagnose(args, report)
    assert calls == [(args, report)]
    assert wine.diagnostic_profile("historical")["target"] == wine.TARGET


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("mode", ["reproduced", "archive_changed", "trace_changed"])
def test_wine_uses_alcohol_column_and_retains_failed_reproduction(
    tmp_path, monkeypatch, weights, mode,
):
    # Alcohol is deliberately the last column, while the first column is
    # incomplete. The tiny fixture uses real model and trace calculations.
    train = np.array([
        [10, 0], [20, 1], [np.nan, 2], [40, 3], [50, 4], [60, 5], [70, 6],
    ], dtype=np.float32)
    query = np.array([[np.nan, 1.25], [np.nan, 2.25]], dtype=np.float32)
    truth = np.array([[22.5, 1.25], [32.5, 2.25]], dtype=np.float64)
    missing = np.isnan(query)
    names = ["target", "alcohol"]
    dataset, case = {"fixture": "Wine"}, {"fixture": "identical prepared inputs"}
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        case=f"released_{weights}_float32_seed101", expected_version="0.3.22",
        baseline=tmp_path / "baseline.json", provenance=provenance,
        data_home=tmp_path, output=tmp_path / "diagnostic.json",
        threshold=1e-5, max_rows=100,
    )
    monkeypatch.setattr(shared, "check_released_package", lambda *args: None)
    monkeypatch.setattr(shared, "version", lambda name: shared.RELEASE_DEPENDENCIES[name])
    monkeypatch.setattr(shared, "verify_published_wheel", lambda *args: {})
    monkeypatch.setattr(shared, "metadata", lambda: {"fixture": "offline"})
    monkeypatch.setattr(shared, "read_baseline", lambda *args: {
        "metadata": {"python": sys.version.split()[0]}, "dataset": dataset,
    })
    preparation_calls = []

    def load_dataset(data_home, **kwargs):
        assert kwargs == {"dataset_id": "wine_quality_white", "download_if_missing": False}
        return train.copy(), names, dataset

    def prepare_case(data, actual_names, **kwargs):
        assert actual_names == names
        preparation_calls.append(kwargs)
        return train.copy(), query.copy(), truth.copy(), missing.copy(), case

    monkeypatch.setattr(shared, "load_dataset", load_dataset)
    monkeypatch.setattr(shared, "prepare_case", prepare_case)
    calls = []
    original_trace = shared.trace_knn_selections

    def trace(model, actual_train, actual_query, wanted):
        calls.append(list(wanted))
        output, captured, selections = original_trace(model, actual_train, actual_query, wanted)
        if mode == "trace_changed":
            output = output.copy()
            output[0, 0] += 1
        return output, captured, selections

    monkeypatch.setattr(shared, "trace_knn_selections", trace)
    report = {"status": "error", "inputs_reproduced": False, "original_reproduced": False}
    previous_threads = shared.faiss.omp_get_max_threads()
    try:
        shared.faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1):
            chosen, outputs = {}, {}
            for index, label in enumerate(("knn", "faiss")):
                model = shared.make_model(label).set_params(weights=weights)
                output = model.fit(train).transform(query)
                outputs[label] = output
                archived = output.copy()
                if label == "knn" and mode != "reproduced":
                    archived[0, 0] += 0.25
                chosen[label] = (index, {
                    "case": case, "output_sha256": shared.array_digest(archived),
                    "imputed_values": archived[missing].tolist(),
                    "model_parameters": {name: model.get_params()[name]
                                         for name in ("copy", "metric", "n_neighbors", "weights")},
                })
            before = deepcopy(chosen)
            monkeypatch.setattr(shared, "select_records", lambda *args: chosen)
            if mode == "trace_changed":
                with pytest.raises(AssertionError):
                    wine.diagnose(args, report)
                assert calls and 0 in calls[0]
                assert report["tracing_completed"] is False
                return
            wine.diagnose(args, report)
    finally:
        shared.faiss.omp_set_num_threads(previous_threads)
    assert chosen == before
    assert preparation_calls[0]["mar_driver"] == "alcohol"
    assert preparation_calls[0]["dtype"] == "float32"
    assert preparation_calls[0]["seed"] == 101
    assert report["inputs_reproduced"] is True
    assert report["tracing_completed"] is True
    assert report["trace_scope"] == "current_execution"
    reproduced = mode == "reproduced"
    assert report["original_reproduced"] is reproduced
    assert report["trace_explains_archived_outputs"] is reproduced
    assert report["status"] == ("ok" if reproduced else "baseline_output_mismatch")
    if not reproduced:
        assert calls and 0 in calls[0]
        assert report["baseline_imputed_values_match"] == {"knn": False, "faiss": True}
        assert report["traced_output_sha256"] == report["output_sha256"]
        feature = next(row for row in report["rows"] if row["query_row_index"] == 0)["features"][0]
        assert feature["feature_name"] == "target"
        assert feature["output_matches_archive"] == {"knn": False, "faiss": True}
        if weights == "uniform":
            selection = feature["knn_observation"]
            mean, residual = selection["exact_mean"], selection["output_minus_exact_mean"]
            assert Fraction(int(residual["numerator"]), int(residual["denominator"])) == (
                Fraction.from_float(feature["outputs"]["knn"])
                - Fraction(int(mean["numerator"]), int(mean["denominator"]))
            )
        with np.load(args.output.with_suffix(".npz"), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["knn_output"], outputs["knn"])
            assert saved["train"].dtype == saved["query"].dtype == np.float32


def test_published_wine_cli_uses_recorded_memory_and_fails_on_mismatch(tmp_path, monkeypatch):
    memory = []

    def config_context(**kwargs):
        memory.append(kwargs)
        return nullcontext()

    def diagnose(args, report):
        report.update(status="baseline_output_mismatch", original_reproduced=False,
                      inputs_reproduced=True, tracing_completed=True)

    monkeypatch.setattr(wine, "diagnose", diagnose)
    monkeypatch.setattr(wine, "config_context", config_context)
    monkeypatch.setattr(wine, "threadpool_limits", lambda **kwargs: nullcontext())
    monkeypatch.setattr(wine.faiss, "omp_set_num_threads", lambda count: None)
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(wine.sys, "argv", [
        "diagnose_wine_quality_output", "--case", "released_distance_float32_seed202",
        "--expected-version", "0.3.22", "--provenance", str(tmp_path / "wheel.json"),
        "--baseline", str(tmp_path / "reference.json"), "--data-home", str(tmp_path),
        "--output", str(output),
    ])
    assert wine.main() == 1
    assert memory == [{"working_memory": 256}]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "baseline_output_mismatch"
    assert saved["original_reproduced"] is False
    assert saved["tracing_completed"] is True


def test_published_wine_cli_requires_explicit_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(wine.sys, "argv", [
        "diagnose_wine_quality_output", "--case", "released_uniform_float32_seed101",
        "--expected-version", "0.3.22", "--provenance", str(tmp_path / "wheel.json"),
        "--data-home", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as error:
        wine.main()
    assert error.value.code == 2
