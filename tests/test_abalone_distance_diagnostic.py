"""Offline distance-diagnostic regressions; execution belongs in GitHub CI."""

from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import sys
from zipfile import ZipFile

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from benchmarks import diagnose_abalone_output as diagnostic
from benchmarks import diagnose_distance_weights as weighting


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["float32", "float64"])
def distance_archive(request, tmp_path):
    path = ROOT / "benchmarks/results/released_abalone_distance_0.3.22.zip"
    assert sha256(path.read_bytes()).hexdigest() == diagnostic.DISTANCE_ARCHIVE_SHA256
    profile = diagnostic.diagnostic_profile(f"released_distance_{request.param}")
    with ZipFile(path) as archive:
        raw = archive.read(f"version_comparison_abalone_{request.param}_distance.json")
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(raw)
    return profile, diagnostic.read_baseline(baseline, profile)


def test_distance_profiles_select_current_release_and_weights(distance_archive):
    profile, baseline = distance_archive
    assert profile["weights"] == "distance"
    assert profile["target"]["seed"] == 101
    assert profile["source"] == baseline["metadata"]["git_commit"]
    assert profile["run"] == baseline["metadata"]["github_run_id"]
    selected = diagnostic.select_records(baseline, profile)
    for label, (index, record) in selected.items():
        assert record is baseline["records"][index]
        assert record["variant"] == ("current" if label == "faiss" else "knn")
        assert record["expected_version"] == "0.3.22"
        assert record["weights"] == record["model_parameters"]["weights"] == "distance"
        assert record["dtype"] == profile["target"]["dtype"]
    assert selected["knn"][1]["case"] == selected["faiss"][1]["case"]


@pytest.mark.parametrize("location", ["parameters", "record", "model"])
@pytest.mark.parametrize("change", ["missing", "uniform"])
def test_distance_weight_mismatches_are_rejected(distance_archive, location, change):
    profile, original = distance_archive
    baseline = deepcopy(original)
    record = diagnostic.select_records(baseline, profile)["faiss"][1]
    target = {
        "parameters": baseline["parameters"],
        "record": record,
        "model": record["model_parameters"],
    }[location]
    if change == "missing":
        target.pop("weights")
    else:
        target["weights"] = "uniform"
    with pytest.raises(ValueError, match="configuration|weights"):
        diagnostic.select_records(baseline, profile)


def test_distance_case_rejects_uniform_run_provenance(distance_archive):
    profile, original = distance_archive
    baseline = deepcopy(original)
    baseline["metadata"]["git_commit"] = diagnostic.RELEASE_SOURCE
    baseline["metadata"]["github_run_id"] = diagnostic.RELEASE_RUN
    with pytest.raises(ValueError, match="benchmark provenance"):
        diagnostic.select_records(baseline, profile)


def test_distance_wheel_requires_distance_archive_provenance(tmp_path):
    profile = diagnostic.diagnostic_profile("released_distance_float64")
    wrong = {
        "kind": "published-wheel", "version": "0.3.22",
        "archive_sha256": diagnostic.RELEASE_ARCHIVE_SHA256,
    }
    with pytest.raises(ValueError, match="published-wheel provenance"):
        diagnostic.verify_published_wheel(tmp_path / "provenance.json", wrong, tmp_path, profile)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_changed_distance_json_is_rejected_before_parsing(tmp_path, monkeypatch, dtype):
    path = tmp_path / "changed.json"
    path.write_bytes(b'{"records": []}')

    def unexpected_parse(*args, **kwargs):
        pytest.fail("Unverified bytes reached the JSON parser")

    monkeypatch.setattr(diagnostic.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="Archived JSON checksum mismatch"):
        diagnostic.read_baseline(path, diagnostic.diagnostic_profile(f"released_distance_{dtype}"))


def test_weighted_reference_uses_square_roots_and_handles_tiny_distances():
    for scale in (Fraction(1), Fraction(1, 10**600)):
        result = weighting.distance_weighted_reference(
            [10.0, 20.0], [scale, 4 * scale],
        )
        assert result["zero_distance_positions"] == []
        assert result["float64_approximations_agree"]
        assert [row["decimal_precision"] for row in result["evaluations"]] == [80, 120]
        for row in result["evaluations"]:
            assert row["weighted_mean_float64"] == float(Fraction(40, 3))


def test_zero_distance_neighbors_exclude_positive_distance_values():
    result = weighting.distance_weighted_reference(
        [1.0, 3.0, 999.0], [Fraction(0), Fraction(0), Fraction(9)],
    )
    assert result["zero_distance_positions"] == [0, 1]
    for row in result["evaluations"]:
        assert row["weighted_mean_float64"] == 2.0
        assert row["normalized_weights"] == ["0.5", "0.5", "0"]


def test_captured_weight_reference_separates_aggregation_and_assignment():
    exact_mean = Fraction(14, 3)
    returned = float(exact_mean)
    assigned = np.float32(returned)
    result = weighting.captured_weight_reference([2.0, 10.0], [1.0, 0.5], returned, assigned)
    mean = result["exact_weighted_mean"]
    assert Fraction(int(mean["numerator"]), int(mean["denominator"])) == exact_mean
    assignment = result["assignment_minus_returned_value"]
    assert Fraction(int(assignment["numerator"]), int(assignment["denominator"])) == (
        Fraction.from_float(float(assigned)) - Fraction.from_float(returned)
    )
    assert assignment["float64"] != 0


def test_zero_weight_sum_is_not_reported_as_a_valid_mean():
    with pytest.raises(ValueError, match="positive sum"):
        weighting.captured_weight_reference([1.0, 2.0], [0.0, 0.0], 0.0, 0.0)


def test_distance_selection_does_not_label_uniform_mean_as_its_reference():
    train = np.arange(10, 60, 10, dtype=np.float64).reshape(-1, 1)
    distances = [Fraction(i * i) for i in range(1, 6)]
    reference = diagnostic.reference_cell(train, 0, distances)
    weights = 1 / np.arange(1, 6, dtype=np.float64)
    output = float(np.average(train[:, 0], weights=weights))
    observation = {
        "captured_weights": weights, "returned_value": output,
        "weight_input_distances": np.arange(1, 6, dtype=np.float64),
    }
    detail = diagnostic.distance_selection_details(
        train, 0, list(range(5)), distances, reference, output, observation,
    )
    assert detail["admissible_exact_top_k"]
    assert "exact_mean" not in detail
    assert "this_query_output_minus_rounded_exact_mean" not in detail
    approximate = detail["distance_weighted_reference"]["evaluations"][-1]["weighted_mean_float64"]
    assert approximate != float(train.mean())
    assert approximate == detail["captured_distance_weighting_reference"]["evaluations"][-1]["weighted_mean_float64"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_weight_traces_preserve_outputs_and_map_duplicate_queries(dtype):
    train = np.array([
        [0, 10, 0], [1, 20, np.nan], [2, 30, 2],
        [3, 40, 3], [4, 50, 4], [5, 60, 5],
    ], dtype=dtype)
    query = np.array([
        [1.25, np.nan, 1.5], [1.25, np.nan, 1.5], [2, np.nan, np.nan],
    ], dtype=dtype)
    before_train, before_query = train.copy(), query.copy()
    expected_cells = {(0, 1), (1, 1), (2, 1), (2, 2)}
    old_threads = diagnostic.faiss.omp_get_max_threads()
    try:
        diagnostic.faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1):
            for method in ("knn", "faiss"):
                model = diagnostic.make_model(method).set_params(weights="distance").fit(train)
                original = model.transform(query)
                if method == "knn":
                    output, _, selections = diagnostic.trace_knn_selections(
                        model, train, query, [0, 1, 2],
                    )
                else:
                    output, _, _, selections = weighting.trace_distance_faiss_selections(
                        model, train, query, [0, 1, 2],
                    )
                    assert sys.getprofile() is None
                assert diagnostic.array_digest(output) == diagnostic.array_digest(original)
                assert set(selections) == expected_cells
                for (row, column), observation in selections.items():
                    ids = observation["training_row_indices"]
                    weights = observation["captured_weights"]
                    distances = observation["weight_input_distances"]
                    assert len(ids) == len(weights) == len(distances) == 5
                    assert observation["assigned_output"] == output[row, column]
                    assert observation["target_dtype"] == np.dtype(dtype).name
                    assert np.isfinite(weights).all() and (weights >= 0).all()
                    if row == 2:
                        assert (distances == 0).any()
                        np.testing.assert_array_equal(weights != 0, distances == 0)
                    reference = weighting.captured_weight_reference(
                        train[ids, column], weights,
                        observation["returned_value"], output[row, column],
                    )
                    assert np.isfinite(reference["exact_weighted_mean"]["float64"])
                np.testing.assert_array_equal(
                    selections[(0, 1)]["training_row_indices"],
                    selections[(1, 1)]["training_row_indices"],
                )
    finally:
        diagnostic.faiss.omp_set_num_threads(old_threads)
    np.testing.assert_array_equal(train, before_train)
    np.testing.assert_array_equal(query, before_query)


def test_faiss_profiler_is_restored_on_trace_failure(monkeypatch):
    model = diagnostic.make_model("faiss").set_params(weights="distance")

    def fail_trace(*args, **kwargs):
        raise RuntimeError("trace sentinel")

    monkeypatch.setattr(weighting, "trace_faiss_searches", fail_trace)
    previous = sys.getprofile()
    assert previous is None
    with pytest.raises(RuntimeError, match="trace sentinel"):
        weighting.trace_distance_faiss_selections(model, None, None, [])
    assert sys.getprofile() is previous


def test_archive_comparison_keeps_disappeared_and_shared_output_changes():
    missing = np.array([[False, True], [False, True], [False, True]])
    current = np.array([[5, 1], [6, 2], [7, 3]], dtype=np.float64)
    outputs = {label: current.copy() for label in ("knn", "faiss")}
    archived_knn = [1.5, 2.25, 3 + 2**-23]
    archived_faiss = [1.0, 2.25, 3.0]
    chosen = {
        "knn": (0, {"imputed_values": archived_knn}),
        "faiss": (1, {"imputed_values": archived_faiss}),
    }
    comparison, archived, scores = diagnostic.compare_archived_imputed_values(
        chosen, outputs, missing,
    )
    # Row 0's old between-method difference has disappeared. Row 1 changed
    # identically in both methods. Row 2 changed by less than the usual threshold.
    assert comparison["changed_query_rows"] == [0, 1, 2]
    assert comparison["methods"]["knn"]["changed_value_count"] == 3
    assert comparison["methods"]["faiss"]["changed_value_count"] == 1
    np.testing.assert_array_equal(scores, [0.5, 0.25, 2**-23])
    assert comparison["methods"]["knn"]["changed_cells"][0] == {
        "query_row_index": 0, "feature_index": 1,
        "archived_value": 1.5, "current_value": 1.0, "absolute_difference": 0.5,
    }
    for label, values in (("knn", archived_knn), ("faiss", archived_faiss)):
        assert np.isnan(archived[label][~missing]).all()
        np.testing.assert_array_equal(archived[label][missing], values)
        np.testing.assert_array_equal(outputs[label], current)
        assert chosen[label][1]["imputed_values"] == values


@pytest.mark.parametrize("values", [[1.0], [1.0, float("nan")]])
def test_archive_comparison_rejects_invalid_masked_values(values):
    missing = np.ones((2, 1), dtype=bool)
    outputs = {label: np.ones((2, 1)) for label in ("knn", "faiss")}
    chosen = {label: (0, {"imputed_values": values}) for label in outputs}
    with pytest.raises(ValueError, match="invalid shape or value"):
        diagnostic.compare_archived_imputed_values(chosen, outputs, missing)


@pytest.mark.parametrize(
    "mode", ["reproduced", "archive_changed", "trace_changed", "uniform_archive_changed"],
)
def test_diagnostic_preserves_reproduction_failure_while_tracing_current_outputs(
    tmp_path, monkeypatch, mode,
):
    weights = "uniform" if mode == "uniform_archive_changed" else "distance"
    case_name = "released_float64" if weights == "uniform" else "released_distance_float64"
    train = np.array([[0, 10], [1, 20], [2, 30], [3, 40], [4, 50], [5, 60]], dtype=np.float64)
    query = np.array([[1.25, np.nan], [2.25, np.nan]], dtype=np.float64)
    truth = np.array([[1.25, 22.5], [2.25, 32.5]], dtype=np.float64)
    missing = np.isnan(query)
    names, dataset, case = ["Length", "target"], {"fixture": "offline"}, {"fixture": "same inputs"}
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        case=case_name, expected_version="0.3.22", baseline=tmp_path / "baseline.json",
        provenance=provenance, data_home=tmp_path, threshold=1e-5, max_rows=20,
        output=tmp_path / "diagnostic.json",
    )
    monkeypatch.setattr(diagnostic, "check_released_package", lambda *args: None)
    monkeypatch.setattr(diagnostic, "version", lambda name: diagnostic.RELEASE_DEPENDENCIES[name])
    monkeypatch.setattr(diagnostic, "verify_published_wheel", lambda *args: {})
    monkeypatch.setattr(diagnostic, "metadata", lambda: {"fixture": "offline"})
    monkeypatch.setattr(diagnostic, "read_baseline", lambda *args: {
        "metadata": {"python": sys.version.split()[0]}, "dataset": dataset,
    })
    monkeypatch.setattr(diagnostic, "load_dataset", lambda *args, **kwargs: (train.copy(), names, dataset))
    monkeypatch.setattr(diagnostic, "prepare_case", lambda *args, **kwargs: (
        train.copy(), query.copy(), truth.copy(), missing.copy(), case,
    ))
    report = {"status": "error", "original_reproduced": False, "inputs_reproduced": False}
    old_threads = diagnostic.faiss.omp_get_max_threads()
    calls = []
    original_trace = diagnostic.trace_knn_selections

    def trace(model, actual_train, actual_query, wanted):
        calls.append(list(wanted))
        output, captured, selections = original_trace(model, actual_train, actual_query, wanted)
        if mode == "trace_changed":
            output = output.copy()
            output[0, 1] += 1
        return output, captured, selections

    monkeypatch.setattr(diagnostic, "trace_knn_selections", trace)
    try:
        diagnostic.faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1):
            chosen, outputs = {}, {}
            for index, label in enumerate(("knn", "faiss")):
                model = diagnostic.make_model(label).set_params(weights=weights)
                outputs[label] = model.fit(train).transform(query)
                archived = outputs[label].copy()
                if label == "knn" and mode != "reproduced":
                    archived[0, 1] += 0.25
                chosen[label] = (index, {
                    "case": case, "output_sha256": diagnostic.array_digest(archived),
                    "imputed_values": archived[missing].tolist(),
                    "model_parameters": {name: model.get_params()[name]
                                         for name in ("copy", "metric", "n_neighbors", "weights")},
                })
            preserved = deepcopy(chosen)
            monkeypatch.setattr(diagnostic, "select_records", lambda *args: chosen)
            assert np.max(np.abs(outputs["knn"] - outputs["faiss"])) < args.threshold
            if mode == "trace_changed":
                with pytest.raises(AssertionError):
                    diagnostic.diagnose(args, report)
                assert calls == [[0]]
                assert report["tracing_completed"] is False
                assert report["original_reproduced"] is False
                return
            diagnostic.diagnose(args, report)
    finally:
        diagnostic.faiss.omp_set_num_threads(old_threads)
    assert chosen == preserved
    assert report["threadpools"]
    if mode == "uniform_archive_changed":
        assert calls == [] and report["rows"] == []
        assert report["status"] == "baseline_output_mismatch"
        assert report["tracing_completed"] is False
        return
    reproduced = mode == "reproduced"
    assert report["original_reproduced"] is reproduced
    assert report["trace_explains_archived_outputs"] is reproduced
    assert report["trace_scope"] == "current_execution"
    assert report["tracing_completed"] is True
    assert report["status"] == ("ok" if reproduced else "baseline_output_mismatch")
    if not reproduced:
        assert calls == [[0]]
        assert report["affected_query_rows"] == 0
        assert report["candidate_query_rows"] == [0]
        assert report["detailed_query_rows"] == [0]
        assert report["archived_output_comparison"]["cells_above_threshold"] == 1
        assert report["cells_above_threshold"] == 0
        feature = report["rows"][0]["features"][0]
        assert feature["observation_scope"] == "current_execution"
        assert feature["output_matches_archive"] == {"knn": False, "faiss": True}
        assert feature["archived_outputs"]["knn"] == chosen["knn"][1]["imputed_values"][0]
        assert report["traced_output_sha256"] == report["output_sha256"]
        with np.load(args.output.with_suffix(".npz"), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["knn_output"], outputs["knn"])
            np.testing.assert_array_equal(arrays["traced_query_rows"], [0])


def test_cli_keeps_nonzero_exit_after_current_execution_trace(tmp_path, monkeypatch):
    def completed_observation(args, report):
        report.update(
            status="baseline_output_mismatch", original_reproduced=False,
            inputs_reproduced=True, tracing_completed=True, trace_scope="current_execution",
        )

    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(diagnostic, "diagnose", completed_observation)
    monkeypatch.setattr(diagnostic, "threadpool_limits", lambda **kwargs: nullcontext())
    monkeypatch.setattr(diagnostic, "config_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(diagnostic.faiss, "omp_set_num_threads", lambda count: None)
    monkeypatch.setattr(diagnostic.sys, "argv", [
        "diagnose_abalone_output", "--case", "released_distance_float32",
        "--expected-version", "0.3.22", "--provenance", str(tmp_path / "provenance.json"),
        "--baseline", str(tmp_path / "baseline.json"), "--data-home", str(tmp_path),
        "--output", str(output),
    ])
    assert diagnostic.main() == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["original_reproduced"] is False
    assert saved["status"] == "baseline_output_mismatch"
    assert saved["tracing_completed"] is True
