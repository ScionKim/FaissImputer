"""Offline regressions for the separate Wine float32 release archives."""

from copy import deepcopy
import json
from statistics import median
import sys

import pytest

from benchmarks import analyze_released_real_data as analysis


@pytest.fixture(scope="module", params=["uniform", "distance"])
def wine_float32_evidence(request):
    weights = request.param
    profile = analysis.WINE_FLOAT32_PROFILES[weights]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], "wine_quality_white",
        weights=weights, dtype="float32",
    )
    return weights, documents, freezes


@pytest.fixture(scope="module")
def wine_float32_result(wine_float32_evidence):
    weights, documents, freezes = wine_float32_evidence
    return analysis.analyze_dataset(
        documents, freezes, dataset_id="wine_quality_white",
        weights=weights, dtype="float32",
    )


def test_wine_float32_uses_its_own_archive_and_runner(
    wine_float32_evidence, wine_float32_result,
):
    weights, documents, _ = wine_float32_evidence
    result = wine_float32_result
    profile = analysis.WINE_FLOAT32_PROFILES[weights]
    assert result["dtypes"] == list(result["by_dtype"]) == ["float32"]
    assert result["source"]["archive"] == profile["archive"]
    assert result["source"]["github_run_id"] == profile["run"]
    assert result["source"]["json_members"]["float32"]["sha256"] == profile["json_sha256"]
    assert result["validation"]["successful_records"] == 27
    assert not any(key.startswith("cross_dtype_") for key in result["validation"])
    assert result["environment"] == result["by_dtype"]["float32"]["environment"]
    assert result["environment"]["cpu_model"] == documents["float32"]["metadata"]["cpu_model"]
    expected_cpu = "7763" if weights == "uniform" else "9V74"
    assert expected_cpu in result["environment"]["cpu_model"]
    text = analysis.render_dataset_report(result)
    assert "Only float32 is measured in this run." in text
    assert "Only float64 is measured" not in text
    assert f'weights="{weights}"' in text
    assert profile["archive"] in text and profile["summary"] in text
    assert "different CPU models" in text
    assert f"--dataset wine_quality_white --dtype float32 --weights {weights}" in text
    assert "bench/released-wine-quality-float32-results" in text
    assert "Training rows observed for each feature" in text
    assert "not a new benchmark" in text
    assert text.endswith("\n") and "\r" not in text


def test_wine_float32_arithmetic_uses_records_not_ratios_of_medians(
    wine_float32_evidence, wine_float32_result,
):
    weights, documents, _ = wine_float32_evidence
    records = documents["float32"]["records"]
    indexed = {(r["variant"], r["seed"], r["repeat"]): r for r in records}
    section = wine_float32_result["by_dtype"]["float32"]
    assert section["configuration"]["weights"] == weights
    for cell in section["cells"]:
        group = [r for r in records if r["variant"] == cell["variant"]]
        seed_group = [r for r in group if r["repeat"] == 1]
        assert cell["record_count"] == len(group) == 9
        assert cell["quality_seed_count"] == len(seed_group) == 3
        assert {r["seed"] for r in cell["quality_samples"]} == {101, 202, 303}
        for field in ("fit_seconds", "transform_seconds", "total_seconds"):
            values = [r[field] for r in group]
            assert cell["timing"][field] == {
                "median": median(values), "min": min(values), "max": max(values),
            }
        for field in ("rmse", "mae"):
            values = [r[field] for r in seed_group]
            assert cell["quality"][field] == {
                "median": median(values), "min": min(values), "max": max(values),
            }
    for comparison in section["comparisons"]:
        numerator = comparison["numerator_variant"]
        denominator = comparison["denominator_variant"]
        assert comparison["pair_count"] == 9
        ratios, left_times, right_times = [], [], []
        for pair in comparison["pairs"]:
            match = pair["match"]
            left = indexed[(numerator, match["seed"], match["repeat"])]
            right = indexed[(denominator, match["seed"], match["repeat"])]
            assert match["dtype"] == "float32"
            assert all(left[k] == right[k] == v for k, v in match.items())
            assert left.get("weights", "uniform") == right.get("weights", "uniform") == weights
            assert left["case"] == right["case"]
            left_times.append(left["total_seconds"])
            right_times.append(right["total_seconds"])
            ratios.append(left_times[-1] / right_times[-1])
        assert comparison["speedup"]["total_seconds"] == {
            "median": median(ratios), "min": min(ratios), "max": max(ratios),
        }
        if numerator == "previous" and denominator == "current":
            assert median(ratios) != median(left_times) / median(right_times)
            assert comparison["matching_output_hashes"] == 9
        for sample in comparison["seed_output_agreement"]:
            left = indexed[(numerator, sample["seed"], 1)]
            right = indexed[(denominator, sample["seed"], 1)]
            differences = [abs(a - b) for a, b in zip(
                left["imputed_values"], right["imputed_values"],
            )]
            assert sample["repeat"] == 1
            assert sample["hidden_entries_above_1e_minus_5"] == sum(d > 1e-5 for d in differences)
            assert sample["max_scored_abs_difference"] == max(differences)
    versus_knn = next(c for c in section["comparisons"]
                     if c["numerator_variant"] == "knn" and c["denominator_variant"] == "current")
    counts = [s["hidden_entries_above_1e_minus_5"] for s in versus_knn["seed_output_agreement"]]
    assert counts == ([2, 0, 0] if weights == "uniform" else [47, 35, 38])


@pytest.mark.parametrize("change", [
    "parameter_dtype", "mixed_weights", "missing", "duplicate", "failed",
    "changed_inputs", "changed_repeat", "stored_summary", "stored_pair",
])
def test_wine_float32_rejects_inconsistent_evidence(wine_float32_evidence, change):
    weights, original, freezes = wine_float32_evidence
    documents = deepcopy(original)
    data = documents["float32"]
    if change == "parameter_dtype":
        data["parameters"]["dtype"] = "float64"
    elif change == "mixed_weights":
        data["records"][0]["weights"] = "distance" if weights == "uniform" else "uniform"
    elif change == "missing":
        data["records"].pop()
    elif change == "duplicate":
        data["records"][-1] = deepcopy(data["records"][0])
    elif change == "failed":
        data["records"][0]["status"] = "error"
    elif change == "changed_inputs":
        data["records"][0]["case"]["fingerprints"]["query"] = "f" * 64
    elif change == "changed_repeat":
        record = next(r for r in data["records"]
                      if r["variant"] == "current" and r["seed"] == 101 and r["repeat"] == 2)
        record["imputed_values"][0] += 1
    elif change == "stored_summary":
        data["summaries"][0]["timing_and_memory"]["total_seconds"]["median"] += 1
    else:
        data["comparisons"][0]["pairs"][0]["timing_ratios"]["total_seconds"] += 1
    with pytest.raises(ValueError):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="wine_quality_white",
            weights=weights, dtype="float32",
        )


def test_wine_float32_cannot_be_read_as_other_dtype_or_weights(wine_float32_evidence):
    weights, documents, freezes = wine_float32_evidence
    path = analysis.ROOT / analysis.WINE_FLOAT32_PROFILES[weights]["archive"]
    other_weights = "distance" if weights == "uniform" else "uniform"
    for dtype, selected_weights in (("float64", weights), ("float32", other_weights)):
        with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
            analysis.read_dataset_archive(
                path, "wine_quality_white", weights=selected_weights, dtype=dtype,
            )
    with pytest.raises(ValueError, match="All archived dtype documents are required"):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="wine_quality_white", weights=weights,
        )
    mixed = {**documents, "float64": documents["float32"]}
    with pytest.raises(ValueError, match="All archived dtype documents are required"):
        analysis.analyze_dataset(
            mixed, freezes, dataset_id="wine_quality_white", weights=weights, dtype="float32",
        )


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_all_existing_real_data_outputs_remain_byte_identical(dataset_id, weights):
    profile = analysis.workload_spec(dataset_id, weights)["profile"]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], dataset_id, weights=weights,
    )
    if dataset_id == "wine_quality_white" and weights == "uniform":
        result = analysis.analyze(documents["float64"], freezes)
        report = analysis.render_report(result)
    else:
        result = analysis.analyze_dataset(documents, freezes, dataset_id=dataset_id, weights=weights)
        report = analysis.render_dataset_report(result)
    summary = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    assert report.encode("utf-8") == (analysis.ROOT / profile["report"]).read_bytes()
    assert summary.encode("utf-8") == (analysis.ROOT / profile["summary"]).read_bytes()


def test_wine_float32_cli_selects_correct_profile(
    monkeypatch, tmp_path, wine_float32_evidence, wine_float32_result,
):
    weights, _, _ = wine_float32_evidence
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dataset", "wine_quality_white",
        "--dtype", "float32", "--weights", weights,
        "--report", str(report), "--summary", str(summary),
    ])
    analysis.main()
    assert report.read_bytes() == analysis.render_dataset_report(wine_float32_result).encode("utf-8")
    assert json.loads(summary.read_bytes()) == wine_float32_result


@pytest.mark.parametrize("output_option", ["--report", "--summary"])
@pytest.mark.parametrize("profile", [
    spec["profile"] for spec in (
        *analysis.WORKLOADS.values(), *analysis.DISTANCE_WORKLOADS.values(),
        *analysis.WINE_FLOAT32_WORKLOADS.values(),
    )
], ids=["wine-uniform", "abalone-uniform", "wine-distance", "abalone-distance",
        "wine-float32-uniform", "wine-float32-distance"])
def test_float32_cli_protects_all_real_data_archives(monkeypatch, output_option, profile):
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dtype", "float32",
        output_option, str(analysis.ROOT / profile["archive"]),
    ])
    with pytest.raises(ValueError, match="Output would overwrite a raw archive"):
        analysis.main()


def test_abalone_cli_requires_both_archived_dtypes(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dataset", "abalone", "--dtype", "float32",
    ])
    with pytest.raises(ValueError, match="Abalone archives contain both dtypes"):
        analysis.main()
