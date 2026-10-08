"""Saved-archive regressions for separate Wine and Abalone MAR results."""

from copy import deepcopy
import json
from statistics import median
import sys

import pytest

from benchmarks import analyze_released_real_data as analysis


@pytest.fixture(scope="module", params=["uniform", "distance"])
def mar_evidence(request):
    weights = request.param
    profile = analysis.WINE_MAR_PROFILES[weights]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], "wine_quality_white",
        weights=weights, mechanism="MAR",
    )
    return weights, documents, freezes


@pytest.fixture(scope="module")
def mar_result(mar_evidence):
    weights, documents, freezes = mar_evidence
    return analysis.analyze_dataset(
        documents, freezes, dataset_id="wine_quality_white",
        weights=weights, mechanism="MAR",
    )


def test_mar_provenance_cases_and_report(mar_evidence, mar_result):
    weights, documents, _ = mar_evidence
    profile = analysis.WINE_MAR_PROFILES[weights]
    result = mar_result
    section = result["by_dtype"]["float64"]
    assert result["mechanism"] == section["configuration"]["mechanism"] == "MAR"
    assert result["dtypes"] == list(result["by_dtype"]) == ["float64"]
    assert section["configuration"].get("weights", "uniform") == weights
    assert result["source"]["archive"] == profile["archive"]
    assert result["source"]["archive_sha256"] == profile["archive_sha256"]
    assert result["source"]["github_run_id"] == profile["run"]
    assert result["source"]["json_members"]["float64"]["sha256"] == profile["json_sha256"]
    assert result["validation"]["successful_records"] == 27
    assert not any(key.startswith("cross_dtype_") for key in result["validation"])
    assert result["environment"] == section["environment"]
    assert result["environment"]["cpu_model"] == documents["float64"]["metadata"]["cpu_model"]
    raw_cases = {r["seed"]: r["case"] for r in documents["float64"]["records"]}
    expected_cutoffs = {101: 10.350000000000001, 202: 10.3, 303: 10.3}
    expected_donors = {101: 1096, 202: 1099, 303: 1084}
    assert [case["seed"] for case in section["cases"]] == [101, 202, 303]
    for case in section["cases"]:
        original = raw_cases[case["seed"]]
        assert {key: case[key] for key in original} == original
        assert case["mar_cutoff"] == expected_cutoffs[case["seed"]]
        assert case["mar_reference_rows"] == 1000
        assert case["mar_low_probability"] == 0.05500000000000001
        assert case["mar_high_probability"] == 0.16500000000000004
        assert case["always_observed"] == ["alcohol"]
        assert case["train_mask"]["missing_per_feature"][-1] == 0
        assert case["query_mask"]["missing_per_feature"][-1] == 0
        assert case["complete_donors"] == expected_donors[case["seed"]]
        assert case["observed_donors_per_feature"] == [
            3000 - count for count in original["train_mask"]["missing_per_feature"]
        ]
    report = analysis.render_dataset_report(result)
    assert "MAR" in report.splitlines()[0]
    assert "### MAR missingness by seed" in report
    assert "cutoff" in report.lower() and "probabilit" in report.lower()
    assert "alcohol" in report and "Complete donors" in report
    assert "Training rows observed for each feature" in report
    assert "output agreement" in report.lower()
    assert "--mechanism MAR" in report and f"--weights {weights}" in report
    assert profile["archive"] in report and profile["summary"] in report
    assert "Only float64 is measured in this run." in report
    assert "not a new benchmark" in report
    assert report.endswith("\n") and "\r" not in report


def test_mar_aggregates_record_timings_and_three_quality_seeds(mar_evidence, mar_result):
    weights, documents, _ = mar_evidence
    records = documents["float64"]["records"]
    indexed = {(r["variant"], r["seed"], r["repeat"]): r for r in records}
    section = mar_result["by_dtype"]["float64"]
    assert {cell["variant"] for cell in section["cells"]} == {"knn", "previous", "current"}
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
        numerator, denominator = comparison["numerator_variant"], comparison["denominator_variant"]
        assert comparison["pair_count"] == len(comparison["pairs"]) == 9
        matched = []
        for pair in comparison["pairs"]:
            match = pair["match"]
            left = indexed[(numerator, match["seed"], match["repeat"])]
            right = indexed[(denominator, match["seed"], match["repeat"])]
            assert match["dtype"] == "float64" and match["mechanism"] == "MAR"
            assert all(left[key] == right[key] == value for key, value in match.items())
            assert left.get("weights", "uniform") == right.get("weights", "uniform") == weights
            assert left["case"] == right["case"]
            matched.append((left, right))
        for field in ("fit_seconds", "transform_seconds", "total_seconds"):
            ratios = [left[field] / right[field] for left, right in matched]
            assert comparison["speedup"][field] == {
                "median": median(ratios), "min": min(ratios), "max": max(ratios),
            }
        assert comparison["matching_output_hashes"] == sum(
            left["output_sha256"] == right["output_sha256"] for left, right in matched
        )
        assert [item["seed"] for item in comparison["seed_output_agreement"]] == [101, 202, 303]
        for sample in comparison["seed_output_agreement"]:
            left = indexed[(numerator, sample["seed"], 1)]
            right = indexed[(denominator, sample["seed"], 1)]
            differences = [abs(a - b) for a, b in zip(left["imputed_values"], right["imputed_values"])]
            assert sample["repeat"] == 1
            assert sample["scored_cells"] == len(differences)
            assert sample["hidden_entries_above_1e_minus_5"] == sum(d > 1e-5 for d in differences)
            assert sample["max_scored_abs_difference"] == max(differences)
            assert sample["denominator_minus_numerator_rmse"] == right["rmse"] - left["rmse"]
            assert sample["denominator_minus_numerator_mae"] == right["mae"] - left["mae"]


@pytest.mark.parametrize("change", [
    "missing_parameter_mechanism", "parameter_mechanism", "parameter_dtype",
    "record_mechanism", "case_mechanism", "mixed_weights", "missing", "duplicate",
    "failed", "changed_inputs", "changed_cutoff", "changed_repeat",
    "stored_summary", "stored_quality", "stored_pair", "stored_ratio",
])
def test_mar_rejects_inconsistent_evidence(mar_evidence, change):
    weights, original, freezes = mar_evidence
    documents = deepcopy(original)
    data = documents["float64"]
    if change == "missing_parameter_mechanism":
        del data["parameters"]["mechanism"]
    elif change == "parameter_mechanism":
        data["parameters"]["mechanism"] = "MCAR"
    elif change == "parameter_dtype":
        data["parameters"]["dtype"] = "float32"
    elif change == "record_mechanism":
        data["records"][0]["mechanism"] = "MCAR"
    elif change == "case_mechanism":
        data["records"][0]["case"]["mechanism"] = "MCAR"
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
    elif change == "changed_cutoff":
        data["records"][0]["case"]["mar_cutoff"] += 1
    elif change == "changed_repeat":
        record = next(r for r in data["records"]
                      if r["variant"] == "current" and r["seed"] == 101 and r["repeat"] == 2)
        record["imputed_values"][0] += 1
    elif change == "stored_summary":
        data["summaries"][0]["timing_and_memory"]["total_seconds"]["median"] += 1
    elif change == "stored_quality":
        data["summaries"][0]["quality"]["rmse"]["median"] += 1
    elif change == "stored_pair":
        data["comparisons"][0]["pairs"][0]["timing_ratios"]["total_seconds"] += 1
    else:
        data["comparisons"][0]["timing_ratios"]["total_seconds"]["median"] += 1
    with pytest.raises(ValueError):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="wine_quality_white",
            weights=weights, mechanism="MAR",
        )


@pytest.mark.parametrize("field,value", [
    ("mar_reference_rows", None), ("mar_reference_rows", 999),
    ("mar_low_probability", 0.1), ("mar_high_probability", 0.1),
    ("mar_cutoff", None), ("mar_cutoff", float("nan")),
    ("mar_cutoff", float("inf")), ("mar_cutoff", True),
    ("driver_missing", 1),
])
def test_mar_checks_metadata_even_when_repeats_agree(mar_evidence, field, value):
    weights, original, freezes = mar_evidence
    documents = deepcopy(original)
    for record in documents["float64"]["records"]:
        if record["seed"] != 101:
            continue
        case = record["case"]
        if field == "driver_missing":
            counts = case["train_mask"]["missing_per_feature"]
            counts[-1] = value
            counts[0] -= value  # Preserve totals to isolate the always-observed driver check.
        else:
            case[field] = value
    with pytest.raises(ValueError):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="wine_quality_white",
            weights=weights, mechanism="MAR",
        )


def test_mar_archive_hash_prevents_cross_mechanism_and_weight_routing(mar_evidence, tmp_path):
    weights, documents, freezes = mar_evidence
    mar_path = analysis.ROOT / analysis.WINE_MAR_PROFILES[weights]["archive"]
    mcar_path = analysis.ROOT / analysis.workload_spec("wine_quality_white", weights)["profile"]["archive"]
    other_weights = "distance" if weights == "uniform" else "uniform"
    for path, mechanism, selected_weights in (
        (mar_path, "MCAR", weights), (mcar_path, "MAR", weights),
        (mar_path, "MAR", other_weights),
    ):
        with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
            analysis.read_dataset_archive(
                path, "wine_quality_white", weights=selected_weights, mechanism=mechanism,
            )
    altered = tmp_path / "altered.zip"
    altered.write_bytes(mar_path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        analysis.read_dataset_archive(altered, "wine_quality_white", weights=weights, mechanism="MAR")
    with pytest.raises(ValueError):
        analysis.analyze_dataset(documents, freezes, dataset_id="wine_quality_white", weights=weights)


def test_mar_rejects_unsupported_wine_dtype_before_reading():
    dataset_id, dtype = "wine_quality_white", "float32"
    with pytest.raises(ValueError):
        analysis.workload_spec(dataset_id, dtype=dtype, mechanism="MAR")
    with pytest.raises(ValueError):
        analysis.read_dataset_archive(
            analysis.ROOT / "nonexistent-mar-input.zip", dataset_id, dtype=dtype, mechanism="MAR",
        )
    with pytest.raises(ValueError):
        analysis.analyze_dataset({}, {}, dataset_id=dataset_id, dtype=dtype, mechanism="MAR")


@pytest.mark.parametrize("dtype", [None, "float64"])
def test_mar_cli_selects_wrapper_and_correct_archive(monkeypatch, tmp_path, mar_evidence, mar_result, dtype):
    weights, _, _ = mar_evidence
    report, summary = tmp_path / "report.md", tmp_path / "summary.json"
    argv = [
        "analyze_released_real_data.py", "--dataset", "wine_quality_white",
        "--mechanism", "MAR", "--weights", weights,
        "--report", str(report), "--summary", str(summary),
    ]
    if dtype is not None:
        argv.extend(["--dtype", dtype])
    monkeypatch.setattr(sys, "argv", argv)
    analysis.main()
    assert report.read_bytes() == analysis.render_dataset_report(mar_result).encode("utf-8")
    assert json.loads(summary.read_bytes()) == mar_result


@pytest.mark.parametrize("output_option", ["--report", "--summary"])
@pytest.mark.parametrize("selected_mechanism,profile", [
    ("MAR", spec["profile"]) for spec in (
        *analysis.WORKLOADS.values(), *analysis.DISTANCE_WORKLOADS.values(),
        *analysis.WINE_FLOAT32_WORKLOADS.values(),
    )
] + [("MCAR", profile) for profile in (
    *analysis.WINE_MAR_PROFILES.values(), *analysis.ABALONE_MAR_PROFILES.values(),
)])
def test_cli_protects_archives_across_mechanisms(monkeypatch, output_option, selected_mechanism, profile):
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--mechanism", selected_mechanism,
        output_option, str(analysis.ROOT / profile["archive"]),
    ])
    with pytest.raises(ValueError, match="Output would overwrite a raw archive"):
        analysis.main()


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_existing_wine_float32_outputs_remain_byte_identical(weights):
    # The preceding float32 regression module already covers the four older profiles.
    profile = analysis.WINE_FLOAT32_PROFILES[weights]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], "wine_quality_white", weights=weights, dtype="float32",
    )
    result = analysis.analyze_dataset(
        documents, freezes, dataset_id="wine_quality_white", weights=weights, dtype="float32",
    )
    assert "mechanism" not in result
    assert result["by_dtype"]["float32"]["configuration"]["mechanism"] == "MCAR"
    summary = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    assert summary.encode("utf-8") == (analysis.ROOT / profile["summary"]).read_bytes()
    assert analysis.render_dataset_report(result).encode("utf-8") == (analysis.ROOT / profile["report"]).read_bytes()


def test_existing_wine_mar_outputs_remain_byte_identical(mar_evidence, mar_result):
    weights, _, _ = mar_evidence
    profile = analysis.WINE_MAR_PROFILES[weights]
    summary = json.dumps(mar_result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    assert summary.encode("utf-8") == (analysis.ROOT / profile["summary"]).read_bytes()
    assert analysis.render_dataset_report(mar_result).encode("utf-8") == (
        analysis.ROOT / profile["report"]
    ).read_bytes()


@pytest.fixture(scope="module", params=["uniform", "distance"])
def abalone_mar_evidence(request):
    weights = request.param
    profile = analysis.ABALONE_MAR_PROFILES[weights]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], "abalone", weights=weights, mechanism="MAR",
    )
    result = analysis.analyze_dataset(
        documents, freezes, dataset_id="abalone", weights=weights, mechanism="MAR",
    )
    return weights, documents, freezes, result


def test_abalone_mar_keeps_two_dtype_evidence_separate(abalone_mar_evidence):
    weights, documents, _, result = abalone_mar_evidence
    profile = analysis.ABALONE_MAR_PROFILES[weights]
    spec = analysis.workload_spec("abalone", weights, mechanism="MAR")
    assert result["dataset_id"] == "abalone" and result["mechanism"] == "MAR"
    assert result["dtypes"] == list(result["by_dtype"]) == ["float32", "float64"]
    assert result["validation"]["successful_records"] == 54
    assert result["validation"]["cross_dtype_source_rows_masks_truth_and_scalers_identical"]
    assert result["validation"]["cross_dtype_environment_identical_except_creation_time"]
    assert result["source"]["archive"] == profile["archive"]
    assert result["source"]["archive_sha256"] == profile["archive_sha256"]
    assert result["source"]["github_run_id"] == profile["run"]
    for dtype in ("float32", "float64"):
        section = result["by_dtype"][dtype]
        records = documents[dtype]["records"]
        indexed = {(r["variant"], r["seed"], r["repeat"]): r for r in records}
        assert section["configuration"]["dtype"] == dtype
        assert section["configuration"]["mechanism"] == "MAR"
        assert section["configuration"]["weights"] == weights
        assert result["source"]["json_members"][dtype]["sha256"] == spec["json_members"][dtype]["sha256"]
        assert section["environment"]["cpu_model"] == documents[dtype]["metadata"]["cpu_model"]
        raw_cases = {r["seed"]: r["case"] for r in records}
        for case in section["cases"]:
            assert {key: case[key] for key in raw_cases[case["seed"]]} == raw_cases[case["seed"]]
            assert case["mar_cutoff"] == {101: 0.54, 202: 0.545, 303: 0.55}[case["seed"]]
            assert case["mar_reference_rows"] == 1000
            assert case["always_observed"] == ["Length"]
            assert case["train_mask"]["missing_per_feature"][0] == 0
            assert case["query_mask"]["missing_per_feature"][0] == 0
            assert case["complete_donors"] == {101: 1536, 202: 1512, 303: 1490}[case["seed"]]
            assert case["observed_donors_per_feature"] == [
                3000 - count for count in case["train_mask"]["missing_per_feature"]
            ]
        for cell in section["cells"]:
            group = [r for r in records if r["variant"] == cell["variant"]]
            seed_group = [r for r in group if r["repeat"] == 1]
            assert cell["record_count"] == len(group) == 9
            assert cell["quality_seed_count"] == len(seed_group) == 3
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
            numerator, denominator = comparison["numerator_variant"], comparison["denominator_variant"]
            assert comparison["pair_count"] == len(comparison["pairs"]) == 9
            for pair in comparison["pairs"]:
                assert pair["match"]["dtype"] == dtype
                assert pair["match"]["mechanism"] == "MAR"
            pairs = [(indexed[(numerator, seed, repeat)], indexed[(denominator, seed, repeat)])
                     for seed in (101, 202, 303) for repeat in (1, 2, 3)]
            for field in ("fit_seconds", "transform_seconds", "total_seconds"):
                ratios = [left[field] / right[field] for left, right in pairs]
                assert comparison["speedup"][field] == {
                    "median": median(ratios), "min": min(ratios), "max": max(ratios),
                }
            if numerator == "previous" and denominator == "current":
                assert comparison["matching_output_hashes"] == 9
                assert comparison["max_scored_abs_difference"] == 0
            for sample in comparison["seed_output_agreement"]:
                left = indexed[(numerator, sample["seed"], 1)]
                right = indexed[(denominator, sample["seed"], 1)]
                differences = [abs(a - b) for a, b in zip(left["imputed_values"], right["imputed_values"])]
                assert sample["hidden_entries_above_1e_minus_5"] == sum(d > 1e-5 for d in differences)
                assert sample["max_scored_abs_difference"] == max(differences)
    report = analysis.render_dataset_report(result)
    assert "float32 and float64, MAR" in report.splitlines()[0]
    assert "### MAR missingness by seed" in report and "Length" in report
    assert "different CPU models" in report
    assert "--dataset abalone --weights " + weights + " --mechanism MAR" in report
    assert "--dtype" not in report
    assert "bench/released-abalone-mar-0.3.22" in report
    assert profile["archive"] in report and profile["summary"] in report
    assert report.endswith("\n") and "\r" not in report


@pytest.mark.parametrize("change", ["missing_dtype", "cutoff", "scaler", "truth"])
def test_abalone_mar_rejects_cross_dtype_input_changes(abalone_mar_evidence, change):
    weights, original, freezes, _ = abalone_mar_evidence
    documents = deepcopy(original)
    if change == "missing_dtype":
        del documents["float64"]
    else:
        # Change all methods/repeats for one seed so per-dtype consistency still holds.
        for record in documents["float64"]["records"]:
            if record["seed"] != 101:
                continue
            case = record["case"]
            if change == "cutoff":
                case["mar_cutoff"] += 1
            elif change == "scaler":
                case["scaler_mean"][0] += 1
            else:
                case["fingerprints"]["truth"] = "f" * 64
    with pytest.raises(ValueError, match="dtype"):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="abalone", weights=weights, mechanism="MAR",
        )


def test_abalone_mar_archive_routing_rejects_other_evidence(abalone_mar_evidence):
    weights, _, _, _ = abalone_mar_evidence
    path = analysis.ROOT / analysis.ABALONE_MAR_PROFILES[weights]["archive"]
    other_weights = "distance" if weights == "uniform" else "uniform"
    for dataset_id, selected_weights, mechanism in (
        ("abalone", weights, "MCAR"), ("abalone", other_weights, "MAR"),
        ("wine_quality_white", weights, "MAR"),
    ):
        with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
            analysis.read_dataset_archive(
                path, dataset_id, weights=selected_weights, mechanism=mechanism,
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_abalone_mar_outer_interfaces_require_both_dtypes(dtype, monkeypatch):
    # Internal per-dtype validation is supported; public archive selection keeps both.
    assert set(analysis.workload_spec("abalone", dtype=dtype, mechanism="MAR")["json_members"]) == {
        "float32", "float64",
    }
    with pytest.raises(ValueError, match="omit dtype"):
        analysis.read_dataset_archive(
            analysis.ROOT / "nonexistent-mar-input.zip", "abalone", dtype=dtype, mechanism="MAR",
        )
    with pytest.raises(ValueError, match="omit dtype"):
        analysis.analyze_dataset({}, {}, dataset_id="abalone", dtype=dtype, mechanism="MAR")
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dataset", "abalone", "--mechanism", "MAR",
        "--dtype", dtype,
    ])
    with pytest.raises(ValueError, match="omit --dtype"):
        analysis.main()


def test_abalone_mar_cli_selects_both_dtypes(abalone_mar_evidence, monkeypatch, tmp_path):
    weights, _, _, result = abalone_mar_evidence
    report, summary = tmp_path / "report.md", tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dataset", "abalone", "--weights", weights,
        "--mechanism", "MAR", "--report", str(report), "--summary", str(summary),
    ])
    analysis.main()
    assert report.read_bytes() == analysis.render_dataset_report(result).encode("utf-8")
    assert json.loads(summary.read_bytes()) == result
