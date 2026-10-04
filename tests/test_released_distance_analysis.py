"""Offline regressions for archived distance-weighted release comparisons."""

from copy import deepcopy
import json
from statistics import median
import sys

import pytest

from benchmarks import analyze_released_real_data as analysis


@pytest.fixture(scope="module", params=["wine_quality_white", "abalone"])
def distance_evidence(request):
    dataset_id = request.param
    profile = analysis.DISTANCE_WORKLOADS[dataset_id]["profile"]
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], dataset_id, weights="distance",
    )
    return dataset_id, documents, freezes


@pytest.fixture(scope="module")
def distance_result(distance_evidence):
    dataset_id, documents, freezes = distance_evidence
    return analysis.analyze_dataset(
        documents, freezes, dataset_id=dataset_id, weights="distance",
    )


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
def test_uniform_reports_and_summaries_remain_byte_identical(dataset_id):
    profile = analysis.WORKLOADS[dataset_id]["profile"]
    path = analysis.ROOT / profile["archive"]
    if dataset_id == "wine_quality_white":
        data, freezes = analysis.read_archive(path)
        result = analysis.analyze(data, freezes)
        report = analysis.render_report(result)
    else:
        documents, freezes = analysis.read_abalone_archive(path)
        result = analysis.analyze_abalone(documents, freezes)
        report = analysis.render_abalone_report(result)
    summary = json.dumps(
        result, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    assert report.encode("utf-8") == (analysis.ROOT / profile["report"]).read_bytes()
    assert summary.encode("utf-8") == (analysis.ROOT / profile["summary"]).read_bytes()


def test_distance_report_separates_datasets_and_dtypes(
    distance_evidence, distance_result,
):
    dataset_id, documents, _ = distance_evidence
    result = distance_result
    expected_dtypes = (
        ["float64"] if dataset_id == "wine_quality_white"
        else ["float32", "float64"]
    )
    assert result["dataset_id"] == dataset_id
    assert result["weights"] == "distance"
    assert result["dtypes"] == expected_dtypes
    assert list(result["by_dtype"]) == expected_dtypes
    assert result["validation"]["successful_records"] == 27 * len(expected_dtypes)
    assert result["validation"]["expected_records"] == 27 * len(expected_dtypes)
    assert set(documents) == set(expected_dtypes)
    if len(expected_dtypes) == 1:
        assert not any(key.startswith("cross_dtype_") for key in result["validation"])
    else:
        assert result["validation"]["cross_dtype_source_rows_masks_truth_and_scalers_identical"]
    text = analysis.render_dataset_report(result)
    assert 'weights="distance"' in text
    assert "uniform weights; mean aggregation" not in text
    assert "not a new benchmark" in text
    assert text.endswith("\n") and "\r" not in text
    assert "Training rows observed for each feature" in text


def test_distance_arithmetic_uses_matched_records_and_three_seed_errors(
    distance_evidence, distance_result,
):
    _, documents, _ = distance_evidence
    for dtype, section in distance_result["by_dtype"].items():
        records = documents[dtype]["records"]
        indexed = {(r["variant"], r["seed"], r["repeat"]): r for r in records}
        assert section["configuration"]["weights"] == "distance"
        assert "weights" in section["aggregation"]["matching"]
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
            assert comparison["pair_count"] == len(comparison["pairs"]) == 9
            ratios, changes = [], []
            numerator_times, denominator_times = [], []
            for pair in comparison["pairs"]:
                match = pair["match"]
                assert match["weights"] == "distance" and match["dtype"] == dtype
                left = indexed[(numerator, match["seed"], match["repeat"])]
                right = indexed[(denominator, match["seed"], match["repeat"])]
                assert all(left[key] == right[key] == value for key, value in match.items())
                assert left["case"] == right["case"]
                left_time, right_time = left["total_seconds"], right["total_seconds"]
                ratios.append(left_time / right_time)
                changes.append(100 * (right_time / left_time - 1))
                numerator_times.append(left_time)
                denominator_times.append(right_time)
                assert pair["timing"]["total_seconds"]["ratio"] == ratios[-1]
            assert comparison["speedup"]["total_seconds"] == {
                "median": median(ratios), "min": min(ratios), "max": max(ratios),
            }
            assert comparison["denominator_duration_change_percent"]["total_seconds"]["median"] == median(changes)
            assert comparison["faster_pairs"]["total_seconds"] == sum(r > 1 for r in ratios)
            if numerator == "previous" and denominator == "current":
                assert median(ratios) != median(numerator_times) / median(denominator_times)
            samples = comparison["seed_output_agreement"]
            assert len(samples) == 3 and {s["seed"] for s in samples} == {101, 202, 303}
            for sample in samples:
                assert sample["repeat"] == 1
                left = indexed[(numerator, sample["seed"], 1)]
                right = indexed[(denominator, sample["seed"], 1)]
                differences = [abs(a - b) for a, b in zip(
                    left["imputed_values"], right["imputed_values"],
                )]
                assert sample["scored_cells"] == len(differences)
                assert sample["different_hidden_entries"] == sum(d != 0 for d in differences)
                assert sample["hidden_entries_above_1e_minus_5"] == sum(d > 1e-5 for d in differences)
                assert sample["max_scored_abs_difference"] == max(differences)


@pytest.mark.parametrize("location", ["parameters", "plan", "record", "model", "pair"])
@pytest.mark.parametrize("change", ["missing", "uniform"])
def test_missing_or_mixed_distance_weights_are_rejected(
    distance_evidence, location, change,
):
    dataset_id, original, freezes = distance_evidence
    documents = deepcopy(original)
    data = next(iter(documents.values()))
    target = {
        "parameters": data["parameters"],
        "plan": data["planned_configs"][0],
        "record": data["records"][0],
        "model": data["records"][0]["model_parameters"],
        "pair": data["comparisons"][0]["pairs"][0]["match"],
    }[location]
    if change == "missing":
        target.pop("weights")
    else:
        target["weights"] = "uniform"
    with pytest.raises(ValueError):
        analysis.analyze_dataset(
            documents, freezes, dataset_id=dataset_id, weights="distance",
        )


@pytest.mark.parametrize("change", ["summary", "pair", "pair_count"])
def test_distance_stored_aggregates_cannot_override_records(distance_evidence, change):
    dataset_id, original, freezes = distance_evidence
    documents = deepcopy(original)
    data = next(iter(documents.values()))
    if change == "summary":
        data["summaries"][0]["timing_and_memory"]["total_seconds"]["median"] += 1
    elif change == "pair":
        data["comparisons"][0]["pairs"][0]["timing_ratios"]["total_seconds"] += 1
    else:
        data["comparisons"][0]["matched_pairs"] = 8
    with pytest.raises(ValueError, match="Stored"):
        analysis.analyze_dataset(
            documents, freezes, dataset_id=dataset_id, weights="distance",
        )


def test_abalone_dtypes_must_share_the_same_source_rows():
    profile = analysis.ABALONE_DISTANCE_PROFILE
    documents, freezes = analysis.read_dataset_archive(
        analysis.ROOT / profile["archive"], "abalone", weights="distance",
    )
    for record in documents["float32"]["records"]:
        if record["seed"] == 101:
            record["case"]["fingerprints"]["query_row_ids"] = "f" * 64
    with pytest.raises(ValueError, match="Cross-dtype row/mask/truth"):
        analysis.analyze_dataset(
            documents, freezes, dataset_id="abalone", weights="distance",
        )


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
def test_distance_archive_cannot_be_read_as_uniform(dataset_id):
    profile = analysis.DISTANCE_WORKLOADS[dataset_id]["profile"]
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        analysis.read_dataset_archive(analysis.ROOT / profile["archive"], dataset_id)


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
def test_changed_distance_archive_is_rejected_before_parsing(tmp_path, dataset_id):
    path = tmp_path / "changed.zip"
    path.write_bytes(b"This is not the preserved artifact")
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        analysis.read_dataset_archive(path, dataset_id, weights="distance")


@pytest.mark.parametrize("output_option", ["--report", "--summary"])
@pytest.mark.parametrize("profile", [
    analysis.PROFILE, analysis.ABALONE_PROFILE,
    analysis.WINE_DISTANCE_PROFILE, analysis.ABALONE_DISTANCE_PROFILE,
], ids=["wine-uniform", "abalone-uniform", "wine-distance", "abalone-distance"])
def test_distance_cli_cannot_overwrite_any_raw_archive(monkeypatch, output_option, profile):
    monkeypatch.setattr(sys, "argv", [
        "analyze_released_real_data.py", "--dataset", "wine_quality_white",
        "--weights", "distance", output_option, str(analysis.ROOT / profile["archive"]),
    ])
    with pytest.raises(ValueError, match="Output would overwrite a raw archive"):
        analysis.main()
