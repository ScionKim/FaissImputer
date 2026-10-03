"""Offline evidence and aggregation regressions for released real-data reports."""

from copy import deepcopy
import json
from statistics import median

import pytest

from benchmarks import analyze_released_real_data as analysis


@pytest.fixture(scope="module")
def evidence():
    return analysis.read_archive(analysis.ROOT / analysis.PROFILE["archive"])


def test_report_uses_paired_times_and_separate_seed_quality(evidence):
    data, freezes = evidence
    result = analysis.analyze(data, freezes)
    comparison = next(item for item in result["comparisons"]
                      if item["numerator_variant"] == "previous")
    previous = {(r["seed"], r["repeat"]): r for r in data["records"]
                if r["variant"] == "previous"}
    current = {(r["seed"], r["repeat"]): r for r in data["records"]
               if r["variant"] == "current"}
    paired = median(previous[key]["total_seconds"] / current[key]["total_seconds"]
                    for key in previous)
    ratio_of_medians = (
        median(r["total_seconds"] for r in previous.values())
        / median(r["total_seconds"] for r in current.values())
    )
    assert comparison["speedup"]["total_seconds"]["median"] == paired
    assert paired != ratio_of_medians
    assert comparison["pair_count"] == 9
    for cell in result["cells"]:
        assert cell["record_count"] == 9
        assert cell["quality_seed_count"] == 3
        assert {r["seed"] for r in cell["quality_samples"]} == {101, 202, 303}
        assert all("feature_quality" in row for row in cell["quality_samples"])
    text = analysis.render_report(result)
    assert text.endswith("\n") and "\r" not in text
    assert "Training rows observed for each feature" in text
    assert "not a new benchmark" in text


def test_existing_wine_outputs_remain_byte_identical(evidence):
    data, freezes = evidence
    result = analysis.analyze(data, freezes)
    report = analysis.render_report(result).encode("utf-8")
    summary = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
    assert report == (analysis.ROOT / analysis.PROFILE["report"]).read_bytes()
    assert summary == (analysis.ROOT / analysis.PROFILE["summary"]).read_bytes()


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "changed_inputs"])
def test_invalid_worker_evidence_is_rejected(evidence, change):
    original, freezes = evidence
    data = deepcopy(original)
    if change == "missing":
        data["records"].pop()
    elif change == "duplicate":
        data["records"][-1] = deepcopy(data["records"][0])
    elif change == "failed":
        data["records"][0]["status"] = "error"
    else:
        data["records"][1]["case"]["fingerprints"]["query"] = "f" * 64
    with pytest.raises(ValueError):
        analysis.analyze(data, freezes)


def test_changed_repeated_output_is_rejected(evidence):
    original, freezes = evidence
    data = deepcopy(original)
    record = next(r for r in data["records"]
                  if r["variant"] == "current" and r["seed"] == 101 and r["repeat"] == 2)
    record["imputed_values"][0] += 1
    with pytest.raises(ValueError, match="across repeats"):
        analysis.analyze(data, freezes)


@pytest.mark.parametrize("change", ["summary", "pair", "pair_count"])
def test_stored_aggregates_cannot_override_record_arithmetic(evidence, change):
    original, freezes = evidence
    data = deepcopy(original)
    if change == "summary":
        data["summaries"][0]["timing_and_memory"]["total_seconds"]["median"] += 1
    elif change == "pair":
        data["comparisons"][0]["pairs"][0]["timing_ratios"]["total_seconds"] += 1
    else:
        data["comparisons"][0]["matched_pairs"] = 8
    with pytest.raises(ValueError, match="Stored"):
        analysis.analyze(data, freezes)


def test_changed_dependencies_are_rejected(evidence):
    data, original_freezes = evidence
    freezes = deepcopy(original_freezes)
    freezes["previous-environment.txt"]["numpy"] = "different-version"
    with pytest.raises(ValueError, match="Dependency mismatch"):
        analysis.analyze(data, freezes)


@pytest.mark.parametrize("reader", [analysis.read_archive, analysis.read_abalone_archive])
def test_changed_archive_is_rejected_before_reading_members(tmp_path, reader):
    archive = tmp_path / "changed.zip"
    archive.write_bytes(b"This is not the preserved artifact")
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        reader(archive)


@pytest.fixture(scope="module")
def abalone_evidence():
    return analysis.read_abalone_archive(
        analysis.ROOT / analysis.ABALONE_PROFILE["archive"]
    )


def test_abalone_aggregates_each_dtype_separately(abalone_evidence):
    documents, freezes = abalone_evidence
    result = analysis.analyze_abalone(documents, freezes)
    assert result["validation"]["successful_records"] == 54
    assert set(result["by_dtype"]) == {"float32", "float64"}
    observed_ratios = {}
    for dtype, section in result["by_dtype"].items():
        assert section["configuration"]["dataset_id"] == "abalone"
        assert section["configuration"]["dtype"] == dtype
        assert section["validation"]["successful_records"] == 27
        assert len(section["cells"]) == len(section["comparisons"]) == 3
        indexed = {
            (row["seed"], row["repeat"], row["variant"]): row
            for row in documents[dtype]["records"]
        }
        for cell in section["cells"]:
            assert cell["record_count"] == 9
            assert cell["quality_seed_count"] == 3
            assert {row["seed"] for row in cell["quality_samples"]} == {101, 202, 303}
            assert all(len(row["feature_quality"]) == 7 for row in cell["quality_samples"])
        for comparison in section["comparisons"]:
            numerator = comparison["numerator_variant"]
            denominator = comparison["denominator_variant"]
            assert comparison["pair_count"] == 9
            assert all(pair["match"]["dtype"] == dtype for pair in comparison["pairs"])
            for phase in analysis.TIMINGS:
                ratios = []
                changes = []
                for seed in (101, 202, 303):
                    for repeat in (1, 2, 3):
                        left = indexed[(seed, repeat, numerator)][phase]
                        right = indexed[(seed, repeat, denominator)][phase]
                        ratios.append(left / right)
                        changes.append(100 * (right / left - 1))
                assert comparison["speedup"][phase]["median"] == median(ratios)
                assert comparison["denominator_duration_change_percent"][phase]["median"] == median(changes)
            if numerator == "previous":
                observed_ratios[dtype] = comparison["speedup"]["total_seconds"]["median"]
    assert observed_ratios["float32"] != observed_ratios["float64"]
    text = analysis.render_abalone_report(result)
    assert text.endswith("\n") and "\r" not in text
    assert text.index("## float32") < text.index("## float64")
    assert "Training rows observed for each feature" in text
    assert "not a new benchmark" in text
    assert "KNN takes less time in every matched" not in text


def test_abalone_output_counts_use_seeds_without_timing_duplicates(abalone_evidence):
    documents, freezes = abalone_evidence
    result = analysis.analyze_abalone(documents, freezes)
    for dtype, expected_count in (("float32", 2), ("float64", 4)):
        section = result["by_dtype"][dtype]
        release = next(c for c in section["comparisons"]
                       if c["numerator_variant"] == "previous")
        assert release["matching_output_hashes"] == 9
        assert release["max_scored_abs_difference"] == 0
        comparison = next(c for c in section["comparisons"]
                          if c["numerator_variant"] == "knn"
                          and c["denominator_variant"] == "current")
        samples = comparison["seed_output_agreement"]
        assert len(samples) == 3
        assert [row["seed"] for row in samples] == [101, 202, 303]
        assert all(row["repeat"] == 1 for row in samples)
        assert [row["hidden_entries_above_1e_minus_5"] for row in samples] == [expected_count, 0, 0]
        for row in samples:
            left = documents[dtype]["records"][row["numerator_record_index"]]
            right = documents[dtype]["records"][row["denominator_record_index"]]
            differences = [abs(a - b) for a, b in zip(
                left["imputed_values"], right["imputed_values"])]
            assert row["max_scored_abs_difference"] == max(differences)
            assert row["different_hidden_entries"] == sum(x != 0 for x in differences)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "summary", "pair"])
def test_invalid_abalone_evidence_in_either_dtype_is_rejected(
    abalone_evidence, dtype, change
):
    original, freezes = abalone_evidence
    documents = deepcopy(original)
    data = documents[dtype]
    if change == "missing":
        data["records"].pop()
    elif change == "duplicate":
        data["records"][-1] = deepcopy(data["records"][0])
    elif change == "failed":
        data["records"][0]["checks_passed"] = False
    elif change == "summary":
        data["summaries"][0]["timing_and_memory"]["total_seconds"]["median"] += 1
    else:
        data["comparisons"][0]["pairs"][0]["timing_ratios"]["total_seconds"] += 1
    with pytest.raises(ValueError):
        analysis.analyze_abalone(documents, freezes)


@pytest.mark.parametrize("change", ["missing_dtype", "swapped_documents", "mixed_record"])
def test_abalone_dtype_boundaries_are_enforced(abalone_evidence, change):
    original, freezes = abalone_evidence
    documents = deepcopy(original)
    if change == "missing_dtype":
        del documents["float64"]
    elif change == "swapped_documents":
        documents["float32"], documents["float64"] = (
            documents["float64"], documents["float32"]
        )
    else:
        documents["float32"]["records"][0] = deepcopy(documents["float64"]["records"][0])
    with pytest.raises(ValueError):
        analysis.analyze_abalone(documents, freezes)


def test_cross_dtype_inputs_are_checked_after_within_dtype_validation(abalone_evidence):
    original, freezes = abalone_evidence
    documents = deepcopy(original)
    for row in documents["float64"]["records"]:
        if row["seed"] == 101:
            row["case"]["fingerprints"]["query_row_ids"] = "f" * 64
    with pytest.raises(ValueError, match="Cross-dtype"):
        analysis.analyze_abalone(documents, freezes)
