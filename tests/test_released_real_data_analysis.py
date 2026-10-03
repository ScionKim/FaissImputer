"""Offline evidence and aggregation regressions for the Wine release report."""

from copy import deepcopy
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


def test_changed_archive_is_rejected_before_reading_members(tmp_path):
    archive = tmp_path / "changed.zip"
    archive.write_bytes(b"This is not the preserved artifact")
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        analysis.read_archive(archive)
