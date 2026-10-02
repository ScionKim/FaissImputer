"""Offline checks for released-package comparisons on held-out real data."""

from collections import Counter
from copy import deepcopy
from statistics import median

import pytest

from benchmarks import benchmark_released_real_data as benchmark


def make_records():
    records = []
    for index, config in enumerate(
        benchmark.build_configs("0.3.21", "0.3.22", "unused-data-cache")
    ):
        seed = config["seed"]
        records.append({
            **config,
            "record_index": index,
            "status": "ok",
            "checks_passed": True,
            "fit_seconds": 0.25,
            "transform_seconds": 0.75,
            "total_seconds": 1.0,
            "worker_peak_rss_mib": 100.0,
            "rmse": seed / 1000,
            "mae": seed / 2000,
            "scored_cells": 2,
            "imputed_values": [float(seed), float(seed + 1)],
            "output_sha256": f"{seed:064x}",
            "case": {
                "seed": seed,
                "query_mask": {"missing_per_feature": [2] + [0] * 10},
                "fingerprints": {
                    name: (
                        "a" * 64 if name == "dataset"
                        else f"{seed * 100 + position:064x}"
                    )
                    for position, name in enumerate(benchmark.FINGERPRINTS)
                },
            },
        })
    return records


def previous_current_pair():
    return [
        row for row in make_records()
        if row["seed"] == 101 and row["repeat"] == 1
        and row["variant"] in ("previous", "current")
    ]


def comparison(records, numerator="previous", denominator="current"):
    return next(
        item for item in benchmark.compare_records(records)
        if item["numerator_variant"] == numerator
        and item["denominator_variant"] == denominator
    )


def test_planned_grid_has_balanced_order_and_correct_installed_versions():
    configs = benchmark.build_configs("0.3.21", "0.3.22", "unused-data-cache")
    assert len(configs) == 27
    assert Counter(row["variant"] for row in configs) == {
        "previous": 9, "current": 9, "knn": 9,
    }
    groups = {}
    for config in configs:
        groups.setdefault((config["seed"], config["repeat"]), []).append(config)
        assert config["expected_version"] == (
            "0.3.21" if config["variant"] == "previous" else "0.3.22"
        )
        assert config["method"] == (
            "KNNImputer" if config["variant"] == "knn"
            else "FaissImputer[available]"
        )
        for name, value in {
            "dataset_id": "wine_quality_white",
            "api": "fit_then_transform",
            "training_policy": "available",
            "train_size": 3000,
            "query_size": 1000,
            "features": 11,
            "n_neighbors": 5,
            "missing_rate": 0.10,
            "mechanism": "MCAR",
            "dtype": "float64",
            "threads": 1,
        }.items():
            assert config[name] == value
    assert set(groups) == {
        (seed, repeat)
        for seed in (101, 202, 303) for repeat in (1, 2, 3)
    }
    for rows in groups.values():
        assert {row["variant"] for row in rows} == {"previous", "current", "knn"}
    for position in range(3):
        assert Counter(rows[position]["variant"] for rows in groups.values()) == {
            "previous": 3, "current": 3, "knn": 3,
        }


def test_speedup_uses_matched_ratios_not_ratio_of_medians():
    records = make_records()
    durations = {
        "previous": {101: 1.0, 202: 10.0, 303: 100.0},
        "current": {101: 2.0, 202: 10.0, 303: 1.0},
    }
    for row in records:
        if row["variant"] in durations:
            duration = durations[row["variant"]][row["seed"]]
            row.update(
                fit_seconds=duration / 2,
                transform_seconds=duration,
                total_seconds=duration * 1.5,
            )
    result = comparison(records)
    assert result["complete"]
    assert result["matched_pairs"] == 9
    assert result["timing_ratios"]["transform_seconds"]["median"] == 1.0
    ratio_of_medians = median(durations["previous"].values()) / median(
        durations["current"].values()
    )
    assert ratio_of_medians == 5.0
    by_index = {row["record_index"]: row for row in records}
    for pair in result["pairs"]:
        left = by_index[pair["numerator_record_index"]]
        right = by_index[pair["denominator_record_index"]]
        assert left["seed"] == right["seed"]
        assert left["repeat"] == right["repeat"]
        assert pair["timing_ratios"]["transform_seconds"] == (
            left["transform_seconds"] / right["transform_seconds"]
        )


@pytest.mark.parametrize(
    "field, other_value",
    [
        ("dataset_id", "abalone"),
        ("dtype", "float32"),
        ("seed", 202),
        ("repeat", 2),
        ("n_neighbors", 1),
        ("api", "fit_transform"),
    ],
)
def test_matching_does_not_cross_inputs_or_configuration(field, other_value):
    records = previous_current_pair()
    current = next(row for row in records if row["variant"] == "current")
    current[field] = other_value
    result = comparison(records)
    assert result["matched_pairs"] == 0
    assert not result["complete"]
    assert result["timing_ratios"]["total_seconds"]["median"] is None


def test_equal_configuration_with_different_input_fingerprint_is_rejected():
    records = previous_current_pair()
    current = next(row for row in records if row["variant"] == "current")
    current["case"]["fingerprints"]["train"] = "f" * 64
    with pytest.raises(ValueError, match="different prepared inputs"):
        benchmark.compare_records(records)


def test_failed_and_unchecked_workers_are_excluded_and_mark_results_incomplete():
    records = make_records()
    current = [row for row in records if row["variant"] == "current"]
    current[0]["status"] = "validation_error"
    current[1]["checks_passed"] = False
    for row in current[:2]:
        row["total_seconds"] = 1000000.0
    result = comparison(records)
    assert result["matched_pairs"] == 7
    assert not result["complete"]
    assert result["timing_ratios"]["total_seconds"]["median"] == 1.0
    unaffected = comparison(records, "knn", "previous")
    assert unaffected["matched_pairs"] == 9
    assert unaffected["complete"]
    summary = next(
        item for item in benchmark.summarize(records)
        if item["variant"] == "current"
    )
    assert summary["successful_records"] == 7
    assert not summary["complete"]
    assert summary["timing_and_memory"]["total_seconds"] == {
        "count": 7, "median": 1.0, "min": 1.0, "max": 1.0,
    }


def test_duplicate_successful_record_is_rejected():
    records = make_records()
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="Duplicate successful"):
        benchmark.compare_records(records)


def test_missing_worker_cannot_be_reported_as_a_complete_comparison():
    records = [
        row for row in make_records()
        if not (row["variant"] == "current" and row["seed"] == 303
                and row["repeat"] == 3)
    ]
    result = comparison(records)
    assert result["expected_pairs"] == 9
    assert result["matched_pairs"] == 8
    assert not result["complete"]


def test_quality_summarizes_seed_datasets_without_counting_timing_repeats():
    records = make_records()
    by_index = {row["record_index"]: row for row in records}
    for summary in benchmark.summarize(records):
        assert summary["complete"]
        assert summary["successful_records"] == 9
        assert summary["timing_and_memory"]["total_seconds"]["count"] == 9
        assert summary["quality_seed_count"] == 3
        selected = [by_index[index] for index in summary["quality_record_indices"]]
        assert {row["seed"] for row in selected} == {101, 202, 303}
        assert all(row["repeat"] == 1 for row in selected)
        assert summary["quality"]["rmse"] == {
            "count": 3, "median": 0.202, "min": 0.101, "max": 0.303,
        }
        assert summary["quality"]["mae"]["count"] == 3
