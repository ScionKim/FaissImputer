"""Offline checks for released-package comparisons on held-out real data."""

from collections import Counter
from copy import deepcopy
from statistics import median

import pytest

from benchmarks import benchmark_released_real_data as benchmark


def make_records(*, dataset_id="wine_quality_white", dtype="float64"):
    records = []
    for index, config in enumerate(
        benchmark.build_configs(
            "0.3.21", "0.3.22", "unused-data-cache",
            dataset_id=dataset_id, dtype=dtype,
        )
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
                "query_mask": {
                    "missing_per_feature": [0, 2] + [0] * (config["features"] - 2),
                },
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


def validation_inputs(dataset_id, dtype):
    config = next(
        row for row in benchmark.build_configs(
            "0.3.21", "0.3.22", "unused-data-cache",
            dataset_id=dataset_id, dtype=dtype,
        )
        if row["variant"] == "current"
    )
    record = next(
        row for row in make_records(dataset_id=dataset_id, dtype=dtype)
        if row["variant"] == "current"
    )
    environment = {
        name: "offline-test"
        for name in (
            *benchmark.COMMON_ENVIRONMENT, "platform", "cpu_model", "git_commit",
        )
    }
    environment["faiss_imputer"] = config["expected_version"]
    names = list(benchmark.UCI_DATASETS[dataset_id]["feature_names"])
    dataset = {"feature_names": names, "dataset_sha256": "a" * 64}
    record.update({
        "environment": deepcopy(environment),
        "dataset": deepcopy(dataset),
        "input_dtype": dtype,
        "output_dtype": dtype,
        "faiss_omp_threads": 1,
        "threadpools": [{"num_threads": 1}],
        "sklearn_working_memory_mib": benchmark.WORKING_MEMORY_MIB,
        "model_parameters": {
            "n_neighbors": 5, "weights": "uniform", "copy": True,
            "metric": "l2", "strategy": "mean",
            "donor_policy": "available", "index_factory": "Flat",
        },
    })
    record["case"].update({
        "mechanism": "MCAR", "input_dtype": dtype, "truth_dtype": "float64",
        "n_train": 3000, "n_query": 1000,
        "nominal_overall_missing_rate": 0.10,
        "feature_names": names,
        "always_observed": [config["mar_driver"]],
        "eligible_base_probability": (
            config["missing_rate"] * config["features"] / (config["features"] - 1)
        ),
        "mar_reference_rows": None,
        "mar_cutoff": None,
        "mar_low_probability": None,
        "mar_high_probability": None,
        "train_mask": {"missing_per_feature": [0] * config["features"]},
    })
    record["feature_quality"] = [
        {
            "feature": name,
            "scored_cells": count,
            "rmse_standardized": 0.1 if count else None,
            "mae_standardized": 0.1 if count else None,
            "rmse_original_units": 0.1 if count else None,
            "mae_original_units": 0.1 if count else None,
        }
        for name, count in zip(
            names, record["case"]["query_mask"]["missing_per_feature"],
        )
    ]
    return record, config, environment, dataset


def test_default_configuration_remains_wine_float64():
    assert benchmark.build_configs("0.3.21", "0.3.22", "unused-data-cache") == (
        benchmark.build_configs(
            "0.3.21", "0.3.22", "unused-data-cache",
            dataset_id="wine_quality_white", dtype="float64",
        )
    )


@pytest.mark.parametrize(
    "dataset_id, features, mar_driver",
    [("wine_quality_white", 11, "alcohol"), ("abalone", 7, "Length")],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_planned_grid_has_balanced_order_and_correct_installed_versions(
    dataset_id, features, mar_driver, dtype
):
    configs = benchmark.build_configs(
        "0.3.21", "0.3.22", "unused-data-cache",
        dataset_id=dataset_id, dtype=dtype,
    )
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
            "dataset_id": dataset_id,
            "api": "fit_then_transform",
            "training_policy": "available",
            "train_size": 3000,
            "query_size": 1000,
            "features": features,
            "n_neighbors": 5,
            "missing_rate": 0.10,
            "mechanism": "MCAR",
            "dtype": dtype,
            "mar_reference_rows": 1000,
            "mar_driver": mar_driver,
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


@pytest.mark.parametrize(
    "overrides", [{"dataset_id": "california_housing"}, {"dtype": "float16"}],
)
def test_unsupported_dataset_or_dtype_is_rejected(overrides):
    with pytest.raises(ValueError):
        benchmark.build_configs("0.3.21", "0.3.22", "unused-data-cache", **overrides)


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_validate_record_accepts_configured_dtype_with_float64_truth(
    dataset_id, dtype
):
    benchmark.validate_record(*validation_inputs(dataset_id, dtype))


@pytest.mark.parametrize(
    "dtype, wrong_dtype", [("float32", "float64"), ("float64", "float32")],
)
@pytest.mark.parametrize("field", ["input_dtype", "output_dtype"])
def test_validate_record_rejects_wrong_input_or_output_dtype(dtype, wrong_dtype, field):
    record, config, environment, dataset = validation_inputs("abalone", dtype)
    record[field] = wrong_dtype
    with pytest.raises(ValueError, match="Unexpected input or output dtype"):
        benchmark.validate_record(record, config, environment, dataset)


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_speedup_uses_matched_ratios_not_ratio_of_medians(dataset_id, dtype):
    records = make_records(dataset_id=dataset_id, dtype=dtype)
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


@pytest.mark.parametrize(
    "other_configuration", [{"dataset_id": "abalone"}, {"dtype": "float32"}],
    ids=["different-dataset", "different-dtype"],
)
@pytest.mark.parametrize(
    "partial", [False, True], ids=["complete-inputs", "nine-mixed-pairs"],
)
@pytest.mark.parametrize("aggregate", [benchmark.summarize, benchmark.compare_records])
def test_aggregates_reject_mixed_configurations(
    other_configuration, partial, aggregate
):
    original = make_records()
    other = make_records(**other_configuration)
    if partial:
        # These incomplete inputs total nine records/pairs per variant,
        # so a count-only completeness check would accept the mixture.
        original = [row for row in original if row["seed"] == 101]
        other = [row for row in other if row["seed"] in (202, 303)]
    records = original + other
    for index, row in enumerate(records):
        row["record_index"] = index
    with pytest.raises(
        ValueError, match="Cannot pool multiple benchmark configurations",
    ):
        aggregate(records)


def test_failed_and_unchecked_other_configurations_do_not_poison_aggregates():
    records = make_records()
    other = make_records(dataset_id="abalone", dtype="float32")
    for index, row in enumerate(other):
        row["record_index"] += len(records)
        if index % 2:
            row["status"] = "validation_error"
        else:
            row["checks_passed"] = False
    mixed = records + other
    assert benchmark.summarize(mixed) == benchmark.summarize(records)
    assert benchmark.compare_records(mixed) == benchmark.compare_records(records)


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


@pytest.mark.parametrize("dataset_id", ["wine_quality_white", "abalone"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_quality_summarizes_seed_datasets_without_counting_timing_repeats(
    dataset_id, dtype
):
    records = make_records(dataset_id=dataset_id, dtype=dtype)
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
