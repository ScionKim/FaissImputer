"""Regenerate the archived Wine release comparison without running imputers."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import math
import re
from zipfile import ZipFile

if __package__:
    from .analyze_released_versions import (
        ROOT, ENVIRONMENT_FIELDS, check_module, finite, formatted,
        parse_freeze, reject_constant, require, stats, table, unique_object,
    )
else:
    from analyze_released_versions import (
        ROOT, ENVIRONMENT_FIELDS, check_module, finite, formatted,
        parse_freeze, reject_constant, require, stats, table, unique_object,
    )


PROFILE = {
    "current": "0.3.22",
    "previous": "0.3.21",
    "labels": {
        "knn": "KNNImputer", "previous": "FaissImputer 0.3.21",
        "current": "FaissImputer 0.3.22",
    },
    "archive": "benchmarks/results/released_wine_quality_0.3.22.zip",
    "report": "docs/benchmarks/released_wine_quality_0.3.22.md",
    "summary": "benchmarks/results/released_wine_quality_0.3.22-summary.json",
    "archive_sha256": "d6a2c52875db764124fbea5b312138356476b7ea55d503b3ee0ae2a514db0a39",
    "json_sha256": "3bc7e665f8ddfb66b453dc930f1eedd2a92e716701e3b18ce9b5a63b0fb2bf1d",
    "commit": "2e07c594b60f7b755c726d991a6548adae6f0331",
    "run": "37076320747",
}
MEMBER = "version_comparison_wine_quality_white.json"
DATASET_MEMBER = "datasets/wine_quality_white.zip"
SOURCE_MEMBER = "winequality-white.csv"
VARIANTS = ("knn", "previous", "current")
SEEDS = (101, 202, 303)
REPEATS = (1, 2, 3)
TIMINGS = ("fit_seconds", "transform_seconds", "total_seconds")
FINGERPRINTS = (
    "dataset", "train_row_ids", "query_row_ids", "raw_train", "raw_query",
    "train_mask", "query_mask", "train", "query", "truth",
)
COMMON_CONFIG = {
    "dataset_id": "wine_quality_white", "api": "fit_then_transform",
    "training_policy": "available", "train_size": 3000, "query_size": 1000,
    "features": 11, "n_neighbors": 5, "missing_rate": 0.1,
    "mechanism": "MCAR", "dtype": "float64", "mar_reference_rows": 1000,
    "mar_driver": "alcohol", "threads": 1,
}
MATCH_FIELDS = tuple(COMMON_CONFIG) + ("seed", "repeat")


def read_archive(path):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == PROFILE["archive_sha256"], "Archive SHA-256 mismatch")
    expected = {
        MEMBER, DATASET_MEMBER, "current-environment.txt",
        "previous-environment.txt", "shared-dependencies.txt",
    }
    with ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(expected) and set(names) == expected,
                "Unexpected or duplicate archive members")
        raw_json = archive.read(MEMBER)
        require(sha256(raw_json).hexdigest() == PROFILE["json_sha256"],
                "Raw JSON SHA-256 mismatch")
        data = json.loads(raw_json, parse_constant=reject_constant,
                          object_pairs_hook=unique_object)
        freezes = {
            name: parse_freeze(archive.read(name).decode("utf-8"))
            for name in sorted(expected - {MEMBER, DATASET_MEMBER})
        }
        dataset_bytes = archive.read(DATASET_MEMBER)
    dataset = data["dataset"]
    require(sha256(dataset_bytes).hexdigest() == dataset["source_archive_sha256"],
            "Source dataset ZIP SHA-256 mismatch")
    require(dataset["source_file"] == SOURCE_MEMBER, "Unexpected dataset source member")
    with ZipFile(BytesIO(dataset_bytes)) as archive:
        require(archive.namelist().count(SOURCE_MEMBER) == 1,
                "Missing or duplicate source CSV")
        require(sha256(archive.read(SOURCE_MEMBER)).hexdigest() == dataset["source_file_sha256"],
                "Source CSV SHA-256 mismatch")
    return data, freezes


def counted_stats(values):
    values = list(values)
    return {"count": len(values), **stats(values)}


def validate(data, freezes):
    require(data["schema_version"] == 1 and data["benchmark"] == "released_real_data"
            and data["complete"] is True, "Unexpected or incomplete raw result")
    params, environment, dataset = data["parameters"], data["metadata"], data["dataset"]
    require(params["previous_version"] == PROFILE["previous"]
            and params["current_version"] == PROFILE["current"], "Unexpected release versions")
    for key, value in {
        "seeds": list(SEEDS), "repeats": len(REPEATS), "expected_workers": 27,
        "worker_timeout_seconds": 300, "worker_run_budget_seconds": 1200,
        "sklearn_working_memory_mib": 256,
    }.items():
        require(params[key] == value, f"Unexpected parameter: {key}")
    require(environment["git_commit"] == PROFILE["commit"]
            and environment["github_run_id"] == PROFILE["run"]
            and environment["github_run_attempt"] == "1", "Unexpected benchmark provenance")
    require(environment["faiss_imputer"] == PROFILE["current"], "Unexpected coordinator package")
    check_module(environment, "current")
    shared = freezes["shared-dependencies.txt"]
    for variant in ("previous", "current"):
        require(freezes[f"{variant}-environment.txt"] ==
                {**shared, "faiss-imputer": PROFILE[variant]},
                f"Dependency mismatch: {variant}")
    for key, package in (("numpy", "numpy"), ("scikit_learn", "scikit-learn"),
                         ("faiss", "faiss-cpu")):
        require(environment[key] == shared[package], f"Dependency freeze mismatch: {key}")
    require(dataset["rows"] == 4898 and dataset["features"] == 11
            and dataset["target_used"] is False and dataset["excluded_columns"] == ["quality"],
            "Unexpected dataset metadata")
    names = dataset["feature_names"]
    require(len(names) == 11 and len(set(names)) == 11 and names[-1] == "alcohol",
            "Unexpected dataset feature order")
    expected_keys = set(product(SEEDS, REPEATS, VARIANTS))
    records, plan = data["records"], data["planned_configs"]
    require(len(records) == len(plan) == len(expected_keys), "Incomplete worker grid")
    expected_order = []
    for seed_index, seed in enumerate(SEEDS):
        for repeat in REPEATS:
            offset = (seed_index + repeat - 1) % len(VARIANTS)
            expected_order.extend((seed, repeat, variant)
                                  for variant in VARIANTS[offset:] + VARIANTS[:offset])
    indexed, cases, consistent = {}, {}, {}
    for index, (record, config) in enumerate(zip(records, plan)):
        key = (record["seed"], record["repeat"], record["variant"])
        require(key == expected_order[index] and key not in indexed, "Worker order/key mismatch")
        indexed[key] = record
        seed, repeat, variant = key
        require(record["record_index"] == index and all(record[k] == v for k, v in config.items()),
                f"Plan or index mismatch: {key}")
        require(record["status"] == "ok" and record["checks_passed"] is True,
                f"Failed worker: {key}")
        for field, value in COMMON_CONFIG.items():
            require(record[field] == value, f"Configuration mismatch: {key}/{field}")
        version = PROFILE["previous"] if variant == "previous" else PROFILE["current"]
        method = "KNNImputer" if variant == "knn" else "FaissImputer[available]"
        require(record["expected_version"] == version and record["method"] == method,
                f"Unexpected method or version: {key}")
        interpreter = params["interpreters"][variant].replace("\\", "/")
        folder = "faiss-previous" if variant == "previous" else "faiss-current"
        require(interpreter.endswith(f"/{folder}/bin/python"), "Unexpected interpreter path")
        worker_env = record["environment"]
        require(all(worker_env[k] == environment[k] for k in ENVIRONMENT_FIELDS),
                f"Worker environment mismatch: {key}")
        require(worker_env["faiss_imputer"] == version, f"Installed version mismatch: {key}")
        check_module(worker_env, variant)
        require(record["faiss_omp_threads"] == 1 and record["threadpools"]
                and all(pool["num_threads"] == 1 for pool in record["threadpools"]),
                f"Native thread count mismatch: {key}")
        require(record["sklearn_working_memory_mib"] == 256,
                f"Working-memory setting mismatch: {key}")
        expected_model = {
            "n_neighbors": 5, "weights": "uniform", "copy": True,
            "metric": "nan_euclidean" if variant == "knn" else "l2",
        }
        if variant != "knn":
            expected_model.update(strategy="mean", donor_policy="available", index_factory="Flat")
        require(record["model_parameters"] == expected_model, f"Model parameter mismatch: {key}")
        require(record["dataset"] == dataset, f"Dataset differs: {key}")
        require(record["input_dtype"] == record["output_dtype"] == "float64",
                f"Output/input dtype mismatch: {key}")
        case = record["case"]
        for field, value in {
            "seed": seed, "mechanism": "MCAR", "input_dtype": "float64",
            "truth_dtype": "float64", "n_train": 3000, "n_query": 1000,
            "feature_names": names, "nominal_overall_missing_rate": 0.1,
            "always_observed": ["alcohol"], "mar_reference_rows": None,
            "mar_cutoff": None, "mar_low_probability": None, "mar_high_probability": None,
        }.items():
            require(case[field] == value, f"Prepared case mismatch: {key}/{field}")
        require(case["eligible_base_probability"] == 0.1 * 11 / 10,
                f"Missingness probability mismatch: {key}")
        for field in FINGERPRINTS:
            require(re.fullmatch("[0-9a-f]{64}", case["fingerprints"][field]) is not None,
                    f"Invalid input fingerprint: {key}/{field}")
        require(case["fingerprints"]["dataset"] == dataset["dataset_sha256"],
                "Dataset-array fingerprint differs")
        for mask_name, size in (("train_mask", 3000), ("query_mask", 1000)):
            mask = case[mask_name]
            counts = mask["missing_per_feature"]
            require(len(counts) == 11 and counts[-1] == 0
                    and all(type(value) is int and 0 <= value <= size for value in counts),
                    f"Invalid missingness counts: {key}/{mask_name}")
            require(math.isclose(mask["overall_missing_rate"], sum(counts) / (size * 11),
                                 rel_tol=1e-14, abs_tol=1e-15)
                    and math.isclose(mask["eligible_missing_rate"], sum(counts) / (size * 10),
                                     rel_tol=1e-14, abs_tol=1e-15), "Missingness rate/count mismatch")
            require(0 <= mask["complete_rows"] <= size
                    and mask["complete_rows"] + mask["rows_with_missing"] == size,
                    "Invalid complete-row count")
        require(case["complete_donors"] == case["train_mask"]["complete_rows"],
                "Complete-donor count mismatch")
        require(len(case["scaler_mean"]) == len(case["scaler_scale"]) == 11,
                "Invalid scaler feature count")
        require(all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in case["scaler_mean"]), "Invalid scaler means")
        for value in case["scaler_scale"]:
            finite(value, "scaler scale", positive=True)
        if seed in cases:
            require(case == cases[seed], f"Inputs or case metadata differ: {key}")
        cases[seed] = case
        for field in TIMINGS + ("worker_peak_rss_mib", "worker_wall_seconds"):
            finite(record[field], field, positive=True)
        require(record["total_seconds"] == record["fit_seconds"] + record["transform_seconds"],
                f"Total-time mismatch: {key}")
        require(re.fullmatch("[0-9a-f]{64}", record["output_sha256"]) is not None,
                f"Invalid output hash: {key}")
        scored = sum(case["query_mask"]["missing_per_feature"])
        require(scored == record["scored_cells"] == len(record["imputed_values"])
                and scored > 0, f"Scored-cell mismatch: {key}")
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in record["imputed_values"]),
                f"Invalid imputed values: {key}")
        for field in ("rmse", "mae"):
            finite(record[field], field)
        require(record["quality_scope"] == case["quality_scope"] == "masked held-out query entries"
                and record["quality_units"] == case["quality_units"] ==
                "standardized using observed training values", "Unexpected quality scope/units")
        features = record["feature_quality"]
        require([item["feature"] for item in features] == names, "Feature quality order mismatch")
        for column, item in enumerate(features):
            count = case["query_mask"]["missing_per_feature"][column]
            require(item["scored_cells"] == count, "Feature scoring count mismatch")
            for field in ("rmse_standardized", "mae_standardized", "rmse_original_units", "mae_original_units"):
                if count:
                    finite(item[field], field)
                else:
                    require(item[field] is None, "Unscored feature metric must be null")
        quality = {field: record[field] for field in
                   ("output_sha256", "imputed_values", "rmse", "mae", "feature_quality")}
        qkey = (seed, variant)
        if qkey in consistent:
            require(quality == consistent[qkey], f"Output or quality differs across repeats: {key}")
        consistent[qkey] = quality
    require(set(indexed) == expected_keys, "Missing workers")
    summaries = data["summaries"]
    require(len(summaries) == 3 and {s["variant"] for s in summaries} == set(VARIANTS),
            "Unexpected stored method summaries")
    for summary in summaries:
        variant = summary["variant"]
        group = [indexed[(seed, repeat, variant)] for seed, repeat in product(SEEDS, REPEATS)]
        seed_group = [indexed[(seed, 1, variant)] for seed in SEEDS]
        require(summary["expected_records"] == summary["successful_records"] == 9
                and summary["complete"] is True and summary["quality_seed_count"] == 3,
                "Stored summary completeness mismatch")
        require(summary["timing_and_memory"] == {
            field: counted_stats(row[field] for row in group)
            for field in TIMINGS + ("worker_peak_rss_mib",)
        }, "Stored timing or RSS summary differs from records")
        require(summary["quality"] == {
            field: counted_stats(row[field] for row in seed_group) for field in ("rmse", "mae")
        }, "Stored quality summary differs from seed records")
        require(summary["quality_record_indices"] == [r["record_index"] for r in seed_group],
                "Stored quality sample indices differ")
    return indexed, cases


def analyze(data, freezes):
    indexed, cases = validate(data, freezes)
    cells, comparisons = [], []
    for variant in VARIANTS:
        group = [indexed[(seed, repeat, variant)] for seed, repeat in product(SEEDS, REPEATS)]
        seed_group = [indexed[(seed, 1, variant)] for seed in SEEDS]
        cells.append({
            "variant": variant, "label": PROFILE["labels"][variant],
            "record_count": len(group), "quality_seed_count": len(seed_group),
            "timing": {field: stats(r[field] for r in group) for field in TIMINGS},
            "memory": {"worker_peak_rss_mib": stats(r["worker_peak_rss_mib"] for r in group)},
            "quality": {field: stats(r[field] for r in seed_group) for field in ("rmse", "mae")},
            "samples": [{field: r[field] for field in
                         ("record_index", "seed", "repeat", "worker_peak_rss_mib", "worker_wall_seconds",
                          "rmse", "mae", "output_sha256") + TIMINGS} for r in group],
            "quality_samples": [{field: r[field] for field in
                                 ("record_index", "seed", "scored_cells", "rmse", "mae",
                                  "feature_quality", "output_sha256")} for r in seed_group],
        })
    expected_comparisons = (("previous", "current"), ("knn", "current"), ("knn", "previous"))
    stored_comparisons = data["comparisons"]
    require(len(stored_comparisons) == len(expected_comparisons), "Unexpected comparison count")
    seen = set()
    for stored in stored_comparisons:
        key = (stored["numerator_variant"], stored["denominator_variant"])
        require(key in expected_comparisons and key not in seen, "Duplicate/unexpected comparison")
        seen.add(key)
    for numerator, denominator in expected_comparisons:
        pairs, raw_pairs = [], []
        for seed, repeat in product(SEEDS, REPEATS):
            left, right = indexed[(seed, repeat, numerator)], indexed[(seed, repeat, denominator)]
            require(all(left[field] == right[field] for field in MATCH_FIELDS)
                    and left["case"] == right["case"], "Mismatched pair inputs/configuration")
            difference = max(abs(a - b) for a, b in zip(left["imputed_values"], right["imputed_values"]))
            finite(difference, "hidden-entry difference")
            timing = {field: {
                "numerator_seconds": left[field], "denominator_seconds": right[field],
                "ratio": left[field] / right[field],
                "denominator_duration_change_percent": 100 * (right[field] / left[field] - 1),
            } for field in TIMINGS}
            for field in TIMINGS:
                finite(timing[field]["ratio"], "paired timing ratio", positive=True)
            hashes_match = left["output_sha256"] == right["output_sha256"]
            pairs.append({
                "match": {field: left[field] for field in MATCH_FIELDS},
                "numerator_record_index": left["record_index"],
                "denominator_record_index": right["record_index"],
                "timing": timing,
                "numerator_output_sha256": left["output_sha256"],
                "denominator_output_sha256": right["output_sha256"],
                "full_output_hash_matches": hashes_match,
                "max_scored_abs_difference": difference,
                "absolute_rmse_difference": abs(left["rmse"] - right["rmse"]),
                "absolute_mae_difference": abs(left["mae"] - right["mae"]),
            })
            raw_pairs.append({
                "match": {field: left[field] for field in MATCH_FIELDS},
                "numerator_record_index": left["record_index"],
                "denominator_record_index": right["record_index"],
                "timing_ratios": {field: timing[field]["ratio"] for field in TIMINGS},
                "output_sha256_equal": hashes_match,
                "max_abs_difference_on_hidden_entries": difference,
            })
        stored = next(c for c in stored_comparisons
                      if (c["numerator_variant"], c["denominator_variant"]) == (numerator, denominator))
        require(stored["expected_pairs"] == stored["matched_pairs"] == len(pairs)
                and stored["complete"] is True, "Stored pair count mismatch")
        require(stored["pairs"] == raw_pairs, "Stored individual pair calculations differ")
        require(stored["timing_ratios"] == {
            field: counted_stats(p["timing"][field]["ratio"] for p in pairs) for field in TIMINGS
        }, "Stored paired ratio summaries differ")
        comparisons.append({
            "numerator_variant": numerator, "denominator_variant": denominator,
            "pair_count": len(pairs), "pairs": pairs,
            "speedup": {field: stats(p["timing"][field]["ratio"] for p in pairs) for field in TIMINGS},
            "denominator_duration_change_percent": {
                field: stats(p["timing"][field]["denominator_duration_change_percent"] for p in pairs)
                for field in TIMINGS
            },
            "faster_pairs": {field: sum(p["timing"][field]["ratio"] > 1 for p in pairs) for field in TIMINGS},
            "matching_output_hashes": sum(p["full_output_hash_matches"] for p in pairs),
            "max_scored_abs_difference": max(p["max_scored_abs_difference"] for p in pairs),
            "max_absolute_rmse_difference": max(p["absolute_rmse_difference"] for p in pairs),
            "max_absolute_mae_difference": max(p["absolute_mae_difference"] for p in pairs),
        })
    return {
        "schema_version": 1,
        "source": {
            "archive": PROFILE["archive"], "archive_sha256": PROFILE["archive_sha256"],
            "json_member": MEMBER, "json_sha256": PROFILE["json_sha256"],
            "benchmark_commit": PROFILE["commit"], "github_run_id": PROFILE["run"], "github_run_attempt": "1",
            "dataset_member": DATASET_MEMBER,
            "dataset_archive_sha256": data["dataset"]["source_archive_sha256"],
            "dataset_source_file": SOURCE_MEMBER,
            "dataset_source_sha256": data["dataset"]["source_file_sha256"],
        },
        "configuration": {**COMMON_CONFIG, "seeds": list(SEEDS), "repeats": len(REPEATS),
                          "weights": "uniform", "strategy": "mean", "index_factory": "Flat",
                          "sklearn_working_memory_mib": 256},
        "environment": {field: data["metadata"][field] for field in ENVIRONMENT_FIELDS},
        "dataset": data["dataset"], "dependency_freezes": freezes,
        "aggregation": {
            "timing_and_memory": "Median [min, max] across 3 seeds x 3 repeats = 9 workers per variant.",
            "quality": "Repeat 1 for each of 3 seeds after exact repeat quality/hash/imputed-value checks.",
            "total_seconds": "Consecutive fit plus first held-out transform; no additional transforms.",
            "speedup": "Median [min, max] of 9 matched numerator/denominator timing ratios; not ratio of medians.",
            "matching": list(MATCH_FIELDS),
            "additional_matching": "Same archived run, prepared case/fingerprints, validated model parameters and native thread settings.",
            "duration_change": "Median [min, max] of 100 * (denominator/numerator time - 1) for matched records.",
            "output_agreement": "Recompute hidden-entry differences from saved imputed_values; compare recorded full-output hashes.",
            "quality_evidence": "Aggregate stored worker errors against held-out truth; underlying truth/mask arrays are not regenerated.",
            "memory": "Peak sampled after validation and before worker JSON serialization; includes setup and warmup, not isolated transform memory.",
            "numeric_precision": "Unrounded binary64 calculations from parsed JSON, serialized at round-trip precision; Markdown alone is rounded.",
        },
        "validation": {
            "successful_records": len(indexed), "expected_records": 27,
            "stored_summaries_checked": len(data["summaries"]),
            "stored_comparisons_checked": len(stored_comparisons),
            "identical_dependencies_except_faiss_imputer": True,
            "installed_versions_and_site_packages_paths_checked": True,
            "all_native_thread_counts_one": True,
            "prepared_cases_identical_across_variants_and_repeats": True,
            "quality_hashes_and_imputed_values_stable_across_repeats": True,
        },
        "cases": [{**cases[seed],
                   "scored_cells": sum(cases[seed]["query_mask"]["missing_per_feature"]),
                   "observed_donors_per_feature": [cases[seed]["n_train"] - count
                                                   for count in cases[seed]["train_mask"]["missing_per_feature"]]}
                  for seed in SEEDS],
        "cells": cells, "comparisons": comparisons,
    }


def render_report(result):
    cells = result["cells"]
    comparisons = result["comparisons"]
    cases = result["cases"]
    environment = result["environment"]
    dataset = result["dataset"]
    first_case = cases[0]
    current, previous = PROFILE["current"], PROFILE["previous"]
    labels = PROFILE["labels"]
    release = next(c for c in comparisons if c["numerator_variant"] == "previous")
    versus_knn = next(c for c in comparisons
                      if c["numerator_variant"] == "knn"
                      and c["denominator_variant"] == "current")

    def milliseconds(value):
        return formatted({key: number * 1000 for key, number in value.items()}, 3)

    lines = [
        f"# Wine Quality White: released {current} and {previous}", "",
        "[Benchmark index](README.md) · [Project README](../../README.md#performance)", "",
        "This report compares two installed FaissImputer releases and KNNImputer "
        "on one held-out Wine Quality White workload. It measures consecutive "
        "`fit(train)` and first `transform(query)` calls with available donors, "
        "float64 inputs, and MCAR missingness. It does not pool same-data APIs, "
        "dtypes, donor policies, or runs.", "",
        "## Evidence and configuration", "",
        f"- [Preserved original artifact](../../{PROFILE['archive']}).",
        f"- [Full-precision analysis](../../{PROFILE['summary']}) and "
        "[generator](../../benchmarks/analyze_released_real_data.py).",
        f"- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/{PROFILE['run']}/attempts/1).",
        f"- Benchmark source commit: `{PROFILE['commit']}`. Package measurements "
        "use installed distributions outside the checkout.",
        f"- Archive SHA-256: `{PROFILE['archive_sha256']}`.",
        f"- `{MEMBER}` SHA-256: `{PROFILE['json_sha256']}`.",
        f"- Runner: {environment['cpu_model']}; {environment['logical_cpus']} logical "
        f"CPUs, {environment['affinity_cpus']} in affinity; all recorded native thread counts are one.",
        f"- Python {environment['python']}; NumPy {environment['numpy']}; "
        f"scikit-learn {environment['scikit_learn']}; Faiss {environment['faiss']}.",
        f"- {first_case['n_train']:,} training rows; {first_case['n_query']:,} held-out "
        f"query rows; {len(first_case['feature_names'])} numerical features; "
        f"k={result['configuration']['n_neighbors']}; uniform weights. Faiss uses "
        'built-in L2, mean aggregation, `donor_policy="available"`, and `index_factory="Flat"`.',
        f"- Nominal overall missingness is {100 * first_case['nominal_overall_missing_rate']:.0f}% "
        f"in training and query inputs. `{first_case['always_observed'][0]}` remains observed; "
        "missingness is sampled on the other features. Actual rates are reported below.",
        f"- Seeds: {', '.join(str(case['seed']) for case in cases)}; "
        f"{len(REPEATS)} fresh workers per seed and variant; "
        f"{result['validation']['successful_records']} successful workers. "
        "Variant order rotates across repetitions.", "",
        "The archived dependency freezes differ only in the FaissImputer release. "
        f"KNNImputer uses the {current} environment. Inputs, split identities, masks, "
        "scaler metadata and fingerprints agree across all variants and repeats "
        "for each seed. Standardization uses observed training values only; "
        "held-out ground truth retains float64 precision.", "",
        "## Aggregation methodology", "",
        f"Timing and memory use **{len(SEEDS) * len(REPEATS)} records = "
        f"{len(SEEDS)} seeds × {len(REPEATS)} repeats** per method. Each table "
        "entry is median [min–max]. `transform_seconds` measures the first held-out "
        "transform. `total_seconds` measures fit plus that transform. Fit and "
        "transform are consecutive, with no explicit garbage collection or RSS "
        "sampling between them. Preparation, warmup, validation, process startup "
        "and serialization are outside the measured interval. No additional "
        "transform calls are measured in this experiment.", "",
        "Speedup is **numerator record time / denominator record time**, calculated "
        "for each matching pair, then summarized as median [min–max]. Each comparison "
        f"contains **{len(SEEDS) * len(REPEATS)} pairs** with the same archived run, "
        "dataset, training/query sizes, features, neighbors, donor regime, metric/weights "
        "configuration, missingness configuration, dtype, API, thread settings, seed, "
        "repeat, and prepared inputs. The ratios are **not obtained by dividing "
        "method-level median times**. Values above one favor the denominator.", "",
        "Duration change is computed for each pair as "
        "`100 * (denominator time / numerator time - 1)` and then summarized. "
        "Negative values mean the denominator took less time. Observed ranges are "
        "not confidence intervals. Timing repetitions reuse each seed dataset; "
        "they do not create additional independent datasets.", "",
        f"RMSE and MAE use **{len(SEEDS)} seed metrics**, taking repeat 1 after "
        "verifying repeat consistency of metrics, stored imputed values and output "
        "hashes. They measure error on masked held-out entries against ground truth, "
        "in units standardized using the training data. They do not measure "
        "method-to-method differences.", "",
        "The analyzer uses unrounded JSON numbers for all calculations. The summary "
        "preserves full-precision timing samples, pair indices, numerators, denominators, "
        "ratios, duration changes, quality samples and input/output fingerprints. "
        "Only this Markdown presentation is rounded. Stored worker RMSE/MAE values "
        "are aggregated; ground-truth and mask arrays are not stored as arrays in "
        "the result JSON, so those underlying errors are not recomputed by this analyzer. "
        "Pairwise hidden-entry output differences are recomputed directly from "
        "the archived `imputed_values` arrays.", "",
        "## Timing", "",
        f"Milliseconds, median [min–max] across {len(SEEDS) * len(REPEATS)} workers per method. "
        "The full-precision summary keeps the original seconds.", "",
        table(
            ["Method", "Fit (ms)", "First transform (ms)", "Fit + first transform (ms)"],
            [[cell["label"], *(milliseconds(cell["timing"][field]) for field in TIMINGS)]
             for cell in cells],
        ), "", "## Matched timing ratios", "",
        f"Median [min–max] of {len(SEEDS) * len(REPEATS)} numerator/denominator ratios "
        "per comparison. Above one favors the denominator; below one favors the numerator.", "",
        table(
            ["Numerator / denominator", "Fit ratio", "First-transform ratio", "Total ratio"],
            [[f"{labels[c['numerator_variant']]} / {labels[c['denominator_variant']]}",
              *(formatted(c["speedup"][field], 4) for field in TIMINGS)]
             for c in comparisons],
        ), "", f"### {current} duration change relative to {previous}", "",
        "Median [min–max] of matched per-pair percentage changes. Negative values mean less time.", "",
        table(
            ["Phase", "Duration change (%)", f"Pairs favoring {current}"],
            [[label, formatted(release["denominator_duration_change_percent"][field], 4),
              f"{release['faster_pairs'][field]}/{release['pair_count']}"]
             for field, label in zip(TIMINGS, ("Fit", "First transform", "Fit + first transform"))],
        ), "", "## Memory", "",
        f"MiB, median [min–max] across {len(SEEDS) * len(REPEATS)} workers per method. "
        "Peak RSS is sampled after validation and before worker JSON serialization. "
        "It includes imports, input preparation, warmup, fit, transform and validation. "
        "It is neither isolated transform memory nor retained model size. "
        "The sklearn working-memory setting is not a process RAM limit.", "",
        table(
            ["Method", "Worker peak RSS (MiB)"],
            [[cell["label"], formatted(cell["memory"]["worker_peak_rss_mib"], 3)]
             for cell in cells],
        ), "", "## Reconstruction quality", "",
        f"Median [min–max] across {len(SEEDS)} seeds, using each seed's masked "
        "held-out query entries. Scored-cell counts are listed below. Similar "
        "aggregate errors do not establish equality of individual imputed values "
        "or algorithmic equivalence.", "",
        table(
            ["Method", "RMSE", "MAE"],
            [[cell["label"], formatted(cell["quality"]["rmse"], 10),
              formatted(cell["quality"]["mae"], 10)] for cell in cells],
        ), "", "### Output agreement", "",
        f"Hash counts cover {len(SEEDS) * len(REPEATS)} matched timing pairs but only "
        f"{len(SEEDS)} distinct seed inputs. Hashes are those recorded for the full "
        "output arrays. Hidden-entry differences are recomputed from stored imputed "
        "values. All differences below are maxima across matching pairs.", "",
        table(
            ["Comparison", "Matching full-output hashes", "Max hidden-entry difference", "Max RMSE difference", "Max MAE difference"],
            [[f"{labels[c['denominator_variant']]} vs {labels[c['numerator_variant']]}",
              f"{c['matching_output_hashes']}/{c['pair_count']}",
              f"{c['max_scored_abs_difference']:.12g}",
              f"{c['max_absolute_rmse_difference']:.12g}",
              f"{c['max_absolute_mae_difference']:.12g}"]
             for c in comparisons if c["denominator_variant"] == "current"],
        ), "", "## Donor counts and observed missingness", "",
        "One row per seed, after input and case metadata checks across variants and "
        "repeats. Complete donors are fully observed training rows; available mode "
        "also uses partially observed rows, separately for each missing feature.", "",
        table(
            ["Seed", "Complete donors", "Training missing (%)", "Query missing (%)", "Queries with missing values", "Scored cells"],
            [[case["seed"], case["complete_donors"],
              f"{100 * case['train_mask']['overall_missing_rate']:.4f}",
              f"{100 * case['query_mask']['overall_missing_rate']:.4f}",
              case["query_mask"]["rows_with_missing"], case["scored_cells"]]
             for case in cases],
        ), "", "### Training rows observed for each feature", "",
        "Counts are training rows minus missing training entries in each feature. "
        "They describe observed donor values, not a guarantee of a defined distance "
        "to every query. Feature-level reconstruction errors in standardized and "
        "source units are preserved in the full-precision summary.", "",
        table(
            ["Feature", *(f"Seed {case['seed']}" for case in cases)],
            [[name, *(case["observed_donors_per_feature"][column] for case in cases)]
             for column, name in enumerate(first_case["feature_names"])],
        ), "", "## Interpretation and limits", "",
        f"- {current} first-transform speedup relative to {previous} is "
        f"{release['speedup']['transform_seconds']['median']:.5f}×; "
        f"{release['faster_pairs']['transform_seconds']}/{release['pair_count']} "
        "matched first transforms favor the current release. Median paired "
        f"duration change is {release['denominator_duration_change_percent']['transform_seconds']['median']:+.4f}% "
        "for first transform and "
        f"{release['denominator_duration_change_percent']['total_seconds']['median']:+.4f}% "
        "for fit plus first transform.",
        f"- KNN/{current} total-time ratio is "
        f"{versus_knn['speedup']['total_seconds']['median']:.5f}×. "
        "In this workload KNN takes less time in every matched total-time pair; "
        "the version-to-version improvement does not make FaissImputer faster "
        "than KNN here.",
        f"- {previous}/{current} full-output hashes match in "
        f"{release['matching_output_hashes']}/{release['pair_count']} pairs, "
        "with maximum hidden-entry difference "
        f"{release['max_scored_abs_difference']:.12g}. Compared with KNN, "
        f"{current} has {versus_knn['matching_output_hashes']}/{versus_knn['pair_count']} "
        "matching full-output hashes and maximum hidden-entry difference "
        f"{versus_knn['max_scored_abs_difference']:.12g}. These observations "
        "apply to these inputs and versions only.",
        "- Native thread counts, package origins and archived dependency versions "
        "are checked. Timings remain descriptive observations from one runner; "
        "no statistical significance, universal speedup or all-input equivalence is established.",
        "- The [published synthetic comparison](released_versions_0.3.22.md) and "
        "[earlier real-data coverage](real-data-datasets-ef04b1b.md) are separate "
        "experiments. Their hardware, versions or workloads differ. Cross-run "
        "absolute timing changes do not isolate a release effect.", "",
        "## Dataset provenance", "",
        f"[{dataset['dataset']}]({dataset['source']}). {dataset['citation']}", "",
        f"The preserved artifact includes the original source ZIP and identifies "
        f"the dataset license as {dataset['license']}. `{dataset['source_file']}` "
        f"contains {dataset['rows']:,} rows; `{', '.join(dataset['excluded_columns'])}` "
        "is excluded from the predictor matrix. No target values are used for "
        "neighbor search or downstream prediction scoring.", "",
        f"- Source ZIP SHA-256: `{dataset['source_archive_sha256']}`.",
        f"- Source CSV SHA-256: `{dataset['source_file_sha256']}`.",
        f"- Parsed predictor-array fingerprint: `{dataset['dataset_sha256']}`.", "",
        "## Reproduction", "",
        "The standard-library-only generator reads the preserved artifact and "
        "reuses validation and formatting helpers from the published synthetic "
        "analysis. It neither installs nor executes imputers. It checks archive "
        "and member hashes, the complete worker grid, input consistency, dependency "
        "freezes, repeated outputs, and every stored summary and comparison before "
        "writing this report and the full-precision JSON.", "",
        "The [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml) "
        "regenerates both historical synthetic reports and this Wine report. "
        "On the first push to `bench/released-wine-quality-0.3.22`, it may generate "
        "the Wine outputs when both committed files are absent. Commit the Markdown "
        "and summary JSON together; subsequent runs compare them byte-for-byte. "
        "Pull requests require both Wine outputs and both synthetic output pairs. "
        "A partially present pair fails. The workflow runs analysis of saved data, "
        "not a new benchmark.", "",
    ]
    return "\n".join(lines)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=(PROFILE["current"],), default=PROFILE["current"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    input_path = args.input if args.input is not None else ROOT / PROFILE["archive"]
    report_path = args.report if args.report is not None else ROOT / PROFILE["report"]
    summary_path = args.summary if args.summary is not None else ROOT / PROFILE["summary"]
    outputs = (report_path.resolve(), summary_path.resolve())
    protected = {
        input_path.resolve(), (ROOT / PROFILE["archive"]).resolve(),
        (ROOT / "benchmarks/results/released_versions_0.3.21.zip").resolve(),
        (ROOT / "benchmarks/results/released_versions_0.3.22.zip").resolve(),
    }
    require(outputs[0] != outputs[1], "Report and summary need different paths")
    require(not protected.intersection(outputs), "Output would overwrite a raw archive")
    data, freezes = read_archive(input_path)
    result = analyze(data, freezes)
    report = render_report(result)
    summary = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for path, text in ((report_path, report), (summary_path, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(
        f"Validated Wine Quality {args.release}: "
        f"{result['validation']['successful_records']} workers; "
        "generated report and full-precision JSON."
    )


if __name__ == "__main__":
    main()
