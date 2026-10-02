"""Regenerate archived published-release comparisons without running imputers."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import math
from pathlib import Path
import re
from statistics import median
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
MEMBER = "version_comparison_q300.json"
RELEASES = {
    "0.3.21": {
        "previous": "0.3.20",
        "archive_sha256": "53ef359c45525b2fff05ea7a3cb9be685ba06c18ac742d68ba8882ad1377e229",
        "json_sha256": "192ab6345645649e9a3587a50f82bd7a3359ab1de1e158414911172639354124",
        "commit": "693ce8ee0f9a44cdfe7aaab5d1af1233701eb0b1",
        "run": "36645461314",
    },
    "0.3.22": {
        "previous": "0.3.21",
        "archive_sha256": "7a633e03bd86110380fae69d86713c420c1bfc3c81ccd61b95e56752c943532d",
        "json_sha256": "f83c041b08242c64819707ccaf3052323d6404f0b465b9e9d8961577eb902bf5",
        "commit": "4b5b9cbdc6d629ffa26fb2b6b47a24c268254b81",
        "run": "36896955077",
    },
}
POLICIES = ("complete", "available")
DTYPES = ("float32", "float64")
VARIANTS = ("knn", "previous", "current")
SEEDS = (101, 202, 303)
REPEATS = (1, 2, 3)
TIMINGS = (
    "fit_seconds", "transform_seconds", "total_seconds",
    "repeated_transform_median_seconds",
)
MEMORY = (
    "worker_peak_rss_mib", "rss_before_fit_mib", "rss_after_fit_mib",
    "fit_rss_change_mib",
)
QUALITY = (
    "rmse", "mae", "max_abs_difference_from_knn",
    "max_abs_difference_from_previous_release",
    "max_abs_difference_from_first_repeat",
)
CASE_FIELDS = (
    "complete_donors", "train_missing_rate", "query_missing_rate",
    "query_patterns", "scored_cells", "fingerprints",
)
SUMMARY_FIELDS = TIMINGS + (
    "worker_peak_rss_mib", "fit_rss_change_mib",
) + QUALITY
ENVIRONMENT_FIELDS = (
    "git_commit", "github_run_id", "github_run_attempt", "python",
    "platform", "cpu_model", "logical_cpus", "affinity_cpus", "numpy",
    "scikit_learn", "faiss",
)
COMMON_CONFIG = {
    "train_size": 20000, "queries": 300, "features": 20, "n_neighbors": 5,
    "metric": "l2 / nan_euclidean", "weights": "uniform",
    "training_policies": list(POLICIES),
    "training_missing_rates": {"complete": 0.0, "available": 0.1},
    "dtypes": list(DTYPES), "query_pattern": "random", "threads": 1,
    "seeds": list(SEEDS), "repeats": 3, "repeated_transforms": 2,
    "sklearn_working_memory_mib": 256, "expected_workers": 108,
}


def release_profile(version):
    evidence = RELEASES[version]
    return {
        **evidence,
        "current": version,
        "archive": f"benchmarks/results/released_versions_{version}.zip",
        "report": f"docs/benchmarks/released_versions_{version}.md",
        "summary": f"benchmarks/results/released_versions_{version}-summary.json",
        "versions": {"knn": version, "previous": evidence["previous"], "current": version},
        "labels": {
            "knn": "KNNImputer",
            "previous": f"FaissImputer {evidence['previous']}",
            "current": f"FaissImputer {version}",
        },
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = list(values)
    require(bool(values), "Cannot summarize an empty sample")
    return {"median": median(values), "min": min(values), "max": max(values)}


def finite(value, name, positive=False):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and (value > 0 if positive else value >= 0),
        f"Invalid numeric value for {name}: {value!r}",
    )


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_freeze(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("==")
        require(len(parts) == 2 and all(parts), f"Unexpected freeze line: {line}")
        name, version = parts
        require(name not in result, f"Duplicate dependency: {name}")
        result[name] = version
    return result


def read_archive(path, profile):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == profile["archive_sha256"], "Archive SHA-256 mismatch")
    with ZipFile(BytesIO(raw)) as archive:
        expected = {
            MEMBER, "current-environment.txt", "previous-environment.txt",
            "shared-dependencies.txt",
        }
        names = archive.namelist()
        require(len(names) == len(expected) and set(names) == expected,
                "Unexpected or duplicate archive members")
        raw_json = archive.read(MEMBER)
        require(sha256(raw_json).hexdigest() == profile["json_sha256"],
                "Raw JSON SHA-256 mismatch")
        freezes = {
            name: parse_freeze(archive.read(name).decode("utf-8"))
            for name in sorted(expected - {MEMBER})
        }
    data = json.loads(raw_json, parse_constant=reject_constant,
                      object_pairs_hook=unique_object)
    return data, freezes


def record_key(record):
    return tuple(record[field] for field in
                 ("training_policy", "dtype", "variant", "seed", "repeat"))


def check_module(environment, variant):
    location = environment["faiss_imputer_module"].replace("\\", "/")
    folder = "faiss-previous" if variant == "previous" else "faiss-current"
    require(location.endswith(
        f"/{folder}/lib/python3.12/site-packages/faiss_imputer/__init__.py"
    ), f"Unexpected installed module path for {variant}")


def validate(data, freezes, profile):
    versions, labels = profile["versions"], profile["labels"]
    require(data["schema_version"] == 1, "Unexpected raw schema")
    params, metadata, records = data["parameters"], data["metadata"], data["records"]
    for key, value in COMMON_CONFIG.items():
        require(params[key] == value, f"Unexpected parameter: {key}")
    require(params["labels"] == labels, "Unexpected method labels")
    require(params["faiss_imputer_environment_versions"] == versions,
            "Unexpected package versions")
    require(metadata["git_commit"] == profile["commit"] and metadata["github_run_id"] == profile["run"]
            and metadata["github_run_attempt"] == "1", "Unexpected provenance")
    require(metadata["faiss_imputer"] == profile["current"], "Unexpected coordinator package")
    check_module(metadata, "current")
    shared = freezes["shared-dependencies.txt"]
    for variant in ("previous", "current"):
        require(freezes[f"{variant}-environment.txt"] ==
                {**shared, "faiss-imputer": versions[variant]},
                f"Dependency mismatch in {variant} environment")
    for field, package in (("numpy", "numpy"), ("faiss", "faiss-cpu"),
                           ("scikit_learn", "scikit-learn")):
        require(metadata[field] == shared[package], f"Freeze mismatch: {field}")
    expected_keys = set(product(POLICIES, DTYPES, VARIANTS, SEEDS, REPEATS))
    require(len(records) == len(expected_keys), "Incomplete worker grid")
    indexed, cases, repeat_quality = {}, {}, {}
    for record in records:
        key = record_key(record)
        require(key in expected_keys and key not in indexed, f"Unexpected/duplicate key: {key}")
        indexed[key] = record
        policy, dtype, variant, seed, repeat = key
        require(record["status"] == "ok" and record["checks_passed"] is True,
                f"Failed worker: {key}")
        method = "KNNImputer" if variant == "knn" else f"FaissImputer[{policy}]"
        require(record["method"] == method and record["expected_version"] == versions[variant],
                f"Incorrect method/version: {key}")
        expected_record = {
            "size": 20000, "queries": 300, "pattern": "random", "threads": 1,
            "prefix_sizes": [20000], "repeated_transforms": 2,
            "input_dtype": dtype, "output_dtype": dtype, "scored_cells": 1200,
            "query_missing_rate": 0.2, "faiss_omp_threads": 1,
        }
        for field, value in expected_record.items():
            require(record[field] == value, f"Incorrect {field}: {key}")
        env = record["environment"]
        require(all(env[field] == metadata[field] for field in ENVIRONMENT_FIELDS),
                f"Worker environment mismatch: {key}")
        require(env["faiss_imputer"] == versions[variant], f"Installed version mismatch: {key}")
        check_module(env, variant)
        require(record["threadpools"] and all(pool["num_threads"] == 1
                for pool in record["threadpools"]), f"Native thread count mismatch: {key}")
        for field in TIMINGS + MEMORY + ("worker_wall_seconds",):
            finite(record[field], field, positive=field != "fit_rss_change_mib")
        for field in QUALITY:
            finite(record[field], field)
        require(record["total_seconds"] == record["fit_seconds"] + record["transform_seconds"],
                f"Total timing mismatch: {key}")
        require(record["fit_rss_change_mib"] == record["rss_after_fit_mib"] - record["rss_before_fit_mib"],
                f"Post-fit RSS delta mismatch: {key}")
        repeated = record["repeated_transform_seconds"]
        require(len(repeated) == 2, f"Repeated transform count mismatch: {key}")
        for value in repeated:
            finite(value, "repeated_transform_seconds", positive=True)
        require(median(repeated) == record["repeated_transform_median_seconds"],
                f"Repeated transform median mismatch: {key}")
        require(record["max_abs_difference_from_first_repeat"] == 0.0,
                f"Output changed across repeats: {key}")
        require(re.fullmatch("[0-9a-f]{64}", record["output_sha256"]) is not None,
                f"Invalid output hash: {key}")
        fp = record["fingerprints"]
        require(fp["prefixes"] == {"20000": fp["train"]}, f"Prefix fingerprint mismatch: {key}")
        require(all(re.fullmatch("[0-9a-f]{64}", fp[field]) for field in ("train", "query", "truth")),
                f"Invalid input fingerprints: {key}")
        observed = {field: record[field] for field in CASE_FIELDS}
        case_key = (policy, dtype, seed)
        if case_key in cases:
            require(observed == cases[case_key], f"Case inputs differ: {key}")
        cases[case_key] = observed
        if policy == "complete":
            require(record["complete_donors"] == 20000 and record["train_missing_rate"] == 0.0,
                    f"Complete policy input mismatch: {key}")
        else:
            require(0 < record["complete_donors"] < 20000
                    and 0 < record["train_missing_rate"] < 1, f"Available policy input mismatch: {key}")
        consistent = {field: record[field] for field in QUALITY + ("output_sha256",)}
        qkey = (policy, dtype, variant, seed)
        if qkey in repeat_quality:
            require(consistent == repeat_quality[qkey], f"Quality/hash varies across repeats: {key}")
        repeat_quality[qkey] = consistent
    require(set(indexed) == expected_keys, "Missing workers")
    summaries = data["summaries"]
    require(len(summaries) == 12, "Unexpected stored summary count")
    seen = set()
    for summary in summaries:
        key = tuple(summary[field] for field in ("training_policy", "dtype", "variant"))
        require(key in set(product(POLICIES, DTYPES, VARIANTS)) and key not in seen,
                f"Unexpected/duplicate stored summary: {key}")
        seen.add(key)
        group = [indexed[key + (seed, repeat)] for seed, repeat in product(SEEDS, REPEATS)]
        require(summary["planned_workers"] == summary["successful_workers"] == 9,
                f"Stored summary count mismatch: {key}")
        require(summary["label"] == labels[key[2]], f"Stored summary label mismatch: {key}")
        for field in SUMMARY_FIELDS:
            require(summary[field] == stats(record[field] for record in group),
                    f"Stored summary disagrees with raw records: {key}/{field}")
    return indexed, cases


def analyze(data, freezes, profile):
    indexed, cases = validate(data, freezes, profile)
    cells, comparisons, case_rows = [], [], []
    for policy, dtype, variant in product(POLICIES, DTYPES, VARIANTS):
        group = [indexed[(policy, dtype, variant, seed, repeat)]
                 for seed, repeat in product(SEEDS, REPEATS)]
        fields = ("seed", "repeat") + TIMINGS + MEMORY + QUALITY + (
            "repeated_transform_seconds", "worker_wall_seconds", "output_sha256",
        )
        quality_samples = [{field: record[field] for field in
                            ("seed", "scored_cells", "output_sha256") + QUALITY}
                           for record in group if record["repeat"] == 1]
        cells.append({
            "training_policy": policy, "dtype": dtype, "variant": variant,
            "label": profile["labels"][variant], "record_count": len(group),
            "quality_seed_count": len(quality_samples),
            "timing": {field: stats(record[field] for record in group) for field in TIMINGS},
            "memory": {field: stats(record[field] for record in group) for field in MEMORY},
            "quality": {field: stats(record[field] for record in quality_samples) for field in QUALITY},
            "samples": [{field: record[field] for field in fields} for record in group],
            "quality_samples": quality_samples,
        })
    for policy, dtype in product(POLICIES, DTYPES):
        for seed in SEEDS:
            case_rows.append({"training_policy": policy, "dtype": dtype, "seed": seed,
                              **cases[(policy, dtype, seed)]})
        for numerator, denominator in (("knn", "previous"), ("knn", "current"),
                                       ("previous", "current")):
            pairs = []
            for seed, repeat in product(SEEDS, REPEATS):
                left = indexed[(policy, dtype, numerator, seed, repeat)]
                right = indexed[(policy, dtype, denominator, seed, repeat)]
                # All remaining configuration fields are validated against the
                # single common configuration; input fingerprints match as well.
                pairs.append({
                    "seed": seed, "repeat": repeat,
                    "timing": {field: {
                        "numerator_seconds": left[field],
                        "denominator_seconds": right[field],
                        "ratio": left[field] / right[field],
                        "denominator_duration_change_percent":
                            100 * (right[field] / left[field] - 1),
                    } for field in TIMINGS},
                    "numerator_output_sha256": left["output_sha256"],
                    "denominator_output_sha256": right["output_sha256"],
                    "full_output_hash_matches": left["output_sha256"] == right["output_sha256"],
                    "worker_max_scored_abs_difference": right[
                        "max_abs_difference_from_knn" if numerator == "knn"
                        else "max_abs_difference_from_previous_release"
                    ],
                    "absolute_rmse_difference": abs(left["rmse"] - right["rmse"]),
                    "absolute_mae_difference": abs(left["mae"] - right["mae"]),
                })
            comparisons.append({
                "training_policy": policy, "dtype": dtype,
                "numerator_variant": numerator, "denominator_variant": denominator,
                "pair_count": len(pairs), "pairs": pairs,
                "speedup": {field: stats(pair["timing"][field]["ratio"] for pair in pairs)
                            for field in TIMINGS},
                "denominator_duration_change_percent": {
                    field: stats(pair["timing"][field]["denominator_duration_change_percent"]
                                 for pair in pairs) for field in TIMINGS
                },
                "matching_output_hashes": sum(pair["full_output_hash_matches"] for pair in pairs),
                "max_worker_scored_abs_difference": max(pair["worker_max_scored_abs_difference"] for pair in pairs),
                "max_absolute_rmse_difference": max(pair["absolute_rmse_difference"] for pair in pairs),
                "max_absolute_mae_difference": max(pair["absolute_mae_difference"] for pair in pairs),
            })
    return {
        "schema_version": 1,
        "source": {
            "archive": profile["archive"], "archive_sha256": profile["archive_sha256"],
            "json_member": MEMBER, "json_sha256": profile["json_sha256"],
            "benchmark_commit": profile["commit"], "github_run_id": profile["run"], "github_run_attempt": "1",
        },
        "configuration": COMMON_CONFIG,
        "environment": {field: data["metadata"][field] for field in ENVIRONMENT_FIELDS},
        "dependency_freezes": freezes,
        "aggregation": {
            "timing_and_memory": "Median [min, max] over 3 seeds x 3 repeats = 9 workers per cell.",
            "quality": "Median [min, max] over 3 seeds, using repeat 1 after exact repeat metric/hash checks.",
            "total_seconds": "fit_seconds + first transform_seconds; additional transforms are excluded.",
            "repeated_transform": "Median of 2 additional transform calls per worker, then median [min, max] across 9 workers.",
            "speedup": "Median [min, max] of 9 matched numerator_seconds / denominator_seconds ratios; not ratio of medians.",
            "matching": "Same archived run, training policy, train/query sizes, features, neighbors, metric/weights configuration, target training missingness, query pattern, dtype, threads, repeated-transform count, seed, repeat, and input fingerprints.",
            "duration_change": "Median [min, max] of 100 * (denominator_seconds / numerator_seconds - 1) over matched workers.",
            "quality_evidence": "Aggregated archived worker RMSE/MAE and scored-cell differences. Prediction/truth arrays are not included in this ZIP.",
            "numeric_precision": "No rounding before JSON serialization. Values are Python binary64 calculations from parsed JSON numbers, serialized with round-trip precision.",
        },
        "validation": {
            "successful_records": len(indexed), "expected_records": 108,
            "method_cells": len(cells), "stored_summaries_checked": len(data["summaries"]),
            "identical_dependencies_except_faiss_imputer": True,
            "installed_versions_and_site_packages_paths_checked": True,
            "all_reported_native_thread_counts_one": True,
            "input_fingerprints_match_within_cases": True,
            "quality_and_output_hashes_stable_across_repeats": True,
        },
        "cases": case_rows, "cells": cells, "comparisons": comparisons,
    }


def formatted(value, digits=6):
    return f"{value['median']:.{digits}f} [{value['min']:.{digits}f}\u2013{value['max']:.{digits}f}]"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def release_interpretation(result, profile):
    """Keep the historical prose stable; derive new observations from records."""
    comparisons = result["comparisons"]
    release_pairs = [c for c in comparisons if c["numerator_variant"] == "previous"]
    current_pairs = [c for c in comparisons if c["numerator_variant"] == "knn"
                     and c["denominator_variant"] == "current"]
    release_hash_matches = sum(c["matching_output_hashes"] for c in release_pairs)
    release_pair_count = sum(c["pair_count"] for c in release_pairs)
    available_speedups = [c["speedup"]["transform_seconds"]["median"]
                          for c in current_pairs if c["training_policy"] == "available"]
    if profile["current"] == "0.3.21":
        complete32 = next(c for c in release_pairs if c["training_policy"] == "complete"
                          and c["dtype"] == "float32")
        return [
            f"- 0.3.21 and 0.3.20 have matching full-output hashes in {release_hash_matches}/{release_pair_count} "
            "paired workers, with zero recorded scored-cell differences. This is evidence "
            "for these measured cases, not a claim of equivalence for all inputs.",
            f"- In the available regime, 0.3.21 first-transform paired speedup against KNN "
            f"is {min(available_speedups):.3f}\u2013{max(available_speedups):.3f}\u00d7 across the two dtypes. "
            "Its full-worker peak RSS is higher than the corresponding KNN baseline. "
            "Timing improvements therefore do not imply a memory improvement for this workload.",
            f"- Complete float32 0.3.21 total duration increased by a median "
            f"{complete32['denominator_duration_change_percent']['total_seconds']['median']:.4f}% "
            "in the matched pairs. The small release-to-release differences are reported "
            "as observations from this run; no statistical significance or cause is established.",
            "- Available-mode hashes differ from KNN despite close RMSE/MAE. Complete-mode "
            "hash agreement here occurs with fully observed training data and does not "
            "generalize to the incomplete-training OFAT setup.",
            "- This general synthetic workload does not demonstrate that the 0.3.21 "
            "precision-repair path was triggered. It is not a targeted underflow/overflow test.",
        ]

    available64 = next(c for c in release_pairs if c["training_policy"] == "available"
                       and c["dtype"] == "float64")
    other_speedups = [c["speedup"]["transform_seconds"]["median"]
                      for c in release_pairs if c is not available64]
    faster_first = sum(p["timing"]["transform_seconds"]["ratio"] > 1
                       for p in available64["pairs"])
    previous32 = next(c for c in result["cells"] if c["training_policy"] == "available"
                      and c["dtype"] == "float32" and c["variant"] == "previous")
    largest_fit = max(previous32["samples"], key=lambda sample: sample["fit_seconds"])
    other_fits = [s["fit_seconds"] for s in previous32["samples"] if s is not largest_fit]
    current, previous = profile["current"], profile["previous"]
    max_release_difference = max(c["max_worker_scored_abs_difference"] for c in release_pairs)
    return [
        f"- {current} and {previous} have matching full-output hashes in "
        f"{release_hash_matches}/{release_pair_count} paired workers. The maximum recorded "
        f"scored-cell difference is {max_release_difference:.12g}. This is evidence for "
        "these measured cases, not a claim of equivalence for all inputs.",
        f"- Available float64 first-transform speedup ({previous}/{current}) is "
        f"{available64['speedup']['transform_seconds']['median']:.4f}\u00d7; "
        f"{faster_first}/{available64['pair_count']} matched first transforms favor {current}. "
        "The median paired duration changes are "
        f"{available64['denominator_duration_change_percent']['transform_seconds']['median']:+.4f}% "
        "for first transform and "
        f"{available64['denominator_duration_change_percent']['total_seconds']['median']:+.4f}% "
        "for fit plus first transform; negative changes mean less time. "
        "The additional-transform paired speedup is "
        f"{available64['speedup']['repeated_transform_median_seconds']['median']:.4f}\u00d7.",
        "- The other three policy/dtype cells have median release-to-release "
        f"first-transform ratios of {min(other_speedups):.4f}\u2013{max(other_speedups):.4f}\u00d7. "
        "These are descriptive observations; no statistical significance is established.",
        f"- In the available regime, {current} first-transform paired speedup against KNN "
        f"is {min(available_speedups):.3f}\u2013{max(available_speedups):.3f}\u00d7 across the two dtypes. "
        "Its median full-worker peak RSS is higher than the corresponding KNN baseline "
        "for both dtypes. Peak RSS does not isolate the selected-distance workspace.",
        f"- Retained timing observation: {previous} available float32, "
        f"seed {largest_fit['seed']}, repeat {largest_fit['repeat']}, has fit time "
        f"{largest_fit['fit_seconds']:.6f} s. The other eight fits range from "
        f"{min(other_fits):.6f} to {max(other_fits):.6f} s. This worker contributes "
        "to the timing range and its matched total-time ratio; no samples are excluded.",
        "- Available-mode full-output hashes differ from KNN despite close RMSE/MAE. "
        "The maximum scored-cell differences and metric differences are reported above. "
        "Complete-mode hash agreement here uses fully observed training data and does "
        "not generalize to incomplete-training OFAT cases or other inputs.",
        f"- This comparison uses {result['environment']['cpu_model']}. The "
        "[earlier published 0.3.21 report](released_versions_0.3.21.md) used AMD EPYC 7763. "
        "Cross-run changes in absolute times or KNN-relative speedups do not isolate "
        "a package-version effect. Release comparisons in this report use matched "
        "records from this one Intel run.",
    ]


def render_report(result, profile):
    cells, comparisons = result["cells"], result["comparisons"]
    environment = result["environment"]
    current, previous, labels = profile["current"], profile["previous"], profile["labels"]
    release_pairs = [c for c in comparisons if c["numerator_variant"] == "previous"]
    reproduction = (
        "The analysis uses only the Python standard library and the preserved ZIP; "
        "it does not install or execute either imputer. The analysis workflow regenerates "
        "this report and the full-precision JSON, compares them byte-for-byte with "
        "the committed copies, and uploads the generated outputs. It runs for relevant "
        "pull requests and can be dispatched manually once present on the default branch."
    )
    if current == "0.3.22":
        reproduction = (
            "The analysis uses only the Python standard library and the preserved ZIP; "
            "it does not install or execute either imputer. The generator selects this "
            "archive with `--release 0.3.22`; omitting `--release` retains the historical "
            "0.3.21 output. The Analyze released-version benchmark results workflow "
            "regenerates both releases and uploads their Markdown reports and full-precision "
            "JSON summaries. Committed outputs are compared byte-for-byte. For initial "
            "preservation, a push to `bench/released-0.3.22` or a manual run may generate "
            "the 0.3.22 files when both are absent; "
            "it still verifies the existing 0.3.21 files. Upload both new outputs, then "
            "the pull-request check requires and verifies both releases. A partially "
            "present output pair fails instead of silently skipping comparison."
        )
    lines = [
        f"# Released-package comparison: {current} and {previous}",
        "",
        "[Benchmark index](README.md) \u00b7 [Project README](../../README.md#performance)",
        "",
        "This report compares installed PyPI releases with KNNImputer on one runner, "
        "using held-out queries. It is a separate-query `fit` followed by `transform` "
        "benchmark, not the same-data OFAT benchmark.",
        "",
        "## Evidence and configuration",
        "",
        f"- [Preserved original artifact](../../{profile['archive']}) (uploaded without modification).",
        f"- [Full-precision analysis](../../{profile['summary']}) and [generator](../../benchmarks/analyze_released_versions.py).",
        f"- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/{profile['run']}/attempts/1).",
        f"- Benchmark source commit: `{profile['commit']}`; this identifies the benchmark runner, not an editable package installation.",
        f"- Archive SHA-256: `{profile['archive_sha256']}`.",
        f"- `{MEMBER}` SHA-256: `{profile['json_sha256']}`.",
        f"- Runner: {environment['cpu_model']}; {environment['logical_cpus']} logical CPUs, "
        f"{environment['affinity_cpus']} CPUs in affinity; requested and recorded native thread counts are one.",
        f"- Python {environment['python']}; NumPy {environment['numpy']}; "
        f"scikit-learn {environment['scikit_learn']}; Faiss {environment['faiss']}.",
        "- 20,000 training rows, 300 held-out queries, 20 features, k=5, uniform weights; "
        'FaissImputer is configured with built-in L2, mean aggregation, and `index_factory="Flat"`. Each query has four '
        "missing features (20%), giving 1,200 scored cells per seed.",
        "- Three seeds (101, 202, 303), three fresh sequential workers per seed and variant. "
        "Method order rotates across repeats. There are 108 successful workers and no failed checks.",
        "- `complete` uses fully observed training data. `available` uses training data "
        "with 10% target random missingness. Each policy's KNN baseline receives the same "
        "inputs as its Faiss variants. These are different input regimes, so cross-policy "
        "speed and quality differences do not isolate donor policy.",
        "",
        "The two environments have identical archived dependency freezes except for "
        f"faiss-imputer ({previous} versus {current}). KNNImputer runs in the {current} environment. "
        "The analyzer checks package versions, installed site-packages locations, common "
        "environment metadata, full worker-grid coverage, input fingerprints, and all "
        "stored summary statistics against raw records.",
        "",
        "## Aggregation methodology",
        "",
        "Timing and RSS cells report median [min\u2013max] over **9 workers = 3 seeds \u00d7 "
        "3 repeats**, separately for policy, dtype, and variant. `total_seconds` is "
        "fit plus the first transform. The two additional transform calls are excluded "
        "from that total: their per-worker median is summarized across the nine workers. "
        "First and subsequent transforms remain separate.",
        "",
        "Speedups are calculated as numerator time / denominator time for each matched "
        "worker pair, followed by median [min\u2013max] of the **9 paired ratios**. Matching "
        "requires the same run, policy, train/query sizes, feature and neighbor counts, "
        "metric/weights configuration, target training missingness, query pattern, dtype, "
        "thread count, additional-transform count, seed, repeat, and input fingerprints. "
        "The validated run-level configuration supplies fields not repeated in each record. "
        "The reported speedups are **not ratios of method-level median times**. Ratios "
        "above one favor the denominator method.",
        "",
        "Quality cells report median [min\u2013max] of **3 seed metrics**, after verifying "
        "that metrics and output hashes agree exactly across the three timing repeats. "
        "Repeat 1 supplies each seed metric. RMSE/MAE measure reconstruction error against "
        "hidden synthetic ground truth; method-to-method differences measure output agreement. "
        "This archive contains worker-computed metrics and hashes, not prediction/truth arrays: "
        "the analysis recomputes their summaries, not the underlying arraywise errors.",
        "",
        "All calculations use unrounded parsed JSON values. The machine-readable summary "
        "preserves raw timing/quality samples, per-pair numerators, denominators, ratios, "
        "input/output hashes, and aggregate binary64 values at round-trip precision. "
        "Only the Markdown presentation is rounded. Ranges are observed minima/maxima, "
        "not confidence intervals; repeated timings of the same seed are not independent datasets.",
        "",
        "## Timing",
        "",
        "Seconds, median [min\u2013max] across nine workers per row. Additional-transform "
        "values are medians of the two extra calls within each worker before aggregation.",
    ]
    for policy in POLICIES:
        lines += ["", f"### {policy.capitalize()} training regime", "", table(
            ["dtype", "Method", "Fit (s)", "First transform (s)", "Fit + first transform (s)", "Additional transform (s)"],
            [[c["dtype"], c["label"], *(formatted(c["timing"][f]) for f in TIMINGS)]
             for c in cells if c["training_policy"] == policy],
        )]
    lines += ["", "## Matched speedups", "",
              "Each entry is median [min\u2013max] of nine paired ratios. KNN/Faiss uses the "
              "KNN baseline for the same training regime. First and additional transforms remain separate.", "", table(
        ["Regime", "dtype", "Denominator", "KNN/Faiss fit", "KNN/Faiss first transform", "KNN/Faiss total", "KNN/Faiss additional transform"],
        [[c["training_policy"], c["dtype"], labels[c["denominator_variant"]],
          *(formatted(c["speedup"][f], 3) for f in TIMINGS)]
         for c in comparisons if c["numerator_variant"] == "knn"],
    ), "", "### Release-to-release comparison", "",
              f"Speedups use {previous} time / {current} time. Total-duration change is calculated "
              f"per pair as `100 * ({current} total / {previous} total - 1)` and then summarized; "
              f"positive percentages mean {current} took longer.", "", table(
        ["Regime", "dtype", "Fit ratio", "First-transform ratio", "Total ratio", "Additional-transform ratio", "Total-duration change (%)"],
        [[c["training_policy"], c["dtype"], *(formatted(c["speedup"][f], 4) for f in TIMINGS),
          formatted(c["denominator_duration_change_percent"]["total_seconds"], 4)]
         for c in release_pairs],
    ), "", "## Memory", "",
              "MiB, median [min\u2013max] over nine workers. Peak RSS covers the full worker "
              "lifetime, including warmup, fit, all three transform calls, and validation; "
              "it is not isolated first-transform memory. Post-fit RSS change is after-fit "
              "RSS minus before-fit RSS and includes allocator effects; it is not exact fitted-model size.", "", table(
        ["Regime", "dtype", "Method", "Full-worker peak RSS (MiB)", "Post-fit RSS change (MiB)"],
        [[c["training_policy"], c["dtype"], c["label"],
          formatted(c["memory"]["worker_peak_rss_mib"], 3),
          formatted(c["memory"]["fit_rss_change_mib"], 3)] for c in cells],
    ), "", "## Reconstruction quality and output agreement", "",
              "RMSE and MAE: median [min\u2013max] across three seeds, with 1,200 missing "
              "query cells scored per seed. Values summarize error against benchmark "
              "ground truth. Similar aggregate errors do not establish equality of "
              "individual imputed values or algorithmic equivalence.", "", table(
        ["Regime", "dtype", "Method", "RMSE", "MAE"],
        [[c["training_policy"], c["dtype"], c["label"], formatted(c["quality"]["rmse"], 10),
          formatted(c["quality"]["mae"], 10)] for c in cells],
    ), "", "Agreement below uses recorded full-output SHA-256 hashes and worker-computed "
              "maximum absolute differences on scored cells. Hash counts cover nine timing "
              "pairs, but only three distinct seed inputs. Differences are maxima over those pairs.", "", table(
        ["Regime", "dtype", "Comparison", "Matching full-output hashes", "Max scored-cell difference", "Max RMSE difference", "Max MAE difference"],
        [[c["training_policy"], c["dtype"],
          f"{current} vs " + ("KNNImputer" if c["numerator_variant"] == "knn" else previous),
          f"{c['matching_output_hashes']}/{c['pair_count']}",
          f"{c['max_worker_scored_abs_difference']:.12g}",
          f"{c['max_absolute_rmse_difference']:.12g}", f"{c['max_absolute_mae_difference']:.12g}"]
         for c in comparisons if c["denominator_variant"] == "current"],
    ), "", "## Donor counts and observed missingness", "",
              "One row per policy/dtype/seed after verifying identical case metadata and "
              "input fingerprints across variants and repeats. Complete donors means fully "
              "observed training rows; available mode can also use partially observed rows "
              "as feature-specific donors, so this is not its total eligible donor count. "
              "Feature-specific donor counts were not recorded in this artifact.", "", table(
        ["Regime", "dtype", "Seed", "Complete donors", "Observed training missing (%)", "Query patterns", "Scored cells"],
        [[c["training_policy"], c["dtype"], c["seed"], c["complete_donors"],
          f"{100*c['train_missing_rate']:.4f}", c["query_patterns"], c["scored_cells"]]
         for c in result["cases"]],
    ), "", "## Interpretation and limits", "",
              *release_interpretation(result, profile),
              "- Startup, data generation, warmup, validation, RSS sampling, and garbage "
              "collection between timed phases are excluded from the measured timings. "
              "The additional transforms are warm calls on the already fitted model. "
              "These results do not establish behavior on all data sizes, hardware, or datasets.",
              "", "## Reproduction", "",
              reproduction,
              "", "[Analysis workflow](../../.github/workflows/analyze-released-versions.yml)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=tuple(RELEASES), default="0.3.21")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    profile = release_profile(args.release)
    input_path = args.input if args.input is not None else ROOT / profile["archive"]
    report_path = args.report if args.report is not None else ROOT / profile["report"]
    summary_path = args.summary if args.summary is not None else ROOT / profile["summary"]
    outputs = (report_path.resolve(), summary_path.resolve())
    protected = {input_path.resolve()} | {
        (ROOT / release_profile(version)["archive"]).resolve() for version in RELEASES
    }
    require(outputs[0] != outputs[1], "Report and summary must use different output paths")
    require(not protected.intersection(outputs), "Output would overwrite a raw archive")
    data, freezes = read_archive(input_path, profile)
    result = analyze(data, freezes, profile)
    report_text = render_report(result, profile)
    summary_text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for path, text in ((report_path, report_text), (summary_path, summary_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(f"Validated {args.release}: 108 workers and 12 stored summaries; "
          "generated report and full-precision JSON.")


if __name__ == "__main__":
    main()
