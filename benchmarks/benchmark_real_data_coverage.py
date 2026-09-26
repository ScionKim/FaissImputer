"""Run isolated real-data imputation comparisons and archive results."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
import time

import numpy as np

from benchmarks.benchmark_real_data_cases import (
    DATASET_DEFAULTS,
    DTYPES,
    MECHANISMS,
    MISSING_RATE,
    load_dataset,
)
from benchmarks.benchmark_real_data_worker import METHODS
from benchmarks.benchmark_scaling_threads import (
    ROOT,
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
)


CASE_FIELDS = (
    "dataset_id", "train_size", "query_size", "mechanism", "dtype", "seed",
)
GROUP_FIELDS = (
    "dataset_id", "train_size", "query_size", "mechanism", "dtype", "method",
)
ENVIRONMENT_FIELDS = (
    "python",
    "numpy",
    "scikit_learn",
    "faiss",
    "faiss_imputer",
    "cpu_model",
    "platform",
    "git_commit",
)
MEASURES = (
    "fit_seconds",
    "transform_seconds",
    "total_seconds",
    "worker_peak_rss_mib",
    "rmse",
    "mae",
)


def config_key(record, fields):
    # Records written before dataset selection implicitly use California.
    return tuple(
        record.get(name, "california_housing")
        if name == "dataset_id" else record[name]
        for name in fields
    )


def case_key(record):
    return config_key(record, CASE_FIELDS)


def distribution(values):
    return {
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_worker(config, timeout):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"

    started = time.perf_counter()
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "benchmarks.benchmark_real_data_worker",
                "--config",
                json.dumps(config),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "timeout_seconds": timeout,
            "wall_seconds": time.perf_counter() - started,
        }
    except OSError as error:
        return {"status": "process_error", "error": str(error)}

    try:
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(payload, dict) or "status" not in payload:
            raise ValueError("Worker result must contain a status")
    except (IndexError, ValueError):
        payload = {
            "status": "invalid_worker_output",
            "worker_stdout": process.stdout[-2000:],
        }

    if process.returncode:
        if payload["status"] == "ok":
            payload["status"] = "process_error"
        payload["returncode"] = process.returncode

    if process.stderr:
        payload["worker_stderr"] = process.stderr[-4000:]
    payload["wall_seconds"] = time.perf_counter() - started
    return payload


def validate_record(
    record,
    values,
    environment,
    dataset,
    case_references,
    query_references,
    output_references,
):
    if record.get("checks_passed") is not True:
        raise ValueError("Worker checks did not pass")
    if (
        record["input_dtype"] != record["dtype"]
        or record["output_dtype"] != record["dtype"]
    ):
        raise ValueError("Unexpected input or output dtype")

    if any(
        record["environment"].get(name) != environment[name]
        for name in ENVIRONMENT_FIELDS
    ):
        raise ValueError("Worker environment differs from the controller")
    if record["dataset"] != dataset:
        raise ValueError("Worker dataset differs from the controller")
    if (
        record["faiss_omp_threads"] != 1
        or any(
            pool.get("num_threads") != 1
            for pool in record["threadpools"]
        )
    ):
        raise ValueError("Worker native thread count differs from one")

    case = record["case"]
    defaults = DATASET_DEFAULTS[
        record.get("dataset_id", "california_housing")
    ]
    expected_case = {
        "always_observed": [record.get("mar_driver", defaults["mar_driver"])],
        "nominal_overall_missing_rate": record.get(
            "missing_rate", MISSING_RATE
        ),
        "mar_reference_rows": (
            record.get("mar_reference_rows", defaults["mar_reference_rows"])
            if record["mechanism"] == "MAR" else None
        ),
        "n_train": record["train_size"],
        "n_query": record["query_size"],
        "seed": record["seed"],
        "mechanism": record["mechanism"],
        "input_dtype": record["dtype"],
        "truth_dtype": "float64",
    }
    if any(case.get(name) != value for name, value in expected_case.items()):
        raise ValueError("Worker case does not match its configuration")
    if case["feature_names"] != dataset["feature_names"]:
        raise ValueError("Feature names differ from the source dataset")
    if case["fingerprints"]["dataset"] != dataset["dataset_sha256"]:
        raise ValueError("Prepared case has an unexpected dataset fingerprint")

    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or values.size != record["scored_cells"]
    ):
        raise ValueError("Invalid output comparison payload")

    for name in MEASURES:
        value = record.get(name)
        if value is None and name == "worker_peak_rss_mib":
            continue
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measurement: {name}")
        if name == "worker_peak_rss_mib" and value == 0:
            raise ValueError("Peak RSS must be positive when available")

    if not math.isclose(
        record["fit_seconds"] + record["transform_seconds"],
        record["total_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("Fit and transform times do not match total time")

    feature_quality = record["feature_quality"]
    counts = case["query_mask"]["missing_per_feature"]
    if (
        len(feature_quality) != len(dataset["feature_names"])
        or len(counts) != len(feature_quality)
        or sum(counts) != record["scored_cells"]
    ):
        raise ValueError("Feature scoring counts do not match the total")

    for name, count, feature in zip(
        dataset["feature_names"], counts, feature_quality
    ):
        if (
            count < 0
            or feature["feature"] != name
            or feature["scored_cells"] != count
        ):
            raise ValueError("Invalid feature scoring metadata")
        for metric in (
            "rmse_standardized",
            "mae_standardized",
            "rmse_original_units",
            "mae_original_units",
        ):
            value = feature[metric]
            if count == 0:
                if value is not None:
                    raise ValueError("Unscored features must have null errors")
            elif value is None or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid feature reconstruction error")

    key = case_key(record)
    if key in case_references and case != case_references[key]:
        raise ValueError("Matching cases received different prepared inputs")

    query_key = (
        record.get("dataset_id", "california_housing"),
        record["query_size"],
        record["mechanism"],
        record["seed"],
    )
    query_signature = tuple(
        case["fingerprints"][name]
        for name in ("query_row_ids", "raw_query", "query_mask")
    ) + (case["mar_cutoff"], case["mar_reference_rows"])
    if (
        query_key in query_references
        and query_signature != query_references[query_key]
    ):
        raise ValueError("Raw query rows or masks changed across training sizes")

    output_key = key + (record["method"],)
    previous = output_references.get(output_key)
    if previous is not None and (
        record["output_sha256"] != previous["sha256"]
        or not np.array_equal(values, previous["values"])
    ):
        raise ValueError("Repeated outputs differ within the same method")

    # Register references only after every validation above succeeds.
    case_references.setdefault(key, case)
    query_references.setdefault(query_key, query_signature)
    output_references.setdefault(
        output_key,
        {"sha256": record["output_sha256"], "values": values},
    )


def update_agreement(key, records, output_references):
    """Fill comparisons only when a successful KNN reference is available."""
    baseline = output_references.get(key + ("KNNImputer",))
    if baseline is None:
        return

    comparisons = {}
    for method in METHODS:
        reference = output_references.get(key + (method,))
        if reference is None:
            continue
        difference = reference["values"] - baseline["values"]
        comparisons[method] = {
            "scored_cells": int(difference.size),
            "max_abs_difference": float(np.max(np.abs(difference))),
        }

    for record in records:
        if record["status"] == "ok" and case_key(record) == key:
            record["agreement_with_knn"] = comparisons[record["method"]]


def summarize(records, configs):
    planned = Counter(
        config_key(config, GROUP_FIELDS)
        for config in configs
    )
    grouped = defaultdict(list)
    for record in records:
        grouped[config_key(record, GROUP_FIELDS)].append(record)

    summaries = []
    for key, expected in planned.items():
        rows = grouped[key]
        successful = [row for row in rows if row["status"] == "ok"]
        summary = dict(zip(GROUP_FIELDS, key))
        summary.update({
            "planned_workers": expected,
            "recorded_workers": len(rows),
            "successful_workers": len(successful),
            "pending_workers": expected - len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        })

        for name in MEASURES:
            values = [
                row[name] for row in successful
                if row.get(name) is not None
            ]
            if values:
                summary[name] = distribution(values)

        comparisons = [
            row["agreement_with_knn"]["max_abs_difference"]
            for row in successful
            if row.get("agreement_with_knn") is not None
        ]
        summary["agreement_with_knn"] = {
            "compared_workers": len(comparisons),
            "max_abs_difference": (
                distribution(comparisons) if comparisons else None
            ),
        }
        summaries.append(summary)

    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_DEFAULTS),
        default="california_housing",
    )
    parser.add_argument("--train-sizes", type=int, nargs="+")
    parser.add_argument("--query-size", type=int)
    parser.add_argument("--mar-reference-rows", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs" / "real_data_coverage.json",
    )
    args = parser.parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.train_sizes is None:
        args.train_sizes = list(defaults["train_sizes"])
    if args.query_size is None:
        args.query_size = defaults["query_size"]
    if args.mar_reference_rows is None:
        args.mar_reference_rows = defaults["mar_reference_rows"]
    mar_driver = defaults["mar_driver"]

    if (
        args.query_size < 1
        or args.repeats < 1
        or args.timeout_seconds < 1
        or args.budget_seconds < 1
        or any(seed < 0 for seed in args.seeds)
        or args.mar_reference_rows < 1
        or any(size < args.mar_reference_rows for size in args.train_sizes)
    ):
        parser.error(
            "Positive sizes, repetitions and time limits are required; "
            "seeds must be nonnegative and training sizes must cover "
            "the positive MAR reference prefix"
        )
    if (
        len(set(args.train_sizes)) != len(args.train_sizes)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Training sizes and seeds must not contain duplicates")

    check_released_package(args.expected_version)
    environment = metadata()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("version") != args.expected_version:
        parser.error("Provenance version differs from the installed candidate")
    if provenance.get("source_commit") != environment["git_commit"]:
        parser.error("Provenance source commit differs from the benchmark")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(provenance.get("wheel_sha256", ""))
    ):
        parser.error("Provenance must contain a lowercase wheel SHA256")

    data_home = args.data_home.expanduser().resolve()
    data, names, dataset = load_dataset(
        data_home,
        download_if_missing=False,
        dataset_id=args.dataset,
    )
    if max(args.train_sizes) + args.query_size > len(data):
        parser.error("Insufficient rows for the requested disjoint splits")
    del data, names

    configs = []
    for seed_index, seed in enumerate(args.seeds):
        for train_size in args.train_sizes:
            for mechanism in MECHANISMS:
                for dtype in DTYPES:
                    for repeat in range(args.repeats):
                        offset = (seed_index * args.repeats + repeat) % len(METHODS)
                        order = METHODS[offset:] + METHODS[:offset]
                        for method in order:
                            configs.append({
                                "method": method,
                                "dataset_id": args.dataset,
                                "mar_driver": mar_driver,
                                "train_size": train_size,
                                "query_size": args.query_size,
                                "mechanism": mechanism,
                                "dtype": dtype,
                                "seed": seed,
                                "repeat": repeat + 1,
                                "missing_rate": MISSING_RATE,
                                "mar_reference_rows": args.mar_reference_rows,
                                "data_home": str(data_home),
                                "expected_version": args.expected_version,
                            })

    results = {
        "schema_version": 2,
        "metadata": environment,
        "provenance": provenance,
        "dataset": dataset,
        "parameters": {
            "dataset_id": args.dataset,
            "mar_driver": mar_driver,
            "train_sizes": args.train_sizes,
            "query_size": args.query_size,
            "mechanisms": list(MECHANISMS),
            "dtypes": list(DTYPES),
            "methods": list(METHODS),
            "seeds": args.seeds,
            "repeats": args.repeats,
            "nominal_overall_missing_rate": MISSING_RATE,
            "mar_reference_rows": args.mar_reference_rows,
            "threads": 1,
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "worker_timeout_seconds": args.timeout_seconds,
            "run_budget_seconds": args.budget_seconds,
            "expected_workers": len(configs),
        },
        "notes": [
            "Real feature values with artificial missingness; target unused.",
            "Training and query rows are disjoint.",
            "Raw query rows and masks are shared across training sizes.",
            (
                f"MAR uses the {mar_driver} median of a common "
                f"{args.mar_reference_rows}-row training prefix."
            ),
            (
                f"{mar_driver} remains observed under MCAR and MAR; "
                "actual missingness is recorded."
            ),
            "Original-unit feature errors refer to supplied source values.",
            "Each case is scaled using its observed training values only.",
            "Standardized inputs and error units can change across cases.",
            "One fresh sequential worker per method and repetition.",
            "Method order rotates across seeds and repetitions.",
            "Each worker uses a separate small untimed warmup.",
            "Fit and first transform are timed consecutively.",
            "Timing excludes loading, preparation, scoring, and validation.",
            "Quality compares outputs with hidden held-out float64 truth.",
            "Original-unit errors are reported separately for each feature.",
            "KNN output agreement is separate from reconstruction quality.",
            "Missing KNN comparisons remain null, not zero.",
            "Repeated timings are not independent accuracy observations.",
            "Peak RSS includes setup and validation, before JSON serialization.",
            "Peak RSS is not retained-model or transform-only memory.",
            "Summaries include successful workers only.",
            "Any failed, inapplicable, or unrun worker makes the command fail.",
        ],
        "records": [],
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    case_references = {}
    query_references = {}
    output_references = {}
    started = time.perf_counter()

    def save_results():
        results["elapsed_seconds"] = time.perf_counter() - started
        results["summaries"] = summarize(results["records"], configs)
        temporary.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)

    save_results()
    for position, config in enumerate(configs, start=1):
        remaining = args.budget_seconds - (time.perf_counter() - started)
        if remaining > 0:
            payload = run_worker(
                config,
                min(args.timeout_seconds, remaining),
            )
        else:
            payload = {"status": "not_run_budget"}

        values = payload.pop("_values", None)
        record = {**payload, **config, "agreement_with_knn": None}

        if record["status"] == "ok":
            try:
                validate_record(
                    record,
                    values,
                    environment,
                    dataset,
                    case_references,
                    query_references,
                    output_references,
                )
            except Exception as error:
                record["status"] = "validation_error"
                record["checks_passed"] = False
                record["error"] = str(error)

        results["records"].append(record)
        if record["status"] == "ok":
            update_agreement(
                case_key(record),
                results["records"],
                output_references,
            )

        save_results()
        print(
            f"{position}/{len(configs)} "
            f"train={config['train_size']} "
            f"{config['mechanism']} {config['dtype']} "
            f"seed={config['seed']} repeat={config['repeat']} "
            f"{config['method']}: {record['status']}",
            flush=True,
        )

    print(f"Results: {output}", flush=True)
    return int(any(row["status"] != "ok" for row in results["records"]))


if __name__ == "__main__":
    raise SystemExit(main())