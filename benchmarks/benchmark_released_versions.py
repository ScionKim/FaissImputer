"""Compare released packages using fresh, sequential workers on one runner."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.benchmark_scaling_threads import (
    FEATURES,
    NEIGHBORS,
    ROOT,
    TRAIN_MISSING_RATE,
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
)


POLICIES = ("complete", "available")
DTYPES = ("float32", "float64")
VARIANTS = ("knn", "previous", "current")
COMMON_ENVIRONMENT = ("python", "numpy", "scikit_learn", "faiss")
MEASURES = (
    "fit_seconds",
    "transform_seconds",
    "total_seconds",
    "repeated_transform_median_seconds",
    "worker_peak_rss_mib",
    "fit_rss_change_mib",
    "rmse",
    "mae",
    "max_abs_difference_from_knn",
    "max_abs_difference_from_previous_release",
    "max_abs_difference_from_first_repeat",
)


def case_key(record):
    return tuple(
        record[name]
        for name in (
            "training_policy", "dtype", "size",
            "queries", "seed", "pattern",
        )
    )


def run_worker(python, config, timeout):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        env[name] = str(config["threads"])

    command = [
        python, "-u", "-m", "benchmarks.benchmark_scaling_threads",
        "--worker", json.dumps(config),
    ]
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    except OSError as error:
        return {"status": "process_error", "error": str(error)}

    if process.returncode:
        return {
            "status": "process_error",
            "returncode": process.returncode,
            "error": process.stderr[-4000:],
        }

    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "status": "invalid_worker_output",
            "error": process.stdout[-2000:] + process.stderr[-2000:],
        }

    if not isinstance(record, dict) or "status" not in record:
        return {"status": "invalid_worker_output"}

    record["worker_wall_seconds"] = time.perf_counter() - started
    if process.stderr.strip():
        record["warnings"] = process.stderr[-4000:]
    return record


def summarize(records, labels):
    summaries = []
    for policy in POLICIES:
        for dtype in DTYPES:
            for variant in VARIANTS:
                rows = [
                    record for record in records
                    if record["training_policy"] == policy
                    and record["dtype"] == dtype
                    and record["variant"] == variant
                ]
                successful = [
                    record for record in rows if record["status"] == "ok"
                ]
                summary = {
                    "training_policy": policy,
                    "dtype": dtype,
                    "variant": variant,
                    "label": labels[variant],
                    "planned_workers": len(rows),
                    "successful_workers": len(successful),
                }
                for name in MEASURES:
                    values = [
                        float(record[name])
                        for record in successful
                        if record.get(name) is not None
                    ]
                    summary[name] = (
                        {
                            "median": median(values),
                            "min": min(values),
                            "max": max(values),
                        }
                        if values else None
                    )
                summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-python", type=Path, required=True)
    parser.add_argument("--previous-version", default="0.3.16")
    parser.add_argument("--current-version", default="0.3.19")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeated-transforms", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs/released_versions.json",
    )
    args = parser.parse_args()

    if args.train_size < NEIGHBORS:
        parser.error(f"train-size must be at least {NEIGHBORS}")
    if min(
        args.queries, args.repeats, args.repeated_transforms,
        args.timeout_seconds, args.budget_seconds,
    ) < 1:
        parser.error("query, repeat, timeout and budget values must be positive")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and unique")
    if args.current_version == args.previous_version:
        parser.error("current and previous versions must differ")
    if not args.previous_python.is_file():
        parser.error("previous-python must identify an existing interpreter")

    check_released_package(args.current_version)
    environment = metadata()

    labels = {
        "knn": "KNNImputer",
        "previous": f"FaissImputer {args.previous_version}",
        "current": f"FaissImputer {args.current_version}",
    }
    versions = {
        "knn": args.current_version,
        "previous": args.previous_version,
        "current": args.current_version,
    }
    interpreters = {
        "knn": sys.executable,
        "current": sys.executable,
        # Keep the venv path rather than resolving its executable symlink.
        "previous": str(args.previous_python.absolute()),
    }

    configs = []
    for seed_index, seed in enumerate(args.seeds):
        for policy in POLICIES:
            for dtype in DTYPES:
                for repeat in range(args.repeats):
                    offset = (seed_index + repeat) % len(VARIANTS)
                    order = VARIANTS[offset:] + VARIANTS[:offset]
                    for variant in order:
                        configs.append({
                            "variant": variant,
                            "method": (
                                "KNNImputer" if variant == "knn"
                                else f"FaissImputer[{policy}]"
                            ),
                            "expected_version": versions[variant],
                            "training_policy": policy,
                            "dtype": dtype,
                            "size": args.train_size,
                            "queries": args.queries,
                            "seed": seed,
                            "pattern": "random",
                            "threads": 1,
                            "repeat": repeat + 1,
                            "prefix_sizes": [args.train_size],
                            "repeated_transforms": args.repeated_transforms,
                        })

    results = {
        "schema_version": 1,
        "metadata": environment,
        "parameters": {
            "labels": labels,
            "faiss_imputer_environment_versions": versions,
            "interpreters": interpreters,
            "train_size": args.train_size,
            "queries": args.queries,
            "features": FEATURES,
            "n_neighbors": NEIGHBORS,
            "metric": "l2 / nan_euclidean",
            "weights": "uniform",
            "training_policies": list(POLICIES),
            "training_missing_rates": {
                "complete": 0.0,
                "available": TRAIN_MISSING_RATE,
            },
            "dtypes": list(DTYPES),
            "query_pattern": "random",
            "threads": 1,
            "seeds": args.seeds,
            "repeats": args.repeats,
            "repeated_transforms": args.repeated_transforms,
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "worker_timeout_seconds": args.timeout_seconds,
            "run_budget_seconds": args.budget_seconds,
            "expected_workers": len(configs),
        },
        "notes": [
            "Fresh sequential workers; method order rotates across repetitions.",
            "KNNImputer runs in the current-release environment.",
            "Python, NumPy, scikit-learn and Faiss versions must match across workers.",
            "Within each policy, dtype and seed, all methods receive identical inputs.",
            "Complete-policy cases use fully observed training data; available-policy "
            "cases use partially observed training data.",
            "Float64 data is generated without a float32 round trip.",
            "Timings exclude startup, data generation, warmup, validation, "
            "RSS sampling and garbage collection between timed phases.",
            "Worker peak RSS covers the full worker lifetime. Post-fit RSS change "
            "includes allocator effects and is not an exact fitted-model size.",
            "RMSE and MAE measure error against hidden synthetic truth. "
            "Differences from other imputers measure output agreement.",
            "Output comparisons pair matching cases and repetitions. "
            "The first-repeat comparison uses the first successful repetition.",
            "Summary statistics describe successful workers; failures and unrun "
            "cases remain in the records and make the command fail.",
            "The Git commit identifies benchmark code; installed package versions "
            "and module paths identify the measured distributions.",
        ],
        "records": [],
        "summaries": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    input_references = {}
    paired_outputs = {}
    first_outputs = {}
    outputs = []
    save()
    started = time.perf_counter()

    for index, config in enumerate(configs, start=1):
        remaining = args.budget_seconds - (time.perf_counter() - started)
        record = (
            run_worker(
                interpreters[config["variant"]],
                config,
                min(args.timeout_seconds, remaining),
            )
            if remaining > 0 else {"status": "not_run_budget"}
        )
        raw_values = record.pop("_values", None)
        record = {**config, **record}

        if record["status"] == "ok":
            try:
                for name in COMMON_ENVIRONMENT:
                    if record["environment"][name] != environment[name]:
                        raise ValueError(f"Environment mismatch: {name}")
                if record["faiss_omp_threads"] != config["threads"]:
                    raise ValueError("Unexpected Faiss thread count")

                key = case_key(record)
                fingerprints = {
                    name: record["fingerprints"][name]
                    for name in ("train", "query", "truth")
                }
                expected = input_references.setdefault(key, fingerprints)
                if fingerprints != expected:
                    raise ValueError("Inputs differ across workers")

                values = np.asarray(raw_values, dtype=np.float64)
                if (
                    values.ndim != 1
                    or values.size == 0
                    or values.size != record["scored_cells"]
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid comparison values")

                paired_outputs[
                    (key, record["repeat"], record["variant"])
                ] = values
                first_outputs.setdefault((key, record["variant"]), values)
                outputs.append((record, values))
            except (KeyError, TypeError, ValueError) as error:
                record["status"] = "validation_error"
                record["checks_passed"] = False
                record["error"] = str(error)

        results["records"].append(record)
        print(
            f"{index}/{len(configs)} "
            f"{config['training_policy']} {config['dtype']} "
            f"seed={config['seed']} repeat={config['repeat']} "
            f"{labels[config['variant']]}: {record['status']}",
            flush=True,
        )
        if record.get("error"):
            print(record["error"], flush=True)
        save()

    for record, values in outputs:
        key = case_key(record)
        comparisons = {
            "max_abs_difference_from_knn": paired_outputs.get(
                (key, record["repeat"], "knn")
            ),
            "max_abs_difference_from_previous_release": paired_outputs.get(
                (key, record["repeat"], "previous")
            ),
            "max_abs_difference_from_first_repeat": first_outputs.get(
                (key, record["variant"])
            ),
        }
        for name, reference in comparisons.items():
            record[name] = (
                None if reference is None
                else float(np.max(np.abs(values - reference)))
            )

    results["summaries"] = summarize(results["records"], labels)
    results["elapsed_seconds"] = time.perf_counter() - started
    save()
    print(f"Results: {args.output}", flush=True)
    return int(any(
        record["status"] != "ok" for record in results["records"]
    ))


if __name__ == "__main__":
    raise SystemExit(main())