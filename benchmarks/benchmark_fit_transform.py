"""Compare fit_transform with consecutive fit and transform calls."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.benchmark_fit_transform_worker import (
    APIS,
    TARGET_MISSING_RATE,
)
from benchmarks.benchmark_scaling_threads import (
    FEATURES,
    METHODS,
    NEIGHBORS,
    ROOT,
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
)


DTYPES = ("float32", "float64")
GROUP_FIELDS = ("size", "dtype", "method", "api")
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
    "total_seconds",
    "fit_seconds",
    "transform_seconds",
    "fit_transform_seconds",
    "worker_peak_rss_mib",
    "fit_phase_peak_rss_mib",
    "transform_phase_peak_rss_mib",
    "rss_before_fit_mib",
    "rss_after_fit_mib",
    "fit_retained_rss_mib",
    "rmse",
    "mae",
)

# Signed deltas validated for finiteness only; allocator behavior can
# in principle make a retained-memory delta negative.
SIGNED_MEASURES = ("fit_retained_rss_mib",)

def run_worker(config, timeout):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"

    command = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.benchmark_fit_transform_worker",
        "--config",
        json.dumps(config),
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

    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(record, dict) or "status" not in record:
            raise ValueError("Missing worker status")
    except (ValueError, IndexError):
        return {
            "status": "invalid_worker_output",
            "error": process.stdout[-2000:] + process.stderr[-4000:],
        }

    record["worker_wall_seconds"] = time.perf_counter() - started
    if process.returncode:
        record["returncode"] = process.returncode
        if record["status"] == "ok":
            record["status"] = "process_error"
    if process.stderr.strip():
        record["worker_stderr"] = process.stderr[-4000:]
    return record


def validate_record(record, values, environment, inputs, outputs):
    if record.get("checks_passed") is not True:
        raise ValueError("Worker checks did not pass")
    if record["input_dtype"] != record["dtype"]:
        raise ValueError("Unexpected input dtype")
    if record["output_dtype"] != record["dtype"]:
        raise ValueError("Unexpected output dtype")

    for field in ENVIRONMENT_FIELDS:
        if record["environment"].get(field) != environment.get(field):
            raise ValueError(f"Worker environment differs: {field}")

    if record.get("faiss_omp_threads") != 1:
        raise ValueError("Faiss thread count differs from one")
    if any(
        pool.get("num_threads") != 1
        for pool in record.get("threadpools", [])
    ):
        raise ValueError("Native thread-pool limit differs from one")

    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or values.size != record["scored_cells"]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid imputed-value comparison payload")

    for measure in MEASURES:
        value = record.get(measure)
        if value is None:
            continue
        if not np.isfinite(value):
            raise ValueError(f"Invalid measurement: {measure}")
        if measure not in SIGNED_MEASURES and value < 0:
            raise ValueError(f"Invalid measurement: {measure}")

    case = (record["size"], record["dtype"], record["seed"])
    fingerprints = record["fingerprints"]
    if case in inputs and inputs[case] != fingerprints:
        raise ValueError("Methods or APIs received different inputs")

    output_key = case + (record["method"],)
    reference = outputs.get(output_key)
    difference = 0.0
    if reference is not None:
        reference_hash, reference_values = reference
        if reference_values.shape != values.shape:
            raise ValueError("Output comparison lengths differ")
        difference = float(np.max(np.abs(values - reference_values)))
        record["max_abs_difference_from_first_success"] = difference
        if (
            not np.array_equal(values, reference_values)
            or record["output_sha256"] != reference_hash
        ):
            raise ValueError(
                "Outputs differ across APIs or repetitions "
                "for the same method"
            )

    # Only validated results may become comparison references.
    inputs.setdefault(case, fingerprints)
    outputs.setdefault(output_key, (record["output_sha256"], values))
    record["max_abs_difference_from_first_success"] = difference


def summarize(records, configs):
    planned = Counter(
        tuple(config[field] for field in GROUP_FIELDS)
        for config in configs
    )
    groups = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in GROUP_FIELDS)
        groups[key].append(record)

    summaries = []
    for key, count in planned.items():
        rows = groups[key]
        successful = [row for row in rows if row["status"] == "ok"]
        summary = {
            **dict(zip(GROUP_FIELDS, key)),
            "planned_workers": count,
            "recorded_workers": len(rows),
            "successful_workers": len(successful),
            "pending_workers": count - len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        }
        for measure in MEASURES:
            values = [
                row[measure]
                for row in successful
                if row.get(measure) is not None
            ]
            if values:
                summary[measure] = {
                    "median": median(values),
                    "min": min(values),
                    "max": max(values),
                }
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 202, 303]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.10)
    parser.add_argument(
        "--phase-memory",
        action="store_true",
        help=(
            "Measure phase-separated memory (retained fitted memory "
            "and fit/transform peak RSS) in each worker. Off by default "
            "so timing runs keep the original benchmark conditions."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs/fit_transform.json",
    )
    args = parser.parse_args()

    if min(args.sizes) <= args.neighbors:
        parser.error(f"sizes must exceed {args.neighbors}")
    if args.features <= 0:
        parser.error("features must be positive")
    if args.neighbors <= 0:
        parser.error("neighbors must be positive")
    if not 0.0 < args.missing_rate < 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and unique")
    if min(
        args.repeats, args.timeout_seconds, args.budget_seconds
    ) < 1:
        parser.error("repeats and time limits must be positive")

    check_released_package(args.expected_version)
    environment = metadata()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("version") != args.expected_version:
        parser.error("provenance version does not match expected-version")
    if provenance.get("source_commit") != environment["git_commit"]:
        parser.error("provenance commit does not match benchmark source")
    wheel_hash = provenance.get("wheel_sha256", "")
    if (
        not isinstance(wheel_hash, str)
        or len(wheel_hash) != 64
        or any(character not in "0123456789abcdef" for character in wheel_hash)
    ):
        parser.error("provenance must include a SHA-256 wheel hash")

    pairs = [(method, api) for method in METHODS for api in APIS]
    configs = []
    for seed_index, seed in enumerate(args.seeds):
        for size in args.sizes:
            for dtype in DTYPES:
                for repeat in range(args.repeats):
                    offset = (
                        seed_index * args.repeats + repeat
                    ) % len(pairs)
                    order = pairs[offset:] + pairs[:offset]
                    for method, api in order:
                        configs.append({
                            "method": method,
                            "api": api,
                            "size": size,
                            "dtype": dtype,
                            "seed": seed,
                            "repeat": repeat + 1,
                            "expected_version": args.expected_version,
                            "measure_phase_memory": args.phase_memory,
                            "features": args.features,
                            "n_neighbors": args.neighbors,
                            "target_missing_rate": args.missing_rate,
                        })

    results = {
        "schema_version": 1,
        "metadata": environment,
        "provenance": provenance,
        "parameters": {
            "sizes": args.sizes,
            "dtypes": list(DTYPES),
            "methods": list(METHODS),
            "apis": list(APIS),
            "seeds": args.seeds,
            "repeats": args.repeats,
            "features": args.features,
            "n_neighbors": args.neighbors,
            "target_missing_rate": args.missing_rate,
            "guaranteed_complete_rows": args.neighbors,
            "threads": 1,
            "phase_memory_measured": args.phase_memory,
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "worker_timeout_seconds": args.timeout_seconds,
            "run_budget_seconds": args.budget_seconds,
            "expected_workers": len(configs),
        },
        "notes": [
            "One fresh sequential worker per method/API measurement.",
            "Method/API order rotates across seeds and repetitions.",
            "All methods receive the same incomplete training data.",
            "The first five rows are kept complete; actual missingness is recorded.",
            "Float64 data is generated without a float32 round trip.",
            "Both APIs use a fresh estimator after a small untimed warmup.",
            "By default, split fit and transform are timed consecutively without "
            "explicit GC or memory sampling between them.",
            "Phase-separated memory measurement is opt-in via "
            "--phase-memory. In that mode, sampler setup/teardown, "
            "garbage collection, and RSS snapshots sit outside the "
            "timed intervals: fit_seconds covers only model.fit() and "
            "transform_seconds only model.transform(). Timings from "
            "--phase-memory runs are for reference only; the sampler "
            "thread adds observer overhead, so they are not canonical "
            "performance comparisons. Default runs preserve the "
            "original benchmark conditions and leave the phase-memory "
            "fields null.",
            "Output equality is required across APIs and repetitions "
            "of the same method, not across different methods.",
            "RMSE and MAE describe reconstruction of masked training entries.",
            "Peak RSS includes preparation and validation and is sampled "
            "before JSON serialization; it is not fitted-model memory.",
            "Retained fitted memory is the current-RSS change across fit, "
            "with a garbage-collection pause before each snapshot. "
            "Allocator-retained memory can exceed live model objects.",
            "Phase peaks are background-sampled current-RSS maxima during "
            "each phase: lower bounds on the true phase peaks that include "
            "baseline process memory. They are null for the fit_transform "
            "API, which cannot attribute memory to separate phases.",
            "Summary statistics include successful workers only; "
            "any failed or unrun worker makes the command fail.",
        ],
        "records": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    inputs = {}
    outputs = {}
    started = time.perf_counter()

    def save_results():
        results["elapsed_seconds"] = time.perf_counter() - started
        results["summaries"] = summarize(results["records"], configs)
        temporary.write_text(
            json.dumps(results, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(args.output)

    save_results()
    for index, config in enumerate(configs, start=1):
        remaining = args.budget_seconds - (time.perf_counter() - started)
        payload = (
            run_worker(config, min(args.timeout_seconds, remaining))
            if remaining > 0
            else {"status": "not_run_budget"}
        )
        values = payload.pop("_values", None)
        record = {**payload, **config}

        if record["status"] == "ok":
            try:
                validate_record(record, values, environment, inputs, outputs)
            except Exception as error:
                record["status"] = "validation_error"
                record["checks_passed"] = False
                record["error"] = str(error)

        results["records"].append(record)
        save_results()
        print(
            f"{index}/{len(configs)} "
            f"rows={config['size']} {config['dtype']} "
            f"seed={config['seed']} repeat={config['repeat']} "
            f"{config['method']} {config['api']}: {record['status']}",
            flush=True,
        )

    print(f"Results: {args.output}", flush=True)
    return int(any(row["status"] != "ok" for row in results["records"]))


if __name__ == "__main__":
    raise SystemExit(main())