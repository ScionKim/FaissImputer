"""Compare installed releases on one held-out dataset and dtype per run."""

import argparse
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
    MISSING_RATE,
    N_NEIGHBORS,
    UCI_DATASETS,
    load_dataset,
)
from benchmarks.benchmark_released_versions import COMMON_ENVIRONMENT, VARIANTS
from benchmarks.benchmark_scaling_threads import (
    ROOT,
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
)


DATASET_ID = "wine_quality_white"
DATASETS = (DATASET_ID, "abalone")
SEEDS = (101, 202, 303)
REPEATS = 3
WEIGHTS = ("uniform", "distance")
TIMINGS = ("fit_seconds", "transform_seconds", "total_seconds")
# Keep legacy uniform-weight match dictionaries unchanged. Non-default
# weights are included by match_configuration below.
MATCH_FIELDS = (
    "dataset_id", "api", "training_policy", "train_size", "query_size",
    "features", "n_neighbors", "missing_rate", "mechanism", "dtype",
    "mar_reference_rows", "mar_driver", "threads", "seed", "repeat",
)
FINGERPRINTS = (
    "dataset", "train_row_ids", "query_row_ids", "raw_train", "raw_query",
    "train_mask", "query_mask", "train", "query", "truth",
)


def record_weights(record):
    """Interpret legacy records without a weights field as uniform."""
    weights = record.get("weights", "uniform")
    if weights not in WEIGHTS:
        raise ValueError(f"Unsupported weights: {weights!r}")
    return weights


def match_configuration(record):
    """Include weights in identity without changing archived uniform pairs."""
    weights = record_weights(record)
    match = {name: record[name] for name in MATCH_FIELDS}
    if weights != "uniform":
        match["weights"] = weights
    return match


def build_configs(
    previous_version, current_version, data_home, *,
    dataset_id=DATASET_ID, dtype="float64", weights="uniform",
):
    if dataset_id not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    record_weights({"weights": weights})
    defaults = DATASET_DEFAULTS[dataset_id]
    features = len(UCI_DATASETS[dataset_id]["feature_names"])
    configs = []
    for seed_index, seed in enumerate(SEEDS):
        for repeat in range(1, REPEATS + 1):
            offset = (seed_index + repeat - 1) % len(VARIANTS)
            for variant in VARIANTS[offset:] + VARIANTS[:offset]:
                configs.append({
                    "variant": variant,
                    "method": (
                        "KNNImputer" if variant == "knn"
                        else "FaissImputer[available]"
                    ),
                    "expected_version": (
                        previous_version if variant == "previous"
                        else current_version
                    ),
                    "dataset_id": dataset_id,
                    "data_home": str(data_home),
                    "api": "fit_then_transform",
                    "training_policy": "available",
                    "train_size": 3000,
                    "query_size": 1000,
                    "features": features,
                    "n_neighbors": N_NEIGHBORS,
                    # Preserve the shape of historical uniform configurations.
                    **({"weights": weights} if weights != "uniform" else {}),
                    "dtype": dtype,
                    "mechanism": "MCAR",
                    "missing_rate": MISSING_RATE,
                    "mar_reference_rows": defaults["mar_reference_rows"],
                    "mar_driver": defaults["mar_driver"],
                    "threads": 1,
                    "seed": seed,
                    "repeat": repeat,
                })
    return configs


def run_worker(python, config, timeout):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [str(python), "-u", "-m", "benchmarks.benchmark_real_data_worker",
             "--config", json.dumps(config)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    except OSError as error:
        return {"status": "process_error", "error": str(error)}
    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(record, dict) or "status" not in record:
            raise ValueError("Worker must return a status object")
    except (ValueError, IndexError):
        return {
            "status": "invalid_worker_output",
            "error": process.stdout[-2000:] + process.stderr[-2000:],
        }
    if process.returncode:
        record["status"] = "process_error"
        record["returncode"] = process.returncode
        record.setdefault("error", process.stderr[-4000:])
    record["worker_wall_seconds"] = time.perf_counter() - started
    if process.stderr.strip():
        record["warnings"] = process.stderr[-4000:]
    return record


def require_number(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"Invalid {name}: {value!r}")


def validate_record(record, config, environment, dataset):
    if record.get("checks_passed") is not True:
        raise ValueError("Worker checks did not pass")
    weights = record_weights(config)
    if record_weights(record) != weights:
        raise ValueError("Worker weights differ from the requested configuration")
    for name in (*COMMON_ENVIRONMENT, "platform", "cpu_model", "git_commit"):
        if record["environment"][name] != environment[name]:
            raise ValueError(f"Environment mismatch: {name}")
    if record["environment"]["faiss_imputer"] != config["expected_version"]:
        raise ValueError("Unexpected installed package version")
    if record["dataset"] != dataset:
        raise ValueError("Dataset provenance differs from the parent")
    if (
        record["input_dtype"] != config["dtype"]
        or record["output_dtype"] != config["dtype"]
    ):
        raise ValueError("Unexpected input or output dtype")
    if record["threads"] != 1 or record["faiss_omp_threads"] != 1:
        raise ValueError("Unexpected thread count")
    if not record["threadpools"] or any(
        pool["num_threads"] != 1 for pool in record["threadpools"]
    ):
        raise ValueError("Native thread pools must use one thread")
    if record["sklearn_working_memory_mib"] != WORKING_MEMORY_MIB:
        raise ValueError("Unexpected sklearn working-memory setting")

    expected_model = {
        "n_neighbors": N_NEIGHBORS, "weights": weights, "copy": True,
        "metric": "nan_euclidean" if config["variant"] == "knn" else "l2",
    }
    if config["variant"] != "knn":
        expected_model.update(
            strategy="mean", donor_policy="available", index_factory="Flat"
        )
    if record["model_parameters"] != expected_model:
        raise ValueError("Unexpected model parameters")

    case = record["case"]
    expected_case = {
        "seed": config["seed"], "mechanism": config["mechanism"],
        "input_dtype": config["dtype"], "truth_dtype": "float64",
        "n_train": config["train_size"], "n_query": config["query_size"],
        "nominal_overall_missing_rate": config["missing_rate"],
        "feature_names": dataset["feature_names"],
        "always_observed": [config["mar_driver"]],
    }
    if any(case[name] != value for name, value in expected_case.items()):
        raise ValueError("Prepared case does not match the requested configuration")
    if len(case["feature_names"]) != config["features"]:
        raise ValueError("Unexpected feature count")
    for name in FINGERPRINTS:
        if not re.fullmatch(r"[0-9a-f]{64}", case["fingerprints"][name]):
            raise ValueError(f"Invalid input fingerprint: {name}")
    if case["fingerprints"]["dataset"] != dataset["dataset_sha256"]:
        raise ValueError("Prepared dataset fingerprint differs")
    if not re.fullmatch(r"[0-9a-f]{64}", record["output_sha256"]):
        raise ValueError("Invalid output fingerprint")

    for name in (*TIMINGS, "worker_peak_rss_mib"):
        require_number(record[name], name, positive=True)
    if not math.isclose(
        record["total_seconds"],
        record["fit_seconds"] + record["transform_seconds"],
        rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError("Total time does not equal fit plus first transform")
    for name in ("rmse", "mae"):
        require_number(record[name], name)
    scored = sum(case["query_mask"]["missing_per_feature"])
    values = np.asarray(record["imputed_values"], dtype=np.float64)
    if (
        scored <= 0 or record["scored_cells"] != scored
        or values.shape != (scored,) or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid hidden-cell comparison values")
    feature_quality = record["feature_quality"]
    if [item["feature"] for item in feature_quality] != dataset["feature_names"]:
        raise ValueError("Unexpected per-feature quality order")
    for column, item in enumerate(feature_quality):
        count = case["query_mask"]["missing_per_feature"][column]
        if item["scored_cells"] != count:
            raise ValueError("Per-feature scoring count differs")
        for name in (
            "rmse_standardized", "mae_standardized",
            "rmse_original_units", "mae_original_units",
        ):
            if count:
                require_number(item[name], name)
            elif item[name] is not None:
                raise ValueError("Unscored feature must have null error metrics")


def distribution(values):
    return (
        {"count": len(values), "median": median(values),
         "min": min(values), "max": max(values)}
        if values else {"count": 0, "median": None, "min": None, "max": None}
    )


def require_single_configuration(records):
    """Reject pooled workloads while allowing multiple seeds and repeats."""
    identities = {
        tuple(
            (name, value) for name, value in match_configuration(row).items()
            if name not in ("seed", "repeat")
        )
        for row in records
    }
    if len(identities) > 1:
        raise ValueError("Cannot pool multiple benchmark configurations")


def summarize(records):
    successful = [
        row for row in records
        if row["status"] == "ok" and row.get("checks_passed") is True
    ]
    require_single_configuration(successful)
    summaries = []
    for variant in VARIANTS:
        rows = [row for row in successful if row["variant"] == variant]
        # Repeat consistency is validated before records become eligible here.
        seed_rows = [row for row in rows if row["repeat"] == 1]
        summaries.append({
            "variant": variant,
            "expected_records": len(SEEDS) * REPEATS,
            "successful_records": len(rows),
            "complete": len(rows) == len(SEEDS) * REPEATS,
            "timing_and_memory": {
                name: distribution([row[name] for row in rows])
                for name in (*TIMINGS, "worker_peak_rss_mib")
            },
            "quality_seed_count": len(seed_rows),
            "quality_record_indices": [row["record_index"] for row in seed_rows],
            "quality": {
                name: distribution([row[name] for row in seed_rows])
                for name in ("rmse", "mae")
            },
        })
    return summaries


def compare_records(records):
    lookup = {}
    for row in records:
        if row["status"] != "ok" or row.get("checks_passed") is not True:
            continue
        key = tuple(match_configuration(row).items())
        identity = (key, row["variant"])
        if identity in lookup:
            raise ValueError("Duplicate successful configuration/variant record")
        lookup[identity] = row
    comparisons = []
    for numerator, denominator in (
        ("previous", "current"), ("knn", "current"), ("knn", "previous"),
    ):
        pairs = []
        for (key, variant), left in lookup.items():
            if variant != numerator or (key, denominator) not in lookup:
                continue
            right = lookup[(key, denominator)]
            if left["case"] != right["case"]:
                raise ValueError("Matched configuration has different prepared inputs")
            left_values = np.asarray(left["imputed_values"], dtype=np.float64)
            right_values = np.asarray(right["imputed_values"], dtype=np.float64)
            if left_values.shape != right_values.shape:
                raise ValueError("Matched outputs have different scoring shapes")
            ratios = {}
            for name in TIMINGS:
                require_number(left[name], name, positive=True)
                require_number(right[name], name, positive=True)
                ratio = left[name] / right[name]
                require_number(ratio, "timing ratio", positive=True)
                ratios[name] = ratio
            pairs.append({
                "match": dict(key),
                "numerator_record_index": left["record_index"],
                "denominator_record_index": right["record_index"],
                "timing_ratios": ratios,
                "output_sha256_equal": left["output_sha256"] == right["output_sha256"],
                "max_abs_difference_on_hidden_entries": float(
                    np.max(np.abs(left_values - right_values))
                ),
            })
        require_single_configuration(pair["match"] for pair in pairs)
        comparisons.append({
            "numerator_variant": numerator,
            "denominator_variant": denominator,
            "expected_pairs": len(SEEDS) * REPEATS,
            "matched_pairs": len(pairs),
            "complete": len(pairs) == len(SEEDS) * REPEATS,
            "timing_ratios": {
                name: distribution([pair["timing_ratios"][name] for pair in pairs])
                for name in TIMINGS
            },
            "pairs": pairs,
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-python", type=Path, required=True)
    parser.add_argument("--previous-version", required=True)
    parser.add_argument("--current-version", required=True)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default=DATASET_ID)
    parser.add_argument("--dtype", choices=DTYPES, default="float64")
    parser.add_argument("--weights", choices=WEIGHTS, default="uniform")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.previous_version == args.current_version:
        parser.error("Current and previous versions must differ")
    if not args.previous_python.is_file():
        parser.error("previous-python must identify an existing interpreter")
    if min(args.timeout_seconds, args.budget_seconds) <= 0:
        parser.error("Timeout and budget must be positive")

    check_released_package(args.current_version)
    environment = metadata()
    data_home = args.data_home.absolute()
    # Preserve the validated source ZIP; workers only read the local cache.
    data, names, dataset = load_dataset(
        data_home, dataset_id=args.dataset, download_if_missing=True
    )
    del data, names
    configs = build_configs(
        args.previous_version, args.current_version, data_home,
        dataset_id=args.dataset, dtype=args.dtype, weights=args.weights,
    )
    interpreters = {
        "knn": sys.executable, "current": sys.executable,
        # Do not resolve the venv interpreter's executable symlink.
        "previous": str(args.previous_python.absolute()),
    }
    results = {
        "schema_version": 1,
        "benchmark": "released_real_data",
        "metadata": environment,
        "dataset": dataset,
        "parameters": {
            "previous_version": args.previous_version,
            "current_version": args.current_version,
            "interpreters": interpreters,
            "dataset_id": args.dataset, "dtype": args.dtype,
            "weights": args.weights,
            "seeds": list(SEEDS), "repeats": REPEATS,
            "expected_workers": len(configs),
            "worker_timeout_seconds": args.timeout_seconds,
            "worker_run_budget_seconds": args.budget_seconds,
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
        },
        "planned_configs": configs,
        "notes": [
            f"One fixed {dataset['dataset']} {args.dtype} {args.weights}-weighted "
            "configuration per JSON.",
            "Different datasets, dtypes and weights are never pooled in summaries or paired ratios.",
            "A missing record-level weights field means uniform for archive compatibility.",
            "Installed releases and KNN run in fresh sequential workers; order rotates.",
            "KNN uses the current-release environment; dependencies are shared.",
            "All variants and repeats for a seed must have identical prepared cases.",
            f"The {DATASET_DEFAULTS[args.dataset]['mar_driver']} feature stays observed; "
            "nominal overall missingness is 10%.",
            "Scaling uses observed training values only; scoring truth is float64.",
            "Fit and first transform are consecutive, without GC or RSS sampling between.",
            "Timing excludes preparation, warmup, validation and worker startup.",
            "Timing and whole-worker RSS use nine workers per variant (3 seeds x 3 repeats).",
            "RSS includes preparation, warmup and validation; it is not a phase peak.",
            "RSS is sampled before worker JSON serialization; working_memory is not a RAM cap.",
            "RMSE/MAE measure held-out ground-truth error, not differences between methods.",
            "Quality summaries use repeat 1 for each seed after repeat-consistency checks.",
            "Ratios are numerator/denominator times for matched configuration, seed and repeat.",
            "Reported speedups are medians of paired ratios, not ratios of median times.",
            "Values above one favor the denominator; fit, first transform and total stay separate.",
            "Output hashes and hidden-entry differences describe only the measured inputs.",
            "No speed or cross-method output-equality threshold is required.",
            "Failures and unrun cases remain in records and cause a nonzero exit.",
            "Full-precision JSON values and pair record indices support regeneration.",
        ],
        "records": [], "summaries": [], "comparisons": [], "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.write("\n")

    case_references = {}
    repeat_references = {}
    save()
    started = time.perf_counter()
    for record_index, config in enumerate(configs):
        remaining = args.budget_seconds - (time.perf_counter() - started)
        payload = (
            run_worker(
                interpreters[config["variant"]], config,
                min(args.timeout_seconds, remaining),
            )
            if remaining > 0 else {"status": "not_run_budget"}
        )
        values = payload.pop("_values", None)
        record = {**config, **payload, "record_index": record_index}
        if record["status"] == "ok":
            record["imputed_values"] = values
            payload["imputed_values"] = values
            try:
                # Validate worker-owned fields before configuration defaults
                # can conceal missing or conflicting worker output.
                validate_record(payload, config, environment, dataset)
                if any(
                    payload[name] != value
                    for name, value in config.items() if name in payload
                ):
                    raise ValueError("Worker output conflicts with its configuration")
                seed = config["seed"]
                if seed in case_references and record["case"] != case_references[seed]:
                    raise ValueError("Prepared inputs differ between workers")
                repeat_key = (seed, config["variant"], record_weights(config))
                signature = {
                    name: record[name] for name in (
                        "output_sha256", "imputed_values", "rmse", "mae", "feature_quality"
                    )
                }
                if (
                    repeat_key in repeat_references
                    and signature != repeat_references[repeat_key]
                ):
                    raise ValueError("Same-method outputs or quality differ across repeats")
                case_references.setdefault(seed, record["case"])
                repeat_references.setdefault(repeat_key, signature)
            except (KeyError, TypeError, ValueError) as error:
                record.update(status="validation_error", checks_passed=False, error=str(error))
        else:
            record["checks_passed"] = False
        results["records"].append(record)
        print(
            f"{record_index + 1}/{len(configs)} seed={config['seed']} "
            f"repeat={config['repeat']} {config['variant']}: {record['status']}",
            flush=True,
        )
        if record.get("error"):
            print(record["error"], flush=True)
        save()

    results["summaries"] = summarize(results["records"])
    results["comparisons"] = compare_records(results["records"])
    results["complete"] = all(
        record["status"] == "ok" and record["checks_passed"] is True
        for record in results["records"]
    ) and all(item["complete"] for item in results["comparisons"])
    results["elapsed_worker_loop_seconds"] = time.perf_counter() - started
    save()
    print(f"Results: {args.output}", flush=True)
    return int(not results["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
