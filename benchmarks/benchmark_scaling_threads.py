"""Synthetic scaling pilot; one fresh, sequential worker per measurement."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median

import faiss
import faiss_imputer
import numpy as np
from sklearn import config_context
from sklearn.impute import KNNImputer
from threadpoolctl import threadpool_info, threadpool_limits

from faiss_imputer import FaissImputer
from benchmarks.benchmark_imputers import make_missing_mask

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("KNNImputer", "FaissImputer[complete]", "FaissImputer[available]")
FEATURES = 20
NEIGHBORS = 5
TRAIN_MISSING_RATE = 0.10
WORKING_MEMORY_MIB = 256


def digest(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def check_released_package(expected_version):
    """When requested, require an installed distribution outside the checkout."""
    if expected_version is None:
        return

    installed_version = version("faiss-imputer")
    if installed_version != expected_version:
        raise RuntimeError(
            f"Expected faiss-imputer {expected_version}, got {installed_version}"
        )

    module_path = Path(faiss_imputer.__file__).resolve()
    install_roots = {
        Path(sysconfig.get_paths()[key]).resolve()
        for key in ("purelib", "platlib")
    }
    if not any(module_path.is_relative_to(root) for root in install_roots):
        raise RuntimeError(
            f"FaissImputer was not imported from the installed package: {module_path}"
        )


def make_data(size, queries, seed, pattern, training_policy="partial"):
    """Fixed-size blocks and independent streams preserve training prefixes."""
    if training_policy not in ("partial", "complete", "available"):
        raise ValueError(f"Unknown training policy: {training_policy}")

    loadings = np.random.default_rng([seed, 0]).normal(size=(5, FEATURES))
    scale = np.sqrt(np.sum(loadings * loadings, axis=0) + 0.15 ** 2)

    def draw(rows, latent_tag, noise_tag, missing_rate):
        latent_rng = np.random.default_rng([seed, latent_tag])
        noise_rng = np.random.default_rng([seed, noise_tag])
        mask_rng = np.random.default_rng([seed, 5])
        output = np.empty((rows, FEATURES), dtype=np.float32)
        block_size = 16384
        for start in range(0, rows, block_size):
            latent = latent_rng.normal(size=(block_size, 5))
            noise = noise_rng.normal(size=(block_size, FEATURES))
            block = ((latent @ loadings + 0.15 * noise) / scale).astype(np.float32)
            if missing_rate:
                block[mask_rng.random(block.shape) < missing_rate] = np.nan
            count = min(block_size, rows - start)
            output[start:start + count] = block[:count]
        return output

    train_missing_rate = (
        0.0 if training_policy == "complete" else TRAIN_MISSING_RATE
    )
    with threadpool_limits(limits=1):
        train = draw(size, 1, 2, train_missing_rate)
        truth = draw(queries, 3, 4, 0.0)
    missing = make_missing_mask(seed + 10000, queries, FEATURES, pattern)
    query = truth.copy()
    query[missing] = np.nan
    return train, query, truth, missing


def make_model(method):
    if method == "KNNImputer":
        return KNNImputer(n_neighbors=NEIGHBORS, weights="uniform")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    policy = "complete" if method == "FaissImputer[complete]" else "available"
    return FaissImputer(
        n_neighbors=NEIGHBORS, metric="l2", strategy="mean",
        index_factory="Flat", donor_policy=policy,
    )


def peak_rss_mib():
    if sys.platform != "linux":
        return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def worker(config):
    check_released_package(config.get("expected_version"))
    train, query, truth, missing = make_data(
        config["size"], config["queries"], config["seed"], config["pattern"],
        config.get("training_policy", "partial"),
    )
    fingerprints = {
        "query": digest(query),
        "truth": digest(truth),
        "prefixes": {
            str(size): digest(train[:size])
            for size in config["prefix_sizes"] if size <= len(train)
        },
    }
    train_hash = digest(train)
    query_before = query.copy()
    complete_donors = int((~np.isnan(train).any(axis=1)).sum())
    details = {
        "complete_donors": complete_donors,
        "train_missing_rate": float(np.isnan(train).mean()),
        "query_missing_rate": float(missing.mean()),
        "query_patterns": int(np.unique(missing, axis=0).shape[0]),
        "fingerprints": fingerprints,
    }
    if config["method"] == "FaissImputer[complete]" and complete_donors < NEIGHBORS:
        return {
            "status": "not_applicable",
            "worker_peak_rss_mib": peak_rss_mib(),
            **details,
        }

    def check_output(output):
        assert output.shape == query_before.shape
        assert np.isfinite(output).all()
        assert not np.shares_memory(output, query)
        np.testing.assert_array_equal(output[~missing], query_before[~missing])
        np.testing.assert_array_equal(query, query_before)
        assert digest(train) == train_hash
        if config["method"].startswith("FaissImputer"):
            assert output.dtype == np.float32

    threads = config["threads"]
    faiss.omp_set_num_threads(threads)
    with threadpool_limits(limits=threads), config_context(
        working_memory=WORKING_MEMORY_MIB
    ):
        warm_train = np.random.default_rng(7).normal(
            size=(32, FEATURES)
        ).astype(np.float32)
        warm_query = warm_train[:8].copy()
        warm_query[:, :4] = np.nan
        make_model(config["method"]).fit(warm_train).transform(warm_query)

        pools = [
            {key: pool.get(key) for key in (
                "internal_api", "prefix", "num_threads", "version", "architecture"
            )}
            for pool in threadpool_info()
        ]
        model = make_model(config["method"])
        gc.collect()

        started = time.perf_counter()
        model.fit(train)
        fitted = time.perf_counter()
        output = model.transform(query)
        finished = time.perf_counter()

        check_output(output)
        first_output = output.copy()
        repeated_times = []
        for _ in range(config.get("repeated_transforms", 0)):
            repeat_started = time.perf_counter()
            repeated_output = model.transform(query)
            repeated_times.append(time.perf_counter() - repeat_started)
            check_output(repeated_output)
            np.testing.assert_array_equal(repeated_output, first_output)

        actual_faiss_threads = int(faiss.omp_get_max_threads())

    values = first_output[missing].astype(np.float64)
    errors = values - truth[missing].astype(np.float64)
    return {
        "status": "ok",
        **details,
        "fit_seconds": fitted - started,
        "transform_seconds": finished - fitted,
        "total_seconds": finished - started,
        "repeated_transform_seconds": repeated_times,
        "repeated_transform_median_seconds": (
            median(repeated_times) if repeated_times else None
        ),
        "worker_peak_rss_mib": peak_rss_mib(),
        "threadpools": pools,
        "faiss_omp_threads": actual_faiss_threads,
        "scored_cells": int(missing.sum()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "output_dtype": str(first_output.dtype),
        "output_sha256": digest(first_output),
        "checks_passed": True,
        "_values": values.tolist(),
    }


def run_worker(config, timeout):
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        env[name] = str(config["threads"])
    command = [
        sys.executable, "-u", "-m", "benchmarks.benchmark_scaling_threads",
        "--worker", json.dumps(config),
    ]
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout", "timeout_seconds": timeout,
            "worker_peak_rss_mib": None,
        }
    if process.returncode:
        return {
            "status": "process_error", "returncode": process.returncode,
            "error": process.stderr[-2000:], "worker_peak_rss_mib": None,
        }
    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "invalid_worker_output", "worker_peak_rss_mib": None}
    record["worker_wall_seconds"] = time.perf_counter() - started
    if process.stderr.strip():
        record["warnings"] = process.stderr[-2000:]
    return record


def metadata():
    try:
        commit = os.environ.get("GITHUB_SHA") or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None
        ),
        "faiss_imputer": version("faiss-imputer"),
        "faiss_imputer_module": str(Path(faiss_imputer.__file__).resolve()),
        "numpy": np.__version__,
        "scikit_learn": version("scikit-learn"),
        "faiss": getattr(faiss, "__version__", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument(
        "--patterns", nargs="+", choices=["fixed", "random"],
        default=["fixed", "random"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeated-transforms", type=int, default=0)
    parser.add_argument("--policy-matched-data", action="store_true")
    parser.add_argument("--expected-version")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "benchmark_outputs/scaling_threads.json",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        try:
            result = worker(json.loads(args.worker))
        except MemoryError:
            result = {
                "status": "memory_error", "worker_peak_rss_mib": peak_rss_mib()
            }
        except Exception as error:
            result = {
                "status": "error", "error": f"{type(error).__name__}: {error}",
                "worker_peak_rss_mib": peak_rss_mib(),
            }
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0

    if min(args.train_sizes) < NEIGHBORS or min(args.seeds) < 0:
        parser.error("train sizes must be >= 5 and seeds must be nonnegative")
    if min(args.queries, args.repeats, args.timeout_seconds, args.budget_seconds) < 1:
        parser.error("queries, repeats, timeout and budget must be positive")
    if args.repeated_transforms < 0:
        parser.error("repeated-transforms must be nonnegative")
    if 1 not in args.threads or any(t not in (1, 2, 4) for t in args.threads):
        parser.error("threads must include 1 and use only 1, 2, or 4")
    for name in ("train_sizes", "seeds", "threads", "patterns"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    args.threads.sort()
    check_released_package(args.expected_version)

    cases = (
        [
            ("complete", ("KNNImputer", "FaissImputer[complete]")),
            ("available", ("KNNImputer", "FaissImputer[available]")),
        ]
        if args.policy_matched_data else [("partial", METHODS)]
    )
    results = {
        "metadata": metadata(),
        "parameters": {
            "train_sizes": args.train_sizes,
            "threads": args.threads,
            "queries": args.queries,
            "features": FEATURES,
            "n_neighbors": NEIGHBORS,
            "seeds": args.seeds,
            "patterns": args.patterns,
            "repeats": args.repeats,
            "repeated_transforms": args.repeated_transforms,
            "policy_matched_data": args.policy_matched_data,
            "training_cases": [
                {
                    "training_policy": policy,
                    "methods": methods,
                    "target_missing_rate": (
                        0.0 if policy == "complete" else TRAIN_MISSING_RATE
                    ),
                }
                for policy, methods in cases
            ],
            "expected_version": args.expected_version,
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "worker_timeout_seconds": args.timeout_seconds,
            "run_budget_seconds": args.budget_seconds,
        },
        "notes": [
            "Synthetic low-rank Gaussian data with theoretical variance normalization.",
            "Queries are held out; each query has four missing features.",
            "Within each training policy, inputs are shared across methods and repeats.",
            "Training prefixes are shared across sizes; data generation uses one thread.",
            "Policy-matched mode uses fully observed training for complete/KNN pairs "
            "and 10% MCAR training for available/KNN pairs.",
            "Without policy-matched mode, all three methods use 10% MCAR training.",
            "Fresh sequential subprocesses; small warmup before timed fit/transform.",
            "repeats counts fresh workers; repeated_transforms counts additional "
            "transform calls on the same fitted model.",
            "transform_seconds is the first transform after fit; total_seconds is "
            "fit plus that first transform and excludes additional transforms.",
            "Timings exclude data generation, validation and process startup.",
            "Memory is full-worker lifetime peak RSS, including imports, inputs, "
            "generation, warmup, validation and repeated transforms.",
            "Worker peak RSS is not retained fitted memory or a phase-specific peak.",
            "sklearn working_memory is a distance-chunk setting, not a process RAM cap.",
            "Timeout includes the entire worker; killed-worker peak memory is unavailable.",
            "Errors, timeouts and unrun cases remain in the results.",
            "Output comparisons use the first successful reference per configuration "
            "and stay within the same training policy.",
            "RMSE and MAE use hidden synthetic truth; differences from KNN measure "
            "output agreement, not imputation quality.",
            "No accuracy or speed threshold, or exact equality to KNN, is required.",
            "git_commit identifies the benchmark definition, not the installed wheel.",
        ],
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    save()
    inputs, references, outputs = {}, {}, []
    configs = []
    for size in args.train_sizes:
        for seed_index, seed in enumerate(args.seeds):
            for pattern in args.patterns:
                for training_policy, methods in cases:
                    for repeat in range(args.repeats):
                        start = repeat % len(args.threads)
                        thread_order = args.threads[start:] + args.threads[:start]
                        for threads in thread_order:
                            offset = (
                                repeat + seed_index + args.threads.index(threads)
                            ) % len(methods)
                            order = methods[offset:] + methods[:offset]
                            for method in order:
                                configs.append({
                                    "size": size,
                                    "queries": args.queries,
                                    "seed": seed,
                                    "pattern": pattern,
                                    "training_policy": training_policy,
                                    "threads": threads,
                                    "method": method,
                                    "repeat": repeat + 1,
                                    "prefix_sizes": args.train_sizes,
                                    "repeated_transforms": args.repeated_transforms,
                                    "expected_version": args.expected_version,
                                })

    results["parameters"]["expected_workers"] = len(configs)
    save()
    started = time.perf_counter()
    for config in configs:
        remaining = args.budget_seconds - (time.perf_counter() - started)
        print(
            f"START {config['training_policy']} {config['size']} "
            f"{config['pattern']} seed={config['seed']} "
            f"repeat={config['repeat']} threads={config['threads']} "
            f"{config['method']}",
            flush=True,
        )
        record = (
            run_worker(config, min(args.timeout_seconds, remaining))
            if remaining > 0 else {"status": "not_run_budget"}
        )
        values = record.pop("_values", None)
        record = {**config, **record}
        if record["status"] == "ok":
            marks = record["fingerprints"]
            policy, seed = config["training_policy"], config["seed"]
            checks = {
                (seed, "query", config["pattern"]): marks["query"],
                (seed, "truth"): marks["truth"],
                **{
                    (policy, seed, "train", n): h
                    for n, h in marks["prefixes"].items()
                },
            }
            if any(
                inputs.setdefault(key, value) != value
                for key, value in checks.items()
            ):
                record["status"] = "input_mismatch"
            else:
                key = (
                    policy, config["size"], seed, config["pattern"],
                    config["threads"], config["method"],
                )
                values = np.asarray(values, dtype=np.float64)
                references.setdefault(key, values)
                outputs.append((record, values))
        results["records"].append(record)
        print(
            f"DONE {record['status']} total={record.get('total_seconds')}",
            flush=True,
        )
        save()

    for record, values in outputs:
        base = (
            record["training_policy"], record["size"],
            record["seed"], record["pattern"],
        )
        comparisons = {
            "max_abs_difference_from_knn": (
                *base, record["threads"], "KNNImputer"
            ),
            "max_abs_difference_from_one_thread": (
                *base, 1, record["method"]
            ),
            "max_abs_difference_from_first_repeat": (
                *base, record["threads"], record["method"]
            ),
        }
        for name, key in comparisons.items():
            reference = references.get(key)
            record[name] = (
                None if reference is None
                else float(np.max(np.abs(values - reference)))
            )
    save()
    print(f"Saved: {args.output}", flush=True)
    return int(any(
        record["status"] not in ("ok", "not_applicable")
        for record in results["records"]
    ))


if __name__ == "__main__":
    raise SystemExit(main())
