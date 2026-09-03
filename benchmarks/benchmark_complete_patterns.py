"""Measure complete-donor timings without changing the imputation algorithm."""

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from sklearn.impute import KNNImputer
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
from benchmarks.benchmark_imputers import (
    N_FEATURES,
    N_NEIGHBORS,
    make_correlated_data,
    make_missing_mask,
)

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("fixed", "eight", "random")
METHODS = ("FaissImputer", "KNNImputer")
RTOL = 1e-6
ATOL = 1e-6


def make_model(name):
    if name == "FaissImputer":
        return FaissImputer(
            n_neighbors=N_NEIGHBORS,
            metric="l2",
            strategy="mean",
            index_factory="Flat",
            donor_policy="complete",
        )
    if name == "KNNImputer":
        return KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights="uniform",
            metric="nan_euclidean",
        )
    raise ValueError(f"Unknown method: {name}")


def check_output(output, query, missing):
    assert output.shape == query.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert not np.shares_memory(output, query)
    np.testing.assert_array_equal(output[~missing], query[~missing])


def profile_transform(model, query):
    """Diagnostic timing only; proxy overhead is part of the residual."""
    seconds = {name: 0.0 for name in ("factory", "train", "add", "search")}
    calls = {name: 0 for name in seconds}
    original_factory = faiss.index_factory

    def measured(name, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            seconds[name] += time.perf_counter() - started
            calls[name] += 1

    class TimedIndex:
        def __init__(self, index):
            self.index = index

        def train(self, *args, **kwargs):
            return measured("train", self.index.train, *args, **kwargs)

        def add(self, *args, **kwargs):
            return measured("add", self.index.add, *args, **kwargs)

        def search(self, *args, **kwargs):
            return measured("search", self.index.search, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.index, name)

    def factory(*args, **kwargs):
        index = measured("factory", original_factory, *args, **kwargs)
        return TimedIndex(index)

    with patch.object(faiss, "index_factory", new=factory):
        started = time.perf_counter()
        output = model.transform(query)
        total = time.perf_counter() - started

    residual = total - sum(seconds.values())
    assert residual >= 0.0
    return output, {
        "transform_seconds": total,
        "index_seconds": seconds,
        "non_index_residual_seconds": residual,
        "index_calls": calls,
    }


def summary(samples):
    result = {}
    for method in METHODS:
        rows = [row for row in samples if row["method"] == method]
        result[method] = {}
        for field in (
            "fit_seconds",
            "first_transform_seconds",
            "repeated_transform_seconds",
            "fit_plus_first_transform_seconds",
        ):
            values = [row[field] for row in rows]
            result[method][field] = {
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
    return result


def run_case(train, query, seed, pattern, repeats):
    train_before = train.copy()
    query_before = query.copy()
    missing = np.isnan(query)
    assert np.isfinite(train).all()
    assert missing.any(axis=1).all()
    assert (~missing).any(axis=1).all()

    case = {
        "seed": seed,
        "pattern": pattern,
        "n_train": train.shape[0],
        "n_query": query.shape[0],
        "n_features": train.shape[1],
        "n_neighbors": N_NEIGHBORS,
        "unique_patterns": int(np.unique(missing, axis=0).shape[0]),
        "missing_rate": float(missing.mean()),
        "samples": [],
        "profiles": [],
        "quality_checks": [],
    }

    for repeat in range(repeats):
        outputs = {}
        order = METHODS if repeat % 2 == 0 else METHODS[::-1]
        for method in order:
            model = make_model(method)
            gc.collect()
            started = time.perf_counter()
            model.fit(train)
            fitted = time.perf_counter()
            first = model.transform(query)
            transformed = time.perf_counter()
            repeated = model.transform(query)
            finished = time.perf_counter()

            case["samples"].append({
                "repeat": repeat + 1,
                "method": method,
                "fit_seconds": fitted - started,
                "first_transform_seconds": transformed - fitted,
                "repeated_transform_seconds": finished - transformed,
                "fit_plus_first_transform_seconds": transformed - started,
            })
            check_output(first, query, missing)
            check_output(repeated, query, missing)
            np.testing.assert_array_equal(first, repeated)
            np.testing.assert_array_equal(train, train_before)
            np.testing.assert_array_equal(query, query_before)
            outputs[method] = first
            del model

        expected = outputs["FaissImputer"]
        other = outputs["KNNImputer"]
        np.testing.assert_allclose(
            expected[missing], other[missing], rtol=RTOL, atol=ATOL
        )
        max_difference = float(np.max(
            np.abs(expected[missing].astype(np.float64) - other[missing])
        ))

        # A fresh fitted estimator keeps first-call profiling independent.
        model = make_model("FaissImputer").fit(train)
        gc.collect()
        profile = {"repeat": repeat + 1}
        for stage in ("first", "repeated"):
            output, measurements = profile_transform(model, query)
            check_output(output, query, missing)
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(train, train_before)
            np.testing.assert_array_equal(query, query_before)
            profile[stage] = measurements
        del model

        case["profiles"].append(profile)
        case["quality_checks"].append({
            "repeat": repeat + 1,
            "knn_max_abs_difference": max_difference,
            "knn_allclose": True,
            "profile_matches_uninstrumented": True,
            "repeated_matches_first": True,
            "inputs_preserved": True,
        })

    case["summary"] = summary(case["samples"])
    return case


def current_commit():
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def warm_up():
    train, truth = make_correlated_data(7, 64, 8)
    missing = make_missing_mask(8, 8, N_FEATURES, "fixed")
    query = truth.copy()
    query[missing] = np.nan
    for method in METHODS:
        make_model(method).fit(train).transform(query)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20000])
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs" / "complete_patterns.json",
    )
    args = parser.parse_args()
    if min(args.train_sizes) < N_NEIGHBORS:
        parser.error(f"train sizes must be at least {N_NEIGHBORS}")
    if args.queries < 8 or args.repeats < 1 or min(args.seeds) < 0:
        parser.error("queries >= 8, repeats >= 1, and seeds >= 0 are required")
    if len(set(args.train_sizes)) != len(args.train_sizes):
        parser.error("train sizes must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not contain duplicates")

    cases = []
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        warm_up()
        for seed in args.seeds:
            train_max, truth = make_correlated_data(
                seed, max(args.train_sizes), args.queries
            )
            for pattern in PATTERNS:
                missing = make_missing_mask(
                    seed + 10000, args.queries, N_FEATURES, pattern
                )
                query = truth.copy()
                query[missing] = np.nan
                for size in args.train_sizes:
                    print(
                        f"START seed={seed} train={size} pattern={pattern}",
                        flush=True,
                    )
                    case = run_case(
                        train_max[:size], query, seed, pattern, args.repeats
                    )
                    cases.append(case)
                    print(
                        f"DONE unique_patterns={case['unique_patterns']}",
                        flush=True,
                    )
                    for method in METHODS:
                        values = case["summary"][method]
                        first = values["first_transform_seconds"]["median"]
                        repeated = values["repeated_transform_seconds"]["median"]
                        print(
                            f"  {method}: first={first * 1000:.3f}ms, "
                            f"repeated={repeated * 1000:.3f}ms",
                            flush=True,
                        )

    results = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "faiss_imputer": version("faiss-imputer"),
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
            "faiss": getattr(faiss, "__version__", "unknown"),
            "native_threads": 1,
        },
        "parameters": {
            "train_sizes": args.train_sizes,
            "queries": args.queries,
            "seeds": args.seeds,
            "repeats": args.repeats,
            "patterns": list(PATTERNS),
            "rtol": RTOL,
            "atol": ATOL,
            "donor_policy": "complete",
            "metric": "l2",
            "index_factory": "Flat",
            "strategy": "mean",
        },
        "notes": [
            "Synthetic complete donors; exactly four missing query features.",
            "Scaling uses the complete maximum-size training set per seed.",
            "This is not an end-to-end preprocessing benchmark.",
            "Native libraries are warmed up with throwaway estimators.",
            "Each uninstrumented sample fits a fresh estimator.",
            "Repeated transform uses that same estimator and query.",
            "Method order alternates between repeats.",
            "Use uninstrumented samples for performance comparisons.",
            "Profiles include proxy overhead and are diagnostic only.",
            "Non-index residual includes validation, copies, grouping, "
            "aggregation, Python overhead, and other unmeasured work.",
            "Agreement checks apply only to these tested inputs.",
            "Peak memory is not measured; measure it before adopting a cache.",
            "No historical wall-clock value is used as a pass/fail threshold.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
