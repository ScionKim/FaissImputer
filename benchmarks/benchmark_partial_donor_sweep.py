"""Held-out synthetic sweep; benchmark-only, not a product implementation."""
import gc
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np
import sklearn
from sklearn.impute import KNNImputer
from threadpoolctl import threadpool_limits

from benchmarks.benchmark_imputers import (
    imputation_errors,
    make_correlated_data,
    make_missing_mask,
)
from benchmarks.benchmark_partial_donors import MatrixNaNBenchmark

SEEDS = [101, 202, 303, 404, 505]
REPEATS = 3
TRAIN_SIZES = [1_000, 5_000, 20_000]
TRAIN_MISSING_RATES = [0.1, 0.3, 0.5]
N_TEST = 500
N_FEATURES = 20
N_NEIGHBORS = 5


def make_models():
    return {
        "KNNImputer": KNNImputer(n_neighbors=N_NEIGHBORS),
        "MatrixNaNBenchmark": MatrixNaNBenchmark(
            n_neighbors=N_NEIGHBORS, donor_policy="available"
        ),
    }


def run_sweep(results):
    warm_train, warm_query = make_correlated_data(7, 128, 16, N_FEATURES)
    warm_train[::3, 0] = np.nan
    warm_query[:, 1] = np.nan
    for model in make_models().values():
        model.fit(warm_train).transform(warm_query)

    for seed_index, seed in enumerate(SEEDS):
        base, truth = make_correlated_data(
            50_000 + seed, max(TRAIN_SIZES), N_TEST, N_FEATURES
        )
        draws = np.random.default_rng(60_000 + seed).random(base.shape)
        mask = make_missing_mask(70_000 + seed, N_TEST, N_FEATURES, "random")
        original_query = truth.copy()
        original_query[mask] = np.nan

        for size_index, n_train in enumerate(TRAIN_SIZES):
            for rate_index, rate in enumerate(TRAIN_MISSING_RATES):
                original_train = base[:n_train].copy()
                original_train[draws[:n_train] < rate] = np.nan
                case = {"seed": seed, "n_train": n_train, "train_missing_rate": rate}
                results["cases"].append({
                    **case,
                    "actual_train_missing_rate": float(np.isnan(original_train).mean()),
                    "complete_donors": int((~np.isnan(original_train).any(axis=1)).sum()),
                })
                print("CASE", case, flush=True)

                for repeat in range(REPEATS):
                    outputs = {}
                    models = list(make_models().items())
                    if (seed_index + size_index + rate_index + repeat) % 2:
                        models.reverse()

                    for name, model in models:
                        train = original_train.copy()
                        query = original_query.copy()
                        gc.collect()
                        started = time.perf_counter()
                        model.fit(train)
                        fitted = time.perf_counter()
                        output = model.transform(query)
                        finished = time.perf_counter()

                        assert output.shape == truth.shape
                        assert np.isfinite(output).all()
                        np.testing.assert_array_equal(output[~mask], original_query[~mask])
                        np.testing.assert_array_equal(train, original_train)
                        np.testing.assert_array_equal(query, original_query)
                        rmse, mae = imputation_errors(truth, output, mask)
                        results["trials"].append({
                            **case,
                            "repeat": repeat + 1,
                            "method": name,
                            "fit_seconds": fitted - started,
                            "transform_seconds": finished - fitted,
                            "total_seconds": finished - started,
                            "rmse": rmse,
                            "mae": mae,
                            "observed_values_unchanged": True,
                            "inputs_unchanged": True,
                        })
                        outputs[name] = output
                        print(
                            f"  {repeat + 1} {name}: total={finished - started:.4f}s",
                            flush=True,
                        )

                    expected = outputs["KNNImputer"]
                    actual = outputs["MatrixNaNBenchmark"]
                    matches = bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-6))
                    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                    bad = ~np.isclose(actual, expected, rtol=1e-6, atol=1e-6)
                    results["quality"].append({
                        **case,
                        "repeat": repeat + 1,
                        "allclose_passed": matches,
                        "max_abs_difference": float(difference.max()),
                        "mismatched_values": int(bad.sum()),
                        "examples": [
                            {"row": int(i), "column": int(j),
                             "knn": float(expected[i, j]), "matrix": float(actual[i, j])}
                            for i, j in np.argwhere(bad)[:5]
                        ],
                    })
                    if not matches:
                        print("  QUALITY MISMATCH:", float(difference.max()), flush=True)


def main():
    results = {
        "metadata": {
            "commit": os.environ.get("GITHUB_SHA", "local"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "faiss_imputer": version("faiss-imputer"),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "faiss": faiss.__version__,
            "threads": 1,
        },
        "parameters": {
            "seeds": SEEDS,
            "repeats": REPEATS,
            "train_sizes": TRAIN_SIZES,
            "train_missing_rates": TRAIN_MISSING_RATES,
            "n_test": N_TEST,
            "n_features": N_FEATURES,
            "n_neighbors": N_NEIGHBORS,
            "query_missing_rate": 0.2,
            "query_pattern": "random, exactly four missing columns per row",
            "rtol": 1e-6,
            "atol": 1e-6,
            "data_design": (
                "Nested training prefixes and masks; identical held-out queries "
                "within each seed. Standardized from complete maximum-training "
                "truth before masking; not an end-to-end preprocessing benchmark."
            ),
        },
        "completed": False,
        "cases": [],
        "trials": [],
        "quality": [],
    }
    path = Path(__file__).resolve().parents[1] / "benchmark_outputs" / "partial_donor_sweep.json"
    faiss.omp_set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            run_sweep(results)
        results["completed"] = True
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        print("Saved:", path, flush=True)

    if not all(row["allclose_passed"] for row in results["quality"]):
        raise AssertionError("Output differences found; inspect partial_donor_sweep.json")


if __name__ == "__main__":
    main()
