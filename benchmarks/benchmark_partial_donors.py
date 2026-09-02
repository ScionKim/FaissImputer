from __future__ import annotations

import argparse
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
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer
from benchmarks.benchmark_imputers import (
    MISSING_RATE,
    N_FEATURES,
    N_NEIGHBORS,
    TRAIN_MISSING_RATE,
    imputation_errors,
    make_correlated_data,
    make_imputers as make_original_imputers,
    make_missing_mask,
    unique_missing_patterns,
)


def make_imputers():
    imputers = make_original_imputers()
    imputers["FaissImputer[complete]"] = imputers.pop("FaissImputer")
    imputers["FaissImputer[available]"] = FaissImputer(
        n_neighbors=N_NEIGHBORS,
        donor_policy="available",
    )
    return imputers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    output = (
        Path(__file__).resolve().parents[1]
        / "benchmark_outputs"
        / "partial_donors.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "commit": os.environ.get("GITHUB_SHA", "unrecorded"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "faiss_imputer": version("faiss-imputer"),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "faiss": faiss.__version__,
            "threads": 1,
        },
        "parameters": {
            "seeds": args.seeds,
            "repeats": args.repeats,
            "n_train": 5000,
            "n_test": 500,
            "n_features": N_FEATURES,
            "n_neighbors": N_NEIGHBORS,
            "train_missing_rate": TRAIN_MISSING_RATE,
            "test_missing_rate": MISSING_RATE,
            "query_pattern": "random",
        },
        "cases": [],
        "trials": [],
    }

    def save():
        output.write_text(
            json.dumps(results, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    save()
    faiss.omp_set_num_threads(1)

    with threadpool_limits(limits=1):
        for seed_index, seed in enumerate(args.seeds):
            train_truth, test_truth = make_correlated_data(
                seed=50_000 + seed,
                n_train=5000,
                n_test=500,
            )
            rng = np.random.default_rng(60_000 + seed)
            train_mask = rng.random(train_truth.shape) < TRAIN_MISSING_RATE
            train = train_truth.copy()
            train[train_mask] = np.nan

            test_mask = make_missing_mask(
                seed=70_000 + seed,
                n_rows=500,
                n_features=N_FEATURES,
                pattern="random",
            )
            query = test_truth.copy()
            query[test_mask] = np.nan

            case = {
                "seed": seed,
                "complete_donors": int((~train_mask.any(axis=1)).sum()),
                "train_mask_patterns": unique_missing_patterns(train_mask),
                "query_mask_patterns": unique_missing_patterns(test_mask),
            }
            results["cases"].append(case)
            save()
            print(f"CASE {case}", flush=True)

            if case["complete_donors"] < N_NEIGHBORS:
                raise ValueError("Not enough donors for complete mode")

            if seed_index == 0:
                print("Small untimed warmup", flush=True)
                warm_train = train_truth[:128].copy()
                warm_train[::2, -1] = np.nan
                for imputer in make_imputers().values():
                    imputer.fit(warm_train).transform(query[:8])

            names = list(make_imputers())
            for repeat in range(args.repeats):
                offset = (seed_index + repeat) % len(names)
                order = names[offset:] + names[:offset]

                for name in order:
                    print(
                        f"START seed={seed} repeat={repeat + 1} {name}",
                        flush=True,
                    )
                    imputer = make_imputers()[name]
                    gc.collect()

                    start = time.perf_counter()
                    imputer.fit(train)
                    fitted = time.perf_counter()
                    prediction = imputer.transform(query)
                    finished = time.perf_counter()

                    if prediction.shape != test_truth.shape:
                        raise AssertionError(f"{name}: unexpected shape")
                    if not np.isfinite(prediction).all():
                        raise AssertionError(f"{name}: non-finite output")
                    np.testing.assert_array_equal(
                        prediction[~test_mask], query[~test_mask]
                    )

                    rmse, mae = imputation_errors(
                        test_truth, prediction, test_mask
                    )
                    row = {
                        "seed": seed,
                        "repeat": repeat + 1,
                        "method": name,
                        "rmse": rmse,
                        "mae": mae,
                        "fit_seconds": fitted - start,
                        "transform_seconds": finished - fitted,
                        "total_seconds": finished - start,
                    }
                    results["trials"].append(row)
                    save()
                    print(
                        f"DONE {name}: RMSE={rmse:.6f}, "
                        f"fit={row['fit_seconds']:.3f}s, "
                        f"transform={row['transform_seconds']:.3f}s",
                        flush=True,
                    )

    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
