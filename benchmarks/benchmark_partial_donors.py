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


class NativeNaNBenchmark(FaissImputer):
    """Benchmark-only prototype; not a production implementation."""

    def _clear_fitted_state(self):
        super()._clear_fitted_state()
        self.__dict__.pop("native_index_", None)

    def _fit_available(self, X):
        if self.metric != "l2" or self.index_factory != "Flat":
            raise ValueError(
                "donor_policy='available' requires "
                "metric='l2' and index_factory='Flat'"
            )
        if not hasattr(faiss, "METRIC_NaNEuclidean"):
            raise RuntimeError("This benchmark needs Faiss NaNEuclidean support.")

        observed = ~np.isnan(X)
        if not observed.any(axis=0).all():
            raise ValueError("X must not contain all-missing columns")

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.nanmedian(X, axis=0)
        self.donors_ = X[observed.any(axis=1)].copy()

        self.native_index_ = faiss.IndexFlat(
            X.shape[1], faiss.METRIC_NaNEuclidean
        )
        self.native_index_.add(
            np.ascontiguousarray(self.donors_, dtype=np.float32)
        )
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted, validate_data

        check_is_fitted(self, ["donor_policy_"])
        if self.donor_policy_ == "complete":
            return super().transform(X)
        X = validate_data(
            self,
            X,
            dtype=np.float32,
            ensure_all_finite="allow-nan",
            reset=False,
        )
        return self._transform_available(X)

    def _transform_available(self, X):
        result = X.copy()
        missing = np.isnan(X)
        result[missing] = np.broadcast_to(self.statistics_, X.shape)[missing]
        rows = np.flatnonzero(missing.any(axis=1) & ~missing.all(axis=1))
        n_donors = self.donors_.shape[0]
        k = min(self.n_neighbors, n_donors)
        batch_size = max(
            1, min(256, (16 * 1024 * 1024) // (12 * n_donors))
        )

        for start in range(0, rows.size, batch_size):
            batch_rows = rows[start:start + batch_size]
            batch_missing = missing[batch_rows]
            columns = np.flatnonzero(batch_missing.any(axis=0))
            queries = np.ascontiguousarray(X[batch_rows], dtype=np.float32)
            search_k = min(n_donors, max(16, 2 * k))

            while True:
                distances, ids = self.native_index_.search(
                    queries, int(search_k)
                )
                valid = (
                    (ids >= 0)
                    & (ids < n_donors)
                    & np.isfinite(distances)
                )
                safe_ids = np.where(valid, ids, 0)
                enough = np.ones(batch_rows.size, dtype=bool)
                for col in columns:
                    values = self.donors_[safe_ids, col]
                    usable = valid & ~np.isnan(values)
                    enough &= (
                        ~batch_missing[:, col]
                        | (usable.sum(axis=1) >= k)
                    )

                if enough.all() or search_k == n_donors:
                    break
                search_k = min(n_donors, 2 * search_k)

            for col in columns:
                values = self.donors_[safe_ids, col]
                usable = valid & ~np.isnan(values)
                chosen = usable & (np.cumsum(usable, axis=1) <= k)
                fill_rows = batch_missing[:, col] & chosen.any(axis=1)
                if not fill_rows.any():
                    continue
                selected = np.where(
                    chosen[fill_rows], values[fill_rows], np.nan
                )
                if self.strategy == "mean":
                    fill = np.nanmean(selected, axis=1)
                else:
                    fill = np.nanmedian(selected, axis=1)
                result[batch_rows[fill_rows], col] = fill

        return result

def make_imputers():
    imputers = make_original_imputers()
    imputers["FaissImputer[complete]"] = imputers.pop("FaissImputer")
    imputers["FaissImputer[available]"] = FaissImputer(
        n_neighbors=N_NEIGHBORS,
        donor_policy="available",
    )
    imputers["FaissImputer[native-NaN experimental]"] = NativeNaNBenchmark(
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

class TimedNativeIndex:
    def __init__(self, index):
        self.index = index
        self.seconds = 0.0
        self.calls = 0
        self.max_k = 0

    def __getattr__(self, name):
        return getattr(self.index, name)

    def search(self, queries, k):
        started = time.perf_counter()
        result = self.index.search(queries, k)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        self.max_k = max(self.max_k, int(k))
        return result

def profile_native_search():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmark_outputs"
        / "partial_donors.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["parameters"]
    seed = params["seeds"][0]
    train, truth = make_correlated_data(
        50000 + seed,
        params["n_train"],
        params["n_test"],
        params["n_features"],
    )
    rng = np.random.default_rng(60000 + seed)
    train[rng.random(train.shape) < params["train_missing_rate"]] = np.nan
    mask = make_missing_mask(
        70000 + seed,
        params["n_test"],
        params["n_features"],
        params["query_pattern"],
    )
    query = truth.copy()
    query[mask] = np.nan
    records = []

    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        imputer = NativeNaNBenchmark(
            n_neighbors=params["n_neighbors"],
            donor_policy="available",
        ).fit(train)

        for repeat in range(1, 4):
            started = time.perf_counter()
            expected = imputer.transform(query)
            normal_seconds = time.perf_counter() - started

            original_index = imputer.native_index_
            timer = TimedNativeIndex(original_index)
            imputer.native_index_ = timer
            try:
                started = time.perf_counter()
                actual = imputer.transform(query)
                profiled_seconds = time.perf_counter() - started
            finally:
                imputer.native_index_ = original_index

            np.testing.assert_array_equal(actual, expected)
            records.append({
                "repeat": repeat,
                "unprofiled_transform_seconds": normal_seconds,
                "profiled_transform_seconds": profiled_seconds,
                "search_seconds": timer.seconds,
                "non_search_seconds": profiled_seconds - timer.seconds,
                "search_calls": timer.calls,
                "max_search_k": timer.max_k,
                "outputs_identical": True,
            })

    data["native_profile"] = {
        "seed": seed,
        "purpose": "diagnostic_only",
        "trials": records,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Saved native search profile:", path)
    

if __name__ == "__main__":
    main()
    profile_native_search()
