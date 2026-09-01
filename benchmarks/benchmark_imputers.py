from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np
import sklearn
from sklearn.impute import KNNImputer, SimpleImputer
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "benchmark_outputs"
RESULTS_PATH = OUTPUT_DIR / "imputer_benchmark_results.json"
REPORT_PATH = OUTPUT_DIR / "imputer_benchmark_report.md"

N_FEATURES = 20
N_NEIGHBORS = 5
MISSING_RATE = 0.20
TRAIN_MISSING_RATE = 0.10
THREADS = 1
PATTERNS = ("fixed", "eight", "random")
ACCURACY_SEEDS = [101, 202, 303, 404, 505]
TIMING_TRAIN_SIZES = [1_000, 5_000, 20_000]
TIMING_TEST_SIZE = 300
TIMING_REPEATS = 7


def make_correlated_data(
    seed: int,
    n_train: int,
    n_test: int,
    n_features: int = N_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent_dim = 5
    loadings = rng.normal(size=(latent_dim, n_features))

    def sample(n_rows: int) -> np.ndarray:
        latent = rng.normal(size=(n_rows, latent_dim))
        noise = 0.15 * rng.normal(size=(n_rows, n_features))
        return latent @ loadings + noise

    train = sample(n_train)
    test = sample(n_test)

    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[scales == 0] = 1.0

    train = ((train - means) / scales).astype(np.float32)
    test = ((test - means) / scales).astype(np.float32)
    return train, test


def make_missing_mask(
    seed: int,
    n_rows: int,
    n_features: int,
    pattern: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if pattern == "fixed":
        mask = np.zeros((n_rows, n_features), dtype=bool)
        missing_columns = rng.choice(
            n_features,
            size=max(1, round(n_features * MISSING_RATE)),
            replace=False,
        )
        mask[:, missing_columns] = True
        return mask

    if pattern == "eight":
        n_missing = max(1, round(n_features * MISSING_RATE))
        row_patterns: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        while len(row_patterns) < 8:
            missing_columns = tuple(
                sorted(
                    rng.choice(
                        n_features,
                        size=n_missing,
                        replace=False,
                    ).tolist()
                )
            )
            if missing_columns in seen:
                continue
            seen.add(missing_columns)
            row_mask = np.zeros(n_features, dtype=bool)
            row_mask[list(missing_columns)] = True
            row_patterns.append(row_mask)

        assignments = np.arange(n_rows) % len(row_patterns)
        rng.shuffle(assignments)
        return np.stack([row_patterns[index] for index in assignments])

    if pattern != "random":
        raise ValueError(f"Unknown pattern: {pattern}")

    n_missing = max(1, round(n_features * MISSING_RATE))
    mask = np.zeros((n_rows, n_features), dtype=bool)
    for row_idx in range(n_rows):
        missing_columns = rng.choice(
            n_features,
            size=n_missing,
            replace=False,
        )
        mask[row_idx, missing_columns] = True
    return mask


def make_imputers() -> dict[str, object]:
    return {
        "SimpleImputer": SimpleImputer(strategy="mean"),
        "KNNImputer": KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights="uniform",
            metric="nan_euclidean",
        ),
        "FaissImputer": FaissImputer(
            n_neighbors=N_NEIGHBORS,
            metric="l2",
            strategy="mean",
            index_factory="Flat",
        ),
    }


def imputation_errors(
    truth: np.ndarray,
    imputed: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[float, float]:
    errors = imputed[missing_mask] - truth[missing_mask]
    rmse = float(np.sqrt(np.mean(np.square(errors, dtype=np.float64))))
    mae = float(np.mean(np.abs(errors), dtype=np.float64))
    return rmse, mae


def unique_missing_patterns(mask: np.ndarray) -> int:
    return int(np.unique(mask, axis=0).shape[0])


def run_accuracy_trials() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for pattern in PATTERNS:
        for seed in ACCURACY_SEEDS:
            train, truth = make_correlated_data(
                seed=seed,
                n_train=5_000,
                n_test=500,
            )
            mask = make_missing_mask(
                seed=seed + 10_000,
                n_rows=truth.shape[0],
                n_features=truth.shape[1],
                pattern=pattern,
            )
            incomplete = truth.copy()
            incomplete[mask] = np.nan

            outputs: dict[str, np.ndarray] = {}
            for method_name, imputer in make_imputers().items():
                outputs[method_name] = imputer.fit(train).transform(incomplete)
                rmse, mae = imputation_errors(truth, outputs[method_name], mask)
                rows.append(
                    {
                        "pattern": pattern,
                        "seed": seed,
                        "method": method_name,
                        "rmse": rmse,
                        "mae": mae,
                        "unique_patterns": unique_missing_patterns(mask),
                    }
                )

            max_abs_difference = float(
                np.max(
                    np.abs(
                        outputs["KNNImputer"][mask]
                        - outputs["FaissImputer"][mask]
                    )
                )
            )
            rows.append(
                {
                    "pattern": pattern,
                    "seed": seed,
                    "method": "KNN_vs_Faiss",
                    "max_abs_difference": max_abs_difference,
                    "unique_patterns": unique_missing_patterns(mask),
                }
            )

            if max_abs_difference > 1e-4:
                raise AssertionError(
                    "FaissImputer and KNNImputer diverged under the "
                    f"controlled comparison: {max_abs_difference}"
                )

    return rows


def run_native_accuracy_trials() -> list[dict[str, object]]:
    """Compare each imputer's native handling of incomplete fitting data."""
    rows: list[dict[str, object]] = []

    for seed in ACCURACY_SEEDS:
        train_truth, test_truth = make_correlated_data(
            seed=50_000 + seed,
            n_train=5_000,
            n_test=500,
        )
        rng = np.random.default_rng(60_000 + seed)
        train_mask = rng.random(train_truth.shape) < TRAIN_MISSING_RATE
        train_incomplete = train_truth.copy()
        train_incomplete[train_mask] = np.nan
        complete_donors = int((~train_mask.any(axis=1)).sum())

        if complete_donors < N_NEIGHBORS:
            raise AssertionError("Native trial did not retain enough complete donors")

        test_mask = make_missing_mask(
            seed=70_000 + seed,
            n_rows=test_truth.shape[0],
            n_features=test_truth.shape[1],
            pattern="random",
        )
        test_incomplete = test_truth.copy()
        test_incomplete[test_mask] = np.nan

        for method_name, imputer in make_imputers().items():
            output = imputer.fit(train_incomplete).transform(test_incomplete)
            rmse, mae = imputation_errors(test_truth, output, test_mask)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "rmse": rmse,
                    "mae": mae,
                    "complete_donors": complete_donors,
                    "total_train_rows": train_truth.shape[0],
                    "train_missing_rate": TRAIN_MISSING_RATE,
                    "test_missing_rate": MISSING_RATE,
                }
            )

    return rows


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def time_case(
    train: np.ndarray,
    incomplete: np.ndarray,
) -> list[dict[str, object]]:
    method_names = ("SimpleImputer", "KNNImputer", "FaissImputer")
    run_orders = (
        ("SimpleImputer", "KNNImputer", "FaissImputer"),
        ("KNNImputer", "FaissImputer", "SimpleImputer"),
        ("FaissImputer", "SimpleImputer", "KNNImputer"),
        ("FaissImputer", "KNNImputer", "SimpleImputer"),
        ("SimpleImputer", "FaissImputer", "KNNImputer"),
        ("KNNImputer", "SimpleImputer", "FaissImputer"),
    )
    samples = {
        method_name: {"fit": [], "transform": [], "total": []}
        for method_name in method_names
    }

    for repeat in range(TIMING_REPEATS):
        for method_name in run_orders[repeat % len(run_orders)]:
            gc.collect()
            imputer = make_imputers()[method_name]

            start = time.perf_counter()
            imputer.fit(train)
            fitted = time.perf_counter()
            output = imputer.transform(incomplete)
            transformed = time.perf_counter()

            if np.isnan(output).any():
                raise AssertionError(f"{method_name} left missing values")

            samples[method_name]["fit"].append(fitted - start)
            samples[method_name]["transform"].append(transformed - fitted)
            samples[method_name]["total"].append(transformed - start)

    rows: list[dict[str, object]] = []
    for method_name in method_names:
        fit = timing_summary(samples[method_name]["fit"])
        transform = timing_summary(samples[method_name]["transform"])
        total = timing_summary(samples[method_name]["total"])
        rows.append(
            {
                "method": method_name,
                "fit_seconds_median": fit["median"],
                "fit_seconds_q25": fit["q25"],
                "fit_seconds_q75": fit["q75"],
                "transform_seconds_median": transform["median"],
                "transform_seconds_q25": transform["q25"],
                "transform_seconds_q75": transform["q75"],
                "total_seconds_median": total["median"],
                "total_seconds_q25": total["q25"],
                "total_seconds_q75": total["q75"],
                "fit_seconds_all": samples[method_name]["fit"],
                "transform_seconds_all": samples[method_name]["transform"],
                "total_seconds_all": samples[method_name]["total"],
            }
        )
    return rows


def warm_up() -> None:
    train, truth = make_correlated_data(7, 256, 32)
    mask = make_missing_mask(8, 32, N_FEATURES, "fixed")
    incomplete = truth.copy()
    incomplete[mask] = np.nan

    for imputer in make_imputers().values():
        imputer.fit(train).transform(incomplete)


def run_timing_trials() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_max, truth = make_correlated_data(
        seed=28_888,
        n_train=max(TIMING_TRAIN_SIZES),
        n_test=TIMING_TEST_SIZE,
    )

    for pattern_index, pattern in enumerate(PATTERNS):
        mask = make_missing_mask(
            seed=30_000 + pattern_index,
            n_rows=truth.shape[0],
            n_features=truth.shape[1],
            pattern=pattern,
        )
        incomplete = truth.copy()
        incomplete[mask] = np.nan

        for n_train in TIMING_TRAIN_SIZES:
            train = train_max[:n_train]
            for timing in time_case(train, incomplete):
                rows.append(
                    {
                        "pattern": pattern,
                        "n_train": n_train,
                        "n_test": TIMING_TEST_SIZE,
                        "n_features": N_FEATURES,
                        "missing_rate": MISSING_RATE,
                        "unique_patterns": unique_missing_patterns(mask),
                        **timing,
                    }
                )

    return rows


def summarize_accuracy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for pattern in PATTERNS:
        for method in ("SimpleImputer", "KNNImputer", "FaissImputer"):
            matches = [
                row
                for row in rows
                if row["pattern"] == pattern and row["method"] == method
            ]
            summary.append(
                {
                    "pattern": pattern,
                    "method": method,
                    "rmse_mean": statistics.mean(row["rmse"] for row in matches),
                    "rmse_stdev": statistics.stdev(row["rmse"] for row in matches),
                    "mae_mean": statistics.mean(row["mae"] for row in matches),
                    "mae_stdev": statistics.stdev(row["mae"] for row in matches),
                }
            )
    return summary


def summarize_native_accuracy(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in ("SimpleImputer", "KNNImputer", "FaissImputer"):
        matches = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "rmse_mean": statistics.mean(row["rmse"] for row in matches),
                "rmse_stdev": statistics.stdev(row["rmse"] for row in matches),
                "mae_mean": statistics.mean(row["mae"] for row in matches),
                "mae_stdev": statistics.stdev(row["mae"] for row in matches),
                "complete_donors_mean": statistics.mean(
                    row["complete_donors"] for row in matches
                ),
            }
        )
    return summary


def summarize_parity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for pattern in PATTERNS:
        matches = [
            row
            for row in rows
            if row["pattern"] == pattern and row["method"] == "KNN_vs_Faiss"
        ]
        summary.append(
            {
                "pattern": pattern,
                "max_abs_difference_across_trials": max(
                    row["max_abs_difference"] for row in matches
                ),
                "unique_patterns_mean": statistics.mean(
                    row["unique_patterns"] for row in matches
                ),
            }
        )
    return summary


def make_report(results: dict[str, object]) -> str:
    accuracy_summary = results["accuracy_summary"]
    native_accuracy_summary = results["native_accuracy_summary"]
    parity_summary = results["parity_summary"]
    timing_rows = results["timing_trials"]

    lines = [
        "# FaissImputer 0.2.0 benchmark",
        "",
        "## Environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- faiss-imputer: {results['metadata']['faiss_imputer_version']}",
        f"- scikit-learn: {results['metadata']['scikit_learn_version']}",
        f"- NumPy: {results['metadata']['numpy_version']}",
        f"- Faiss: {results['metadata']['faiss_version']}",
        f"- OS: {results['metadata']['platform']}",
        f"- Logical CPUs: {results['metadata']['logical_cpus']}",
        f"- Threads allowed per native library: {THREADS}",
        "",
        "## Method",
        "",
        "- Correlated synthetic numeric data, standardized from complete training data.",
        "- In the controlled accuracy, parity, and runtime sections, complete training rows are supplied to every imputer and missing values are introduced only in held-out test rows.",
        f"- {N_NEIGHBORS} uniform neighbors; Faiss uses exact `Flat` L2 search.",
        f"- Accuracy uses {len(ACCURACY_SEEDS)} seeds, 5,000 train rows, 500 test rows, {N_FEATURES} features, and {MISSING_RATE:.0%} missingness.",
        f"- Timing is the median of {TIMING_REPEATS} runs with {TIMING_TEST_SIZE} test rows. Method order is rotated, the same test matrix and mask are reused across train sizes, and larger training sets extend smaller ones by prefix.",
        "- Every test row has exactly four missing values. `fixed` means one shared pattern, `eight` means eight controlled patterns, and `random` means each row independently chooses four missing columns.",
        "",
        "## Accuracy on deliberately hidden values",
        "",
        "| Pattern | Method | RMSE mean | RMSE SD | MAE mean | MAE SD |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in accuracy_summary:
        lines.append(
            "| {pattern} | {method} | {rmse_mean:.6f} | {rmse_stdev:.6f} | "
            "{mae_mean:.6f} | {mae_stdev:.6f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Native behavior with incomplete fitting data",
            "",
            f"The fitting matrix also has {TRAIN_MISSING_RATE:.0%} independently missing cells. FaissImputer uses only the remaining complete rows as donors, while KNNImputer can use partially observed rows.",
            "To isolate that donor-pool policy, this synthetic experiment standardizes from the complete training truth before cells are hidden; it is not an end-to-end preprocessing benchmark.",
            "",
            "| Method | RMSE mean | RMSE SD | MAE mean | MAE SD | Complete rows available |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in native_accuracy_summary:
        lines.append(
            "| {method} | {rmse_mean:.6f} | {rmse_stdev:.6f} | "
            "{mae_mean:.6f} | {mae_stdev:.6f} | "
            "{complete_donors_mean:.1f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## KNNImputer/FaissImputer parity",
            "",
            "The parity check applies to this controlled continuous dataset, which did not contain distance ties.",
            "",
            "| Pattern | Largest absolute difference | Mean unique patterns |",
            "|---|---:|---:|",
        ]
    )
    for row in parity_summary:
        lines.append(
            "| {pattern} | {max_abs_difference_across_trials:.9f} | "
            "{unique_patterns_mean:.1f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "The transform IQR is the middle 50% of the seven measurements.",
            "",
            "| Pattern | Train rows | Unique patterns | Method | Fit median (ms) | Transform median (ms) | Transform IQR (ms) | Total median (ms) |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            "| {pattern} | {n_train:,} | {unique_patterns} | {method} | "
            "{fit_ms:.2f} | {transform_ms:.2f} | {transform_q25_ms:.2f}–{transform_q75_ms:.2f} | {total_ms:.2f} |".format(
                **row,
                fit_ms=row["fit_seconds_median"] * 1_000,
                transform_ms=row["transform_seconds_median"] * 1_000,
                transform_q25_ms=row["transform_seconds_q25"] * 1_000,
                transform_q75_ms=row["transform_seconds_q75"] * 1_000,
                total_ms=row["total_seconds_median"] * 1_000,
            )
        )

    lines.extend(
        [
            "",
            "## Transform speed relative to KNNImputer",
            "",
            "Values above 1 mean FaissImputer was faster; values below 1 mean it was slower.",
            "",
            "| Pattern | Train rows | KNN/Faiss transform ratio |",
            "|---|---:|---:|",
        ]
    )
    for pattern in PATTERNS:
        for n_train in TIMING_TRAIN_SIZES:
            matches = {
                row["method"]: row
                for row in timing_rows
                if row["pattern"] == pattern and row["n_train"] == n_train
            }
            ratio = (
                matches["KNNImputer"]["transform_seconds_median"]
                / matches["FaissImputer"]["transform_seconds_median"]
            )
            lines.append(f"| {pattern} | {n_train:,} | {ratio:.2f}x |")

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The controlled sections isolate numerical imputation with complete training donors. They do not establish results for every dataset, hardware configuration, or missing-data mechanism. The separate incomplete-fitting-data section deliberately compares each implementation's native donor policy; there, KNNImputer and FaissImputer 0.2.0 use different donor pools and should not be interpreted as the same algorithm with different search engines.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(THREADS)

    with threadpool_limits(limits=THREADS):
        warm_up()
        accuracy_trials = run_accuracy_trials()
        native_accuracy_trials = run_native_accuracy_trials()
        timing_trials = run_timing_trials()

    results = {
        "metadata": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": platform.python_version(),
            "faiss_imputer_version": version("faiss-imputer"),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "faiss_version": faiss.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "logical_cpus": os.cpu_count(),
            "threads": THREADS,
        },
        "parameters": {
            "n_features": N_FEATURES,
            "n_neighbors": N_NEIGHBORS,
            "missing_rate": MISSING_RATE,
            "train_missing_rate_for_native_trials": TRAIN_MISSING_RATE,
            "accuracy_seeds": ACCURACY_SEEDS,
            "timing_train_sizes": TIMING_TRAIN_SIZES,
            "timing_test_size": TIMING_TEST_SIZE,
            "timing_repeats": TIMING_REPEATS,
        },
        "accuracy_trials": accuracy_trials,
        "accuracy_summary": summarize_accuracy(accuracy_trials),
        "native_accuracy_trials": native_accuracy_trials,
        "native_accuracy_summary": summarize_native_accuracy(
            native_accuracy_trials
        ),
        "parity_summary": summarize_parity(accuracy_trials),
        "timing_trials": timing_trials,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(make_report(results), encoding="utf-8")
    print(REPORT_PATH)
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()

