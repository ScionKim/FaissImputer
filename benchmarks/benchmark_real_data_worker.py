"""Measure one held-out real-data imputation run in an isolated worker."""

import argparse
import gc
import json
import time
import traceback

import faiss
import numpy as np
from sklearn import config_context
from sklearn.impute import KNNImputer, SimpleImputer
from threadpoolctl import threadpool_info, threadpool_limits

from faiss_imputer import FaissImputer

from benchmarks.benchmark_real_data_cases import (
    MAR_REFERENCE_ROWS,
    MISSING_RATE,
    N_NEIGHBORS,
    QUERY_SIZE,
    array_digest,
    load_dataset,
    prepare_case,
)
from benchmarks.benchmark_scaling_threads import (
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
    peak_rss_mib,
)


METHODS = (
    "SimpleImputer[mean]",
    "SimpleImputer[median]",
    "KNNImputer",
    "FaissImputer[complete]",
    "FaissImputer[available]",
)


def make_model(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    if method == "SimpleImputer[mean]":
        return SimpleImputer(strategy="mean", copy=True)

    if method == "SimpleImputer[median]":
        return SimpleImputer(strategy="median", copy=True)

    if method == "KNNImputer":
        return KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights="uniform",
            metric="nan_euclidean",
            copy=True,
        )

    policy = "complete" if method == "FaissImputer[complete]" else "available"
    return FaissImputer(
        n_neighbors=N_NEIGHBORS,
        metric="l2",
        strategy="mean",
        weights="uniform",
        index_factory="Flat",
        donor_policy=policy,
        copy=True,
    )


def quality_summary(output, truth, missing, case):
    """Score hidden query entries, keeping original units feature-specific."""
    values = output[missing].astype(np.float64)
    errors = values - truth[missing].astype(np.float64)
    if not len(errors):
        raise ValueError("No hidden query entries are available for scoring")

    feature_quality = []
    for column, name in enumerate(case["feature_names"]):
        selected = missing[:, column]
        count = int(selected.sum())
        record = {
            "feature": name,
            "scored_cells": count,
            "rmse_standardized": None,
            "mae_standardized": None,
            "rmse_original_units": None,
            "mae_original_units": None,
        }

        if count:
            column_errors = (
                output[selected, column].astype(np.float64)
                - truth[selected, column].astype(np.float64)
            )
            original_errors = column_errors * case["scaler_scale"][column]

            record.update({
                "rmse_standardized": float(np.sqrt(
                    np.mean(column_errors * column_errors)
                )),
                "mae_standardized": float(np.mean(np.abs(column_errors))),
                "rmse_original_units": float(np.sqrt(
                    np.mean(original_errors * original_errors)
                )),
                "mae_original_units": float(np.mean(np.abs(original_errors))),
            })

        feature_quality.append(record)

    quality = {
        "quality_scope": "masked held-out query entries",
        "quality_units": "standardized using observed training values",
        "scored_cells": int(len(errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "feature_quality": feature_quality,
    }
    return quality, values


def worker(config):
    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    check_released_package(config.get("expected_version"))

    with threadpool_limits(limits=1), config_context(
        working_memory=WORKING_MEMORY_MIB
    ):
        faiss.omp_set_num_threads(1)

        data, names, dataset_metadata = load_dataset(
            config["data_home"],
            download_if_missing=False,
        )
        train, query, truth, missing, case = prepare_case(
            data,
            names,
            config["seed"],
            config["mechanism"],
            train_size=config["train_size"],
            query_size=config.get("query_size", QUERY_SIZE),
            dtype=config["dtype"],
            missing_rate=config.get("missing_rate", MISSING_RATE),
            mar_reference_rows=config.get(
                "mar_reference_rows", MAR_REFERENCE_ROWS
            ),
        )
        del data, names

        common = {
            "environment": metadata(),
            "dataset": dataset_metadata,
            "case": case,
        }

        if (
            method == "FaissImputer[complete]"
            and case["complete_donors"] < N_NEIGHBORS
        ):
            return {
                **common,
                "status": "not_applicable",
                "reason": "Fewer complete donors than n_neighbors",
                "checks_passed": None,
            }

        train_before = train.copy()
        query_before = query.copy()

        # Warm the selected method on separate synthetic data.
        rng = np.random.default_rng(7)
        warm_train = rng.normal(
            size=(32, train.shape[1])
        ).astype(train.dtype)
        warm_query = rng.normal(
            size=(8, query.shape[1])
        ).astype(query.dtype)
        warm_train[::4, 1] = np.nan
        warm_query[:, 1:] = np.nan

        warm_model = make_model(method)
        warm_model.fit(warm_train)
        warm_model.transform(warm_query)
        del warm_model, warm_train, warm_query

        pools = [
            {
                key: pool.get(key)
                for key in (
                    "internal_api",
                    "prefix",
                    "num_threads",
                    "version",
                    "architecture",
                )
            }
            for pool in threadpool_info()
        ]

        model = make_model(method)
        recorded_parameters = {
            name: value
            for name, value in model.get_params(deep=False).items()
            if name in {
                "n_neighbors",
                "weights",
                "strategy",
                "metric",
                "donor_policy",
                "index_factory",
                "copy",
            }
        }
        gc.collect()

        started = time.perf_counter()
        model.fit(train)
        fitted = time.perf_counter()
        output = model.transform(query)
        finished = time.perf_counter()

        # Fit and transform are consecutive, without explicit GC or
        # memory sampling between them.
        fit_seconds = fitted - started
        transform_seconds = finished - fitted
        total_seconds = finished - started

        assert isinstance(output, np.ndarray)
        assert output.shape == query_before.shape
        assert output.dtype == query_before.dtype
        assert np.isfinite(output).all()
        assert not np.shares_memory(output, query)

        np.testing.assert_array_equal(
            output[~missing], query_before[~missing]
        )
        np.testing.assert_array_equal(train, train_before)
        np.testing.assert_array_equal(query, query_before)

        quality, values = quality_summary(output, truth, missing, case)

        result = {
            **common,
            "status": "ok",
            "model_parameters": recorded_parameters,
            "input_dtype": str(query.dtype),
            "output_dtype": str(output.dtype),
            "threads": 1,
            "threadpools": pools,
            "faiss_omp_threads": int(faiss.omp_get_max_threads()),
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "total_seconds": total_seconds,
            **quality,
            "output_sha256": array_digest(output),
            "checks_passed": True,
            "_values": values.tolist(),
        }

        # Includes preparation, warmup, and validation; excludes the
        # JSON serialization performed by main().
        result["worker_peak_rss_mib"] = peak_rss_mib()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        result = worker(json.loads(args.config))
        payload = json.dumps(result, allow_nan=False)
    except Exception:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": traceback.format_exc(),
                },
                allow_nan=False,
            ),
            flush=True,
        )
        return 1

    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())