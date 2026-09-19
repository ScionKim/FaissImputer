"""Measure one same-data imputation operation in an isolated worker."""

import argparse
import gc
import json
import time
import traceback

import faiss
import numpy as np
from sklearn import config_context
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.benchmark_scaling_threads import (
    FEATURES,
    METHODS,
    NEIGHBORS,
    PhasePeakSampler,
    WORKING_MEMORY_MIB,
    check_released_package,
    current_rss_mib,
    digest,
    make_model,
    metadata,
    peak_rss_mib,
)


APIS = ("fit_transform", "fit_then_transform")
TARGET_MISSING_RATE = 0.10


def make_training_data(size, seed, dtype):
    """Generate incomplete training data and its hidden ground truth."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if size <= NEIGHBORS:
        raise ValueError(f"size must exceed {NEIGHBORS}")

    loadings = np.random.default_rng([seed, 0]).normal(
        size=(5, FEATURES)
    )
    latent = np.random.default_rng([seed, 1]).normal(size=(size, 5))
    noise = np.random.default_rng([seed, 2]).normal(
        size=(size, FEATURES)
    )
    scale = np.sqrt(np.sum(loadings * loadings, axis=0) + 0.15**2)
    truth = ((latent @ loadings + 0.15 * noise) / scale).astype(dtype)

    missing = (
        np.random.default_rng([seed, 5]).random(truth.shape)
        < TARGET_MISSING_RATE
    )
    # Guarantee enough complete donors and at least one observed value
    # in every feature. Record the resulting actual missingness rate.
    missing[:NEIGHBORS] = False
    if not missing.any():
        raise ValueError("The generated case contains no missing entries")

    data = truth.copy()
    data[missing] = np.nan
    return data, truth, missing


def worker(config):
    method = config["method"]
    api = config["api"]
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if api not in APIS:
        raise ValueError(f"Unknown API: {api}")

    check_released_package(config.get("expected_version"))

    with threadpool_limits(limits=1), config_context(
        working_memory=WORKING_MEMORY_MIB
    ):
        faiss.omp_set_num_threads(1)

        data, truth, missing = make_training_data(
            config["size"], config["seed"], config["dtype"]
        )
        before = data.copy()
        fingerprints = {
            "input": digest(data),
            "truth": digest(truth),
            "missing": digest(missing),
        }

        # Warm the selected API on a separate, small dataset.
        warm_data, _, _ = make_training_data(32, 7, config["dtype"])
        warm_model = make_model(method)
        if api == "fit_transform":
            warm_model.fit_transform(warm_data)
        else:
            warm_model.fit(warm_data)
            warm_model.transform(warm_data)
        del warm_model, warm_data

        environment = metadata()
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
        gc.collect()

        measure_phase_memory = bool(config.get("measure_phase_memory", False))

        if api == "fit_transform":
            # A single call cannot attribute memory to separate phases.
            phase_memory = {
                "rss_before_fit_mib": None,
                "rss_after_fit_mib": None,
                "fit_retained_rss_mib": None,
                "fit_phase_peak_rss_mib": None,
                "transform_phase_peak_rss_mib": None,
            }

            started = time.perf_counter()
            output = model.fit_transform(data)
            finished = time.perf_counter()

            fit_seconds = None
            transform_seconds = None
            total_seconds = finished - started

        elif not measure_phase_memory:
            # Canonical timing path: identical to the original benchmark,
            # with no sampler thread and no GC between the phases.
            phase_memory = {
                "rss_before_fit_mib": None,
                "rss_after_fit_mib": None,
                "fit_retained_rss_mib": None,
                "fit_phase_peak_rss_mib": None,
                "transform_phase_peak_rss_mib": None,
            }

            started = time.perf_counter()
            model.fit(data)
            fitted = time.perf_counter()
            output = model.transform(data)
            finished = time.perf_counter()

            fit_seconds = fitted - started
            transform_seconds = finished - fitted
            total_seconds = finished - started

        else:
            # Phase-memory path: sampler setup/teardown, GC, and RSS
            # snapshots all sit outside the timed intervals, so
            # fit_seconds covers only model.fit() and transform_seconds
            # only model.transform().
            gc.collect()
            rss_before_fit = current_rss_mib()

            fit_sampler = PhasePeakSampler()
            fit_sampler.start()

            fit_started = time.perf_counter()
            model.fit(data)
            fit_finished = time.perf_counter()

            fit_phase_peak = fit_sampler.stop()

            gc.collect()
            rss_after_fit = current_rss_mib()

            transform_sampler = PhasePeakSampler()
            transform_sampler.start()

            transform_started = time.perf_counter()
            output = model.transform(data)
            transform_finished = time.perf_counter()

            transform_phase_peak = transform_sampler.stop()

            fit_seconds = fit_finished - fit_started
            transform_seconds = transform_finished - transform_started
            total_seconds = fit_seconds + transform_seconds

            phase_memory = {
                "rss_before_fit_mib": rss_before_fit,
                "rss_after_fit_mib": rss_after_fit,
                "fit_retained_rss_mib": (
                    None
                    if rss_before_fit is None or rss_after_fit is None
                    else rss_after_fit - rss_before_fit
                ),
                "fit_phase_peak_rss_mib": fit_phase_peak,
                "transform_phase_peak_rss_mib": transform_phase_peak,
            }

        assert output.shape == before.shape
        assert output.dtype == before.dtype
        assert np.isfinite(output).all()
        assert not np.shares_memory(output, data)
        np.testing.assert_array_equal(output[~missing], before[~missing])
        np.testing.assert_array_equal(data, before)

        values = output[missing].astype(np.float64)
        errors = values - truth[missing].astype(np.float64)

        result = {
            "status": "ok",
            "environment": environment,
            "fingerprints": fingerprints,
            "input_dtype": str(data.dtype),
            "output_dtype": str(output.dtype),
            "features": FEATURES,
            "n_neighbors": NEIGHBORS,
            "target_missing_rate": TARGET_MISSING_RATE,
            "actual_missing_rate": float(missing.mean()),
            "guaranteed_complete_rows": NEIGHBORS,
            "complete_donors": int((~missing.any(axis=1)).sum()),
            "missing_patterns": int(np.unique(missing, axis=0).shape[0]),
            "threads": 1,
            "threadpools": pools,
            "faiss_omp_threads": int(faiss.omp_get_max_threads()),
            "sklearn_working_memory_mib": WORKING_MEMORY_MIB,
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "fit_transform_seconds": (
                total_seconds if api == "fit_transform" else None
            ),
            "total_seconds": total_seconds,
            "quality_scope": "masked training entries",
            "scored_cells": int(missing.sum()),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae": float(np.mean(np.abs(errors))),
            "output_sha256": digest(np.ascontiguousarray(output)),
            "checks_passed": True,
            "_values": values.tolist(),
        }

        # Whole-process peak RSS, kept for comparison with the new
        # phase-separated measurements. Includes preparation and
        # validation, but precedes serialization.
        result["worker_peak_rss_mib"] = peak_rss_mib()
        result.update(phase_memory)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        result = worker(json.loads(args.config))
    except Exception:
        print(
            json.dumps(
                {"status": "error", "error": traceback.format_exc()},
                allow_nan=False,
            ),
            flush=True,
        )
        return 1

    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())