"""Small real-data imputation benchmark; not a clinical evaluation."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from faiss_imputer import FaissImputer

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "SimpleImputer",
    "KNNImputer",
    "FaissImputer[complete]",
    "FaissImputer[available]",
)
N_NEIGHBORS = 5
MISSING_RATE = 0.20


def make_model(name):
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    if name == "SimpleImputer":
        return SimpleImputer(strategy="mean")
    if name == "KNNImputer":
        return KNNImputer(
            n_neighbors=N_NEIGHBORS,
            weights="uniform",
            metric="nan_euclidean",
        )
    policy = "complete" if name == "FaissImputer[complete]" else "available"
    return FaissImputer(
        n_neighbors=N_NEIGHBORS,
        metric="l2",
        strategy="mean",
        index_factory="Flat",
        donor_policy=policy,
    )


def mask_summary(mask, high_age, eligible):
    by_age = {}
    for label, rows in (("low", ~high_age), ("high", high_age)):
        values = mask[np.ix_(rows, eligible)]
        by_age[label] = {
            "rows": int(rows.sum()),
            "eligible_missing_rate": (
                float(values.mean()) if values.size else None
            ),
        }
    return {
        "overall_missing_rate": float(mask.mean()),
        "eligible_missing_rate": float(mask[:, eligible].mean()),
        "missing_per_feature": mask.sum(axis=0).tolist(),
        "by_age": by_age,
    }


def prepare_case(data, names, seed, mechanism):
    train_raw, query_raw = train_test_split(
        data, test_size=0.25, random_state=seed
    )
    age = names.index("age")
    eligible = [
        i for i, name in enumerate(names)
        if name not in ("age", "sex")
    ]
    cutoff = float(np.median(train_raw[:, age]))
    base_probability = MISSING_RATE * data.shape[1] / len(eligible)
    rng = np.random.default_rng(seed + 10000)
    masked = []
    masks = []
    summaries = []

    for raw in (train_raw, query_raw):
        high_age = raw[:, age] > cutoff
        probabilities = np.full(len(raw), base_probability)
        if mechanism == "MAR":
            probabilities *= np.where(high_age, 1.5, 0.5)
        elif mechanism != "MCAR":
            raise ValueError(f"Unknown mechanism: {mechanism}")

        mask = np.zeros(raw.shape, dtype=bool)
        mask[:, eligible] = (
            rng.random((len(raw), len(eligible)))
            < probabilities[:, None]
        )
        observed = raw.copy()
        observed[mask] = np.nan
        masked.append(observed)
        masks.append(mask)
        summaries.append(mask_summary(mask, high_age, eligible))

    if not (~np.isnan(masked[0])).any(axis=0).all() or not masks[1].any():
        raise ValueError(
            "Case lacks training observations or scored query cells"
        )

    scaler = StandardScaler().fit(masked[0])
    train = scaler.transform(masked[0]).astype(np.float32)
    query = scaler.transform(masked[1]).astype(np.float32)
    truth = scaler.transform(query_raw).astype(np.float32)
    metadata = {
        "seed": seed,
        "mechanism": mechanism,
        "n_train": len(train),
        "n_query": len(query),
        "complete_donors": int((~np.isnan(train).any(axis=1)).sum()),
        "age_cutoff": cutoff,
        "train_mask": summaries[0],
        "query_mask": summaries[1],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return train, query, truth, masks[1], metadata


def timing_summary(samples):
    result = {}
    for field in ("fit_seconds", "transform_seconds", "total_seconds"):
        values = [sample[field] for sample in samples]
        result[field] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return result


def run_case(data, names, seed, mechanism, repeats):
    train, query, truth, missing, case = prepare_case(
        data, names, seed, mechanism
    )
    train_before, query_before = train.copy(), query.copy()
    outputs = {}
    records = {}

    for name in METHODS:
        if (
            name == "FaissImputer[complete]"
            and case["complete_donors"] < N_NEIGHBORS
        ):
            records[name] = {
                "status": "not_applicable",
                "reason": "Fewer complete donors than n_neighbors",
            }
        else:
            records[name] = {"status": "ok", "samples": []}

    for repeat in range(repeats):
        offset = repeat % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        for name in order:
            record = records[name]
            if record["status"] != "ok":
                continue

            model = make_model(name)
            gc.collect()
            started = time.perf_counter()
            model.fit(train)
            fitted = time.perf_counter()
            output = model.transform(query)
            finished = time.perf_counter()

            record["samples"].append({
                "repeat": repeat + 1,
                "fit_seconds": fitted - started,
                "transform_seconds": finished - fitted,
                "total_seconds": finished - started,
            })
            assert output.shape == query.shape
            assert np.isfinite(output).all()
            assert not np.shares_memory(output, query)
            np.testing.assert_array_equal(
                output[~missing], query[~missing]
            )
            np.testing.assert_array_equal(train, train_before)
            np.testing.assert_array_equal(query, query_before)
            if name.startswith("FaissImputer"):
                assert output.dtype == np.float32

            if name in outputs:
                np.testing.assert_array_equal(output, outputs[name])
            else:
                outputs[name] = output.copy()
            del model

    for name, record in records.items():
        if record["status"] != "ok":
            continue

        errors = (
            outputs[name][missing].astype(np.float64)
            - truth[missing].astype(np.float64)
        )
        record["quality"] = {
            "scored_cells": int(missing.sum()),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae": float(np.mean(np.abs(errors))),
            "max_abs_difference_from_knn": float(np.max(np.abs(
                outputs[name][missing].astype(np.float64)
                - outputs["KNNImputer"][missing].astype(np.float64)
            ))),
        }
        record["timing"] = timing_summary(record["samples"])
        record["checks_passed"] = True

    case["methods"] = records
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[101, 202, 303, 404, 505],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs" / "real_data.json",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(
        seed < 0 or seed >= 2**32 for seed in args.seeds
    ):
        parser.error(
            "repeats >= 1 and seeds in [0, 2**32) are required"
        )
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not contain duplicates")

    dataset = load_diabetes(scaled=False)
    data = np.asarray(dataset.data, dtype=np.float64)
    names = list(dataset.feature_names)
    assert np.isfinite(data).all()
    cases = []

    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(7)
        warm_train = rng.normal(
            size=(32, data.shape[1])
        ).astype(np.float32)
        warm_query = rng.normal(
            size=(8, data.shape[1])
        ).astype(np.float32)
        warm_query[:, 2:4] = np.nan
        for name in METHODS:
            make_model(name).fit(warm_train).transform(warm_query)

        for seed in args.seeds:
            for mechanism in ("MCAR", "MAR"):
                case = run_case(
                    data, names, seed, mechanism, args.repeats
                )
                cases.append(case)
                print(
                    f"seed={seed} {mechanism} "
                    f"complete_donors={case['complete_donors']}",
                    flush=True,
                )
                for name, record in case["methods"].items():
                    if record["status"] != "ok":
                        print(
                            f"  {name}: {record['reason']}",
                            flush=True,
                        )
                        continue
                    rmse = record["quality"]["rmse"]
                    total = record["timing"]["total_seconds"]["median"]
                    print(
                        f"  {name}: RMSE={rmse:.6f}, "
                        f"total={total * 1000:.3f}ms",
                        flush=True,
                    )

    results = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "faiss_imputer": version("faiss-imputer"),
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
            "faiss": getattr(faiss, "__version__", "unknown"),
            "native_threads": 1,
            "dataset": "sklearn.datasets.load_diabetes(scaled=False)",
            "dataset_sha256": hashlib.sha256(
                np.ascontiguousarray(data, dtype="<f8").tobytes()
            ).hexdigest(),
            "feature_names": names,
        },
        "parameters": {
            "seeds": args.seeds,
            "timing_repeats": args.repeats,
            "n_neighbors": N_NEIGHBORS,
            "faiss_metric": "l2",
            "faiss_strategy": "mean",
            "faiss_index_factory": "Flat",
            "nominal_overall_missing_rate": MISSING_RATE,
            "always_observed": ["age", "sex"],
        },
        "notes": [
            "Real feature values with artificial missingness; target is unused.",
            "MCAR masks eligible cells independently at probability 0.25.",
            "MAR uses probability 0.125 below/equal to the training age median, "
            "and 0.375 above it; age and sex always remain observed.",
            "Mechanisms reuse the same split and seeded uniform draws.",
            "Scaling is fitted only on observed training values, shared by all methods.",
            "Inputs and scaled scoring truth use float32.",
            "RMSE/MAE score hidden query cells in mechanism-specific standardized units.",
            "Different mechanisms have different scored cells and fitted scales.",
            "Repetitions measure timing, not independent accuracy evidence.",
            "Native libraries are warmed; method order rotates across repetitions.",
            "Timing excludes data preparation, scoring, and checks.",
            "No equality to KNNImputer, accuracy ranking, or speed threshold is required.",
            "Not-applicable complete-policy cases are recorded, not silently removed.",
            "Peak memory is not measured; this small dataset is not a scalability test.",
            "Not a clinical evaluation or a claim about natural missingness.",
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
