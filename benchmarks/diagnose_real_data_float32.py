"""Diagnose one real-data float32 disagreement without timing benchmarks."""

import argparse
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import faiss
import numpy as np
from sklearn import config_context
from sklearn.impute import KNNImputer, _knn
from threadpoolctl import threadpool_info, threadpool_limits

from faiss_imputer import FaissImputer
from benchmarks.benchmark_real_data_cases import (
    array_digest,
    load_dataset,
    prepare_case,
)
from benchmarks.benchmark_scaling_threads import (
    ROOT,
    WORKING_MEMORY_MIB,
    check_released_package,
    metadata,
)


K = 5
TARGET = {
    "train_size": 15000,
    "query_size": 3000,
    "mechanism": "MAR",
    "dtype": "float32",
    "seed": 303,
}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_model(method):
    if method == "knn":
        return KNNImputer(
            n_neighbors=K,
            weights="uniform",
            metric="nan_euclidean",
            copy=True,
        )
    return FaissImputer(
        n_neighbors=K,
        donor_policy="available",
        metric="l2",
        index_factory="Flat",
        strategy="mean",
        weights="uniform",
        copy=True,
    )


def direct_squared_distances(train, query_row):
    """Independent float64 subtraction over shared observed features."""
    donors = train.astype(np.float64)
    row = query_row.astype(np.float64)
    shared = ~np.isnan(donors) & ~np.isnan(row)
    counts = shared.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("This diagnostic requires shared observed features")

    delta = np.zeros_like(donors)
    np.subtract(donors, row, out=delta, where=shared)
    squared = np.sum(delta * delta, axis=1)
    squared *= train.shape[1] / counts

    if not np.isfinite(squared).all():
        raise ValueError("Direct reference distances must remain finite")
    return squared, counts


def selected_values(train, column, ids):
    ids = np.asarray(ids, dtype=np.intp)
    values = train[ids, column]
    if not len(ids) or not np.isfinite(values).all():
        raise ValueError("Selected donors must contain finite target values")
    return {
        "training_row_indices": ids,
        "target_values": values,
        "mean_float32": float(np.mean(values, dtype=np.float32)),
        "mean_float64": float(np.mean(values, dtype=np.float64)),
    }


def boundary_details(distances, eligible, train, column):
    """Describe exact boundary ties; do not merge merely close distances."""
    ordered = eligible[
        np.argsort(distances[eligible], kind="stable")
    ]
    k = min(K, len(ordered))
    if not k:
        raise ValueError("No eligible donors")

    cutoff = distances[ordered[k - 1]]
    closer = eligible[distances[eligible] < cutoff]
    tied = eligible[distances[eligible] == cutoff]
    slots = k - len(closer)

    fixed_sum = np.sum(train[closer, column], dtype=np.float64)
    tied_values = np.sort(train[tied, column].astype(np.float64))
    next_distance = (
        float(distances[ordered[k]]) if k < len(ordered) else None
    )

    return {
        "cutoff": float(cutoff),
        "next_distance": next_distance,
        "next_gap": (
            next_distance - float(cutoff)
            if next_distance is not None else None
        ),
        "strictly_closer_count": len(closer),
        "exact_boundary_tie_count": len(tied),
        "boundary_slots": slots,
        "smallest_float64_mean_at_boundary": float(
            (fixed_sum + tied_values[:slots].sum()) / k
        ),
        "largest_float64_mean_at_boundary": float(
            (fixed_sum + tied_values[-slots:].sum()) / k
        ),
    }


def trace_knn_distances(model, train, query, wanted_rows):
    """Capture actual distance rows before sklearn's reducer uses them."""
    missing_rows = np.flatnonzero(np.isnan(query).any(axis=1))
    wanted = {
        position: int(row)
        for position, row in enumerate(missing_rows)
        if int(row) in wanted_rows
    }
    captured = {}
    original = _knn.pairwise_distances_chunked

    def traced_pairwise(X, Y=None, **kwargs):
        np.testing.assert_array_equal(X, query[missing_rows])
        np.testing.assert_array_equal(Y, train)
        reducer = kwargs.pop("reduce_func", None)
        if reducer is None:
            raise RuntimeError("Unexpected KNNImputer distance interface")

        def traced_reducer(distances, start):
            for position, row in wanted.items():
                if start <= position < start + len(distances):
                    if row in captured:
                        raise RuntimeError("KNN distance row captured twice")
                    captured[row] = distances[position - start].copy()
            return reducer(distances, start)

        return original(
            X, Y, reduce_func=traced_reducer, **kwargs
        )

    with patch.object(
        _knn, "pairwise_distances_chunked", traced_pairwise
    ):
        output = model.transform(query)

    if set(captured) != set(wanted_rows):
        raise RuntimeError("Some requested KNN distance rows were not captured")
    return output, captured


def trace_faiss_searches(model, train, query, wanted_rows):
    """Capture finished searches for the requested query values."""
    np.testing.assert_array_equal(model.donors_, train)
    keys = {array_digest(query[row]) for row in wanted_rows}
    matching_rows = {key: [] for key in keys}
    logs = {key: [] for key in keys}

    for row in range(len(query)):
        key = array_digest(query[row])
        if key in matching_rows:
            matching_rows[key].append(row)

    index = model.available_index_
    original = index.search
    donor_counts = (~np.isnan(train)).sum(axis=0)

    def traced_search(queries, search_k):
        distances, ids = original(queries, search_k)

        for local_row, query_row in enumerate(queries):
            key = array_digest(query_row)
            if key not in logs:
                continue

            valid = (
                (ids[local_row] >= 0)
                & (ids[local_row] < len(train))
                & np.isfinite(distances[local_row])
            )
            valid_ids = ids[local_row, valid]
            valid_distances = distances[local_row, valid]
            columns = np.flatnonzero(np.isnan(query_row))
            selections = {}
            enough = True

            for column in columns:
                usable = ~np.isnan(train[valid_ids, column])
                eligible_ids = valid_ids[usable]
                required = min(K, int(donor_counts[column]))
                enough &= len(eligible_ids) >= required

                selections[str(column)] = {
                    "training_row_indices": eligible_ids[:K],
                    "search_squared_distances": valid_distances[usable][:K],
                }
                if eligible_ids.size:
                    selections[str(column)].update(
                        selected_values(
                            train, column, eligible_ids[:K]
                        )
                    )

            finished = (
                enough
                or not valid.all()
                or search_k >= len(train)
            )
            if finished:
                logs[key].append({
                    "requested_candidates": int(search_k),
                    "precision_refined": local_row in index.precise_rows,
                    "features": selections,
                })

        return distances, ids

    with patch.object(index, "search", traced_search):
        output = model.transform(query)

    if any(not records for records in logs.values()):
        raise RuntimeError("Some requested Faiss searches were not captured")
    return output, logs, matching_rows


def describe_cell(
    train, column, squared, shared_counts, knn_distances, searches
):
    eligible = np.flatnonzero(~np.isnan(train[:, column]))
    if not np.isfinite(knn_distances[eligible]).all():
        raise ValueError("Unexpected undefined KNN distances in this case")

    ordered = eligible[
        np.argsort(squared[eligible], kind="stable")
    ]
    k = min(K, len(eligible))
    reference_ids = ordered[:k]

    # Reconstruction from captured distances, not instrumented sklearn IDs.
    reconstructed_ids = eligible[
        np.argpartition(knn_distances[eligible], k - 1)[:k]
    ]

    candidate_ids = set(ordered[:k + 1].tolist())
    candidate_ids.update(reconstructed_ids.tolist())
    for search in searches:
        candidate_ids.update(
            search["features"][str(column)]["training_row_indices"].tolist()
        )

    donor_details = []
    for donor in sorted(candidate_ids):
        distance = float(knn_distances[donor])
        donor_details.append({
            "training_row_index": donor,
            "standardized_values": train[donor],
            "target_value": float(train[donor, column]),
            "shared_feature_count": int(shared_counts[donor]),
            "direct_squared_distance": float(squared[donor]),
            "knn_captured_distance": distance,
            "knn_captured_distance_squared": distance * distance,
            "donor_observed_squared_norm": float(np.nansum(
                train[donor].astype(np.float64) ** 2
            )),
        })

    return {
        "reference": {
            "distance_units": "squared missing-aware Euclidean",
            "tie_rule": "training-row order for exactly equal distances",
            **selected_values(train, column, reference_ids),
            "boundary": boundary_details(
                squared, eligible, train, column
            ),
        },
        "knn_reconstruction": {
            "selection_kind": (
                "np.argpartition on captured distances; "
                "not instrumented donor IDs"
            ),
            "distance_units": "missing-aware Euclidean",
            "distance_dtype": str(knn_distances.dtype),
            **selected_values(train, column, reconstructed_ids),
            "boundary": boundary_details(
                knn_distances, eligible, train, column
            ),
        },
        "donor_details": donor_details,
    }


def diagnose(args, report):
    check_released_package(args.expected_version)
    environment = metadata()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    report.update({
        "environment": environment,
        "provenance": provenance,
        "baseline_provenance": baseline["provenance"],
        "target": TARGET,
        "difference_threshold": args.threshold,
        "notes": [
            "This is a diagnostic, not a performance benchmark.",
            "Float64 controls use the prepared float32 values promoted to float64.",
            "The controls do not refit preprocessing on original float64 data.",
            "Direct reference distances use shared-feature float64 subtraction.",
            "KNN donor IDs are reconstructed from actual captured distances.",
            "A matching reconstructed mean does not prove actual KNN donor IDs.",
            "Faiss traces describe finished searches for identical query values.",
            "All donor indices refer to the prepared training-row order.",
            "Missing values in recorded arrays are represented as null.",
        ],
    })

    if provenance["version"] != args.expected_version:
        raise ValueError("Candidate version differs from provenance")
    if provenance["source_commit"] != environment["git_commit"]:
        raise ValueError("Candidate source differs from benchmark scripts")

    for name in ("numpy", "scikit_learn", "faiss"):
        if environment[name] != baseline["metadata"][name]:
            raise ValueError(f"Dependency differs from the original run: {name}")

    baseline_records = {
        row["method"]: row
        for row in baseline["records"]
        if all(row.get(name) == value for name, value in TARGET.items())
        and row["repeat"] == 1
        and row["method"] in ("KNNImputer", "FaissImputer[available]")
    }
    if len(baseline_records) != 2:
        raise ValueError("The archived target case is missing")

    data, names, dataset = load_dataset(
        args.data_home, download_if_missing=False
    )
    train, query, _, missing, case = prepare_case(
        data,
        names,
        seed=303,
        mechanism="MAR",
        train_size=15000,
        query_size=3000,
        dtype="float32",
        mar_reference_rows=10000,
    )
    del data

    for record in baseline_records.values():
        if case["fingerprints"] != record["case"]["fingerprints"]:
            raise ValueError("Prepared inputs differ from the archived case")

    report["dataset"] = dataset
    report["case"] = case
    models = {}
    outputs = {}

    for dtype_label, dtype in (
        ("float32", np.float32),
        ("promoted_float64", np.float64),
    ):
        fitted_input = train.astype(dtype, copy=True)
        query_input = query.astype(dtype, copy=True)
        train_hash = array_digest(fitted_input)
        query_hash = array_digest(query_input)

        for method in ("faiss", "knn"):
            label = f"{method}_{dtype_label}"
            model = make_model(method).fit(fitted_input)
            output = model.transform(query_input)

            if (
                output.shape != query.shape
                or output.dtype != np.dtype(dtype)
                or not np.isfinite(output).all()
            ):
                raise ValueError(f"Invalid output: {label}")
            np.testing.assert_array_equal(
                output[~missing], query_input[~missing]
            )
            if (
                array_digest(fitted_input) != train_hash
                or array_digest(query_input) != query_hash
            ):
                raise ValueError(f"Input modified: {label}")

            models[label] = model
            outputs[label] = output

    report["output_sha256"] = {
        label: array_digest(output)
        for label, output in outputs.items()
    }
    report["baseline_output_hashes_match"] = {
        label: report["output_sha256"][label]
        == baseline_records[method]["output_sha256"]
        for label, method in (
            ("faiss_float32", "FaissImputer[available]"),
            ("knn_float32", "KNNImputer"),
        )
    }

    comparisons = {}
    for left, right in (
        ("faiss_float32", "knn_float32"),
        ("faiss_float32", "faiss_promoted_float64"),
        ("knn_float32", "knn_promoted_float64"),
        ("faiss_promoted_float64", "knn_promoted_float64"),
    ):
        difference = np.abs(
            outputs[left][missing].astype(np.float64)
            - outputs[right][missing].astype(np.float64)
        )
        comparisons[f"{left} vs {right}"] = {
            "max_abs_difference": float(difference.max()),
            "cells_above_threshold": int(
                (difference > args.threshold).sum()
            ),
        }
    report["comparisons"] = comparisons

    difference = np.abs(
        outputs["faiss_float32"].astype(np.float64)
        - outputs["knn_float32"].astype(np.float64)
    )
    row_maxima = difference.max(axis=1)
    affected = np.flatnonzero(row_maxima > args.threshold)
    ordered = affected[
        np.argsort(-row_maxima[affected], kind="stable")
    ]
    wanted_rows = [int(row) for row in ordered[:args.max_rows]]

    report["affected_query_rows"] = int(len(affected))
    report["detailed_query_rows"] = wanted_rows
    report["details_truncated"] = len(affected) > len(wanted_rows)
    report["rows"] = []

    if wanted_rows:
        traced_knn, captured = trace_knn_distances(
            models["knn_float32"], train, query, wanted_rows
        )
        traced_faiss, searches, matching_rows = trace_faiss_searches(
            models["faiss_float32"], train, query, wanted_rows
        )
        np.testing.assert_array_equal(traced_knn, outputs["knn_float32"])
        np.testing.assert_array_equal(traced_faiss, outputs["faiss_float32"])
        report["traced_outputs_unchanged"] = True

        for row in wanted_rows:
            squared, shared_counts = direct_squared_distances(
                train, query[row]
            )
            key = array_digest(query[row])
            row_report = {
                "query_row_index": row,
                "standardized_query": query[row],
                "query_observed_squared_norm": float(np.nansum(
                    query[row].astype(np.float64) ** 2
                )),
                "max_abs_output_difference": float(row_maxima[row]),
                "query_rows_with_identical_values": matching_rows[key],
                "faiss_finished_searches": searches[key],
                "features": [],
            }

            for column in np.flatnonzero(missing[row]):
                detail = describe_cell(
                    train,
                    column,
                    squared,
                    shared_counts,
                    captured[row],
                    searches[key],
                )
                detail.update({
                    "feature_index": int(column),
                    "feature_name": names[column],
                    "outputs": {
                        label: float(output[row, column])
                        for label, output in outputs.items()
                    },
                })
                row_report["features"].append(detail)

            report["rows"].append(row_report)

    report["threadpools"] = [
        {
            name: pool.get(name)
            for name in ("internal_api", "num_threads", "version", "architecture")
        }
        for pool in threadpool_info()
    ]
    report["status"] = (
        "ok"
        if all(report["baseline_output_hashes_match"].values())
        else "baseline_output_mismatch"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            ROOT / "benchmarks" / "results"
            / "real-data-coverage-8f289647.json"
        ),
    )
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "benchmark_outputs"
            / "real_data_float32_diagnostic.json"
        ),
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.threshold)
        or args.threshold <= 0
        or args.max_rows < 1
    ):
        parser.error("Use a positive finite threshold and positive max-rows")

    report = {"status": "error"}
    try:
        with threadpool_limits(limits=1), config_context(
            working_memory=WORKING_MEMORY_MIB
        ):
            faiss.omp_set_num_threads(1)
            diagnose(args, report)
    except Exception as error:
        report.update({
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Status: {report['status']}", flush=True)
    print(f"Results: {args.output}", flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())