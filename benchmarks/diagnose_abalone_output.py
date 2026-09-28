"""Diagnose the archived Abalone float64 case without timing measurements."""

import argparse
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import faiss
import numpy as np
from sklearn import config_context
from sklearn.impute import KNNImputer, _knn
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.benchmark_real_data_cases import (
    array_digest, load_dataset, prepare_case,
)
from benchmarks.benchmark_scaling_threads import (
    ROOT, WORKING_MEMORY_MIB, check_released_package, metadata,
)
from benchmarks.diagnose_real_data_float32 import (
    boundary_details, direct_squared_distances, json_safe,
    make_model, trace_faiss_searches,
)


SOURCE = "ef04b1b0274d3abdd977440dd61e7d91e328c64d"
BASELINE_SHA256 = "4ce80d0aad87ddc9831a2131d752feff3278d563343ef643096c07e7f87fd64c"
CORE_SHA256 = {
    "__init__.py": "05cde2459315df68e0af593b92224a998b206b36531eda3b98194520a2d4b58f",
    "_matrix.py": "3454e6db843d74cde9c26628fee11cdcb658e13cfacec5319d61815bab4545a6",
    "faiss_imputer.py": "bf82b170a6af3161b49cf623f714b5b80ee849ff91a599f5fc05268f9acbd17a",
}
DEPENDENCIES = {
    "numpy": "2.5.3", "scikit-learn": "1.9.1", "faiss-cpu": "1.15.1",
    "scipy": "1.18.1", "threadpoolctl": "3.7.0", "joblib": "1.6.0",
}
TARGET = {
    "dataset_id": "abalone", "train_size": 3000, "query_size": 1000,
    "mechanism": "MAR", "dtype": "float64", "seed": 303,
}
METHODS = {"knn": "KNNImputer", "faiss": "FaissImputer[available]"}
K = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "float64": float(value),
    }


def exact_distances(train_fractions, query_row):
    """Exact squared distances for the represented binary64 inputs."""
    query = [
        None if np.isnan(value) else Fraction.from_float(float(value))
        for value in query_row
    ]
    distances = []
    for donor in train_fractions:
        pairs = [
            (left, right) for left, right in zip(donor, query)
            if left is not None and right is not None
        ]
        require(pairs, "Reference requires at least one shared feature")
        distance = sum(((left - right) ** 2 for left, right in pairs), Fraction(0))
        distances.append(distance * len(query) / len(pairs))
    return distances


def reference_cell(train, column, distances):
    eligible = np.flatnonzero(~np.isnan(train[:, column])).tolist()
    ordered = sorted(eligible, key=lambda index: (distances[index], index))
    k = min(K, len(ordered))
    require(k == K, "Expected five eligible neighbors")
    cutoff = distances[ordered[k - 1]]
    closer = [index for index in eligible if distances[index] < cutoff]
    tied = [index for index in eligible if distances[index] == cutoff]
    slots = k - len(closer)
    fixed = sum((Fraction.from_float(float(train[i, column])) for i in closer), Fraction(0))
    values = sorted(Fraction.from_float(float(train[i, column])) for i in tied)
    return {
        "eligible_count": len(eligible),
        "representative_ids": ordered[:k],
        "tie_rule": "Training-row order is one representative, not the only valid tie choice.",
        "cutoff_squared_distance": rational(cutoff),
        "next_squared_gap": rational(distances[ordered[k]] - cutoff) if len(ordered) > k else None,
        "strictly_closer_ids": closer,
        "exact_boundary_tie_ids": tied,
        "boundary_slots": slots,
        "minimum_exact_mean": rational((fixed + sum(values[:slots], Fraction(0))) / k),
        "maximum_exact_mean": rational((fixed + sum(values[-slots:], Fraction(0))) / k),
    }


def selection_details(train, column, ids, distances, reference, observed_output):
    ids = [int(index) for index in ids]
    require(len(ids) == K and len(set(ids)) == K, "Expected five distinct donors")
    require(all(0 <= index < len(train) for index in ids), "Invalid donor index")
    require(np.isfinite(train[ids, column]).all(), "Donor lacks target value")
    cutoff = Fraction(
        int(reference["cutoff_squared_distance"]["numerator"]),
        int(reference["cutoff_squared_distance"]["denominator"]),
    )
    omitted = sorted(set(reference["strictly_closer_ids"]) - set(ids))
    farther = [index for index in ids if distances[index] > cutoff]
    mean = sum(
        (Fraction.from_float(float(train[index, column])) for index in ids),
        Fraction(0),
    ) / K
    return {
        "training_row_indices": ids,
        "target_values": train[ids, column],
        "exact_squared_distances": [rational(distances[index]) for index in ids],
        "exact_mean": rational(mean),
        "this_query_output_minus_rounded_exact_mean": float(observed_output - float(mean)),
        "admissible_exact_top_k": not omitted and not farther,
        "omitted_strictly_closer_ids": omitted,
        "selected_strictly_farther_ids": farther,
    }


def trace_knn_selections(model, train, query, wanted_rows):
    """Observe actual argpartition results inside sklearn's imputation call."""
    missing = np.isnan(query)
    missing_rows = np.flatnonzero(missing.any(axis=1))
    wanted = set(wanted_rows)
    captured, selections, active = {}, {}, {}
    original_pairwise = _knn.pairwise_distances_chunked
    original_calc = model._calc_impute
    original_partition = np.argpartition

    def traced_calc(dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        require(active.get("columns"), "Unexpected KNN column dispatch")
        column = active["columns"].pop(0)
        chunk_rows = active["rows"]
        receiver_positions = np.flatnonzero(missing[chunk_rows, column])
        receivers = chunk_rows[receiver_positions]
        eligible = np.flatnonzero(~np.isnan(train[:, column]))
        expected = active["distances"][receiver_positions][:, eligible]
        np.testing.assert_array_equal(dist_pot_donors, expected)
        np.testing.assert_array_equal(fit_X_col, train[eligible, column])
        np.testing.assert_array_equal(mask_fit_X_col, np.zeros(len(eligible), dtype=bool))
        require(n_neighbors == K and np.isfinite(expected).all(),
                "Unsupported KNN donor count or undefined distance")
        partitions = []

        def traced_partition(array, kth, axis=-1, **kwargs):
            result = original_partition(array, kth, axis=axis, **kwargs)
            if array is dist_pot_donors:
                require(kth == K - 1 and axis == 1, "Unexpected KNN partition")
                partitions.append(result[:, :K].copy())
            return result

        with patch.object(_knn.np, "argpartition", traced_partition):
            values = original_calc(
                dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col
            )
        require(len(partitions) == 1, "Expected one observed KNN partition")
        for position, row in enumerate(receivers):
            if int(row) in wanted:
                local_ids = partitions[0][position]
                cell = (int(row), column)
                require(cell not in selections, "KNN cell captured twice")
                selections[cell] = {
                    "eligible_count": len(eligible),
                    "eligible_local_indices": local_ids,
                    "training_row_indices": eligible[local_ids],
                    "captured_distances": dist_pot_donors[position, local_ids].copy(),
                    "returned_value": float(values[position]),
                }
        return values

    def traced_pairwise(X, Y=None, **kwargs):
        np.testing.assert_array_equal(X, query[missing_rows])
        np.testing.assert_array_equal(Y, train)
        reducer = kwargs.pop("reduce_func", None)
        require(reducer is not None, "Unexpected KNN distance interface")

        def traced_reducer(distances, start):
            require(not active, "Unexpected nested KNN reducer")
            rows = missing_rows[start:start + len(distances)]
            require(distances.shape == (len(rows), len(train)), "KNN chunk shape changed")
            active.update(
                rows=rows, distances=distances.copy(),
                columns=np.flatnonzero(missing[rows].any(axis=0)).tolist(),
            )
            for position, row in enumerate(rows):
                if int(row) in wanted:
                    require(int(row) not in captured, "KNN row captured twice")
                    captured[int(row)] = distances[position].copy()
            try:
                result = reducer(distances, start)
                require(not active["columns"], "KNN columns were not all observed")
                return result
            finally:
                active.clear()

        return original_pairwise(X, Y, reduce_func=traced_reducer, **kwargs)

    with patch.object(_knn, "pairwise_distances_chunked", traced_pairwise), patch.object(
        model, "_calc_impute", traced_calc
    ):
        output = model.transform(query)
    expected_cells = {
        (row, int(column)) for row in wanted_rows
        for column in np.flatnonzero(missing[row])
    }
    require(set(captured) == wanted and set(selections) == expected_cells,
            "Some requested KNN observations were not captured")
    for (row, column), selection in selections.items():
        require(selection["returned_value"] == output[row, column],
                "Captured KNN value differs from transform output")
    return output, captured, selections


def diagnose(args, report):
    check_released_package(args.expected_version)
    raw = args.baseline.read_bytes()
    require(sha256(raw).hexdigest() == BASELINE_SHA256, "Archived JSON checksum mismatch")
    baseline = json.loads(raw)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    require(provenance["source_commit"] == baseline["provenance"]["source_commit"] == SOURCE,
            "Expected the original benchmark library source")
    require(provenance["version"] == args.expected_version
            == baseline["provenance"]["version"], "Library version mismatch")
    require(sys.version.split()[0] == baseline["metadata"]["python"], "Python version mismatch")
    installed = {name: version(name) for name in DEPENDENCIES}
    require(installed == DEPENDENCIES, "Numerical dependency version mismatch")

    package = Path(inspect.getfile(type(make_model("faiss")))).parent
    core_hashes = {name: sha256((package / name).read_bytes()).hexdigest() for name in CORE_SHA256}
    require(core_hashes == CORE_SHA256, "Installed library code differs from archived wheel")
    report.update({
        "environment": metadata(), "provenance": provenance,
        "baseline_provenance": baseline["provenance"],
        "baseline_sha256": BASELINE_SHA256, "core_sha256": core_hashes,
        "dependencies": installed, "target": TARGET, "threshold": args.threshold,
        "knn_source_sha256": sha256(Path(inspect.getfile(KNNImputer)).read_bytes()).hexdigest(),
        "notes": [
            "No timing measurements are taken.",
            "The original binary64 prepared inputs are used without downcasting.",
            "Exact rational distances describe represented binary64 values, not ideal pre-scaling data.",
            "Eligibility requires an observed target and at least one shared observed feature.",
            "Exact boundary ties can admit multiple valid neighbor sets.",
            "Float64 direct distances and captured KNN distances have separate floating-point tie summaries.",
            "KNN IDs come from observed argpartition results, not a reconstructed selection.",
            "Faiss selections are reconstructed from actual finished search results.",
            "Duplicate query values may have multiple Faiss traces; ambiguous ordered selections stay separate.",
            "All row indices are zero-based in the prepared arrays.",
            "Threshold and max-rows limit detailed diagnostics, not archived output arrays.",
            "An output hash mismatch prevents claiming reproduction of the archived outputs.",
        ],
    })
    chosen = {}
    for label, method in METHODS.items():
        matches = [
            (index, row) for index, row in enumerate(baseline["records"])
            if row["method"] == method and row["repeat"] == 1
            and all(row.get(field) == value for field, value in TARGET.items())
        ]
        require(len(matches) == 1, "Archived target record is missing or duplicated")
        chosen[label] = matches[0]
        require(matches[0][1]["status"] == "ok"
                and matches[0][1]["checks_passed"] is True, "Archived worker failed")

    data, names, dataset = load_dataset(
        args.data_home, dataset_id="abalone", download_if_missing=False
    )
    require(dataset == baseline["dataset"], "Source dataset metadata mismatch")
    train, query, truth, missing, case = prepare_case(
        data, names, seed=303, mechanism="MAR", train_size=3000,
        query_size=1000, dtype="float64", missing_rate=0.1,
        mar_reference_rows=1000, mar_driver="Length",
    )
    for _, record in chosen.values():
        require(case["fingerprints"] == record["case"]["fingerprints"],
                "Prepared inputs differ from the archived case")
    require(not np.isnan(train[:, 0]).any() and not np.isnan(query[:, 0]).any(),
            "Expected an always-observed shared Length feature")
    report.update(dataset=dataset, case=case, inputs_reproduced=True,
                  baseline_record_indices={label: pair[0] for label, pair in chosen.items()})
    input_hashes = (array_digest(train), array_digest(query))
    models, outputs = {}, {}
    for label in METHODS:
        model = make_model(label)
        expected_parameters = chosen[label][1]["model_parameters"]
        require(all(model.get_params()[name] == value for name, value in expected_parameters.items()),
                "Model configuration differs from the benchmark")
        models[label] = model.fit(train)
        outputs[label] = model.transform(query)
        require(outputs[label].shape == query.shape and outputs[label].dtype == np.float64
                and np.isfinite(outputs[label]).all(), "Invalid model output")
        np.testing.assert_array_equal(outputs[label][~missing], query[~missing])
        require((array_digest(train), array_digest(query)) == input_hashes, "Input was modified")

    report["output_sha256"] = {label: array_digest(value) for label, value in outputs.items()}
    report["baseline_output_hashes_match"] = {
        label: report["output_sha256"][label] == pair[1]["output_sha256"]
        for label, pair in chosen.items()
    }
    report["original_reproduced"] = all(report["baseline_output_hashes_match"].values())
    difference = np.abs(outputs["faiss"] - outputs["knn"])
    row_maxima = np.where(missing, difference, 0.0).max(axis=1)
    affected = np.flatnonzero(row_maxima > args.threshold)
    ordered = affected[np.argsort(-row_maxima[affected], kind="stable")]
    wanted = [int(row) for row in ordered[:args.max_rows]]
    report.update({
        "max_abs_output_difference": float(difference[missing].max()),
        "cells_above_threshold": int((difference[missing] > args.threshold).sum()),
        "affected_query_rows": len(affected), "detailed_query_rows": wanted,
        "details_truncated": len(affected) > len(wanted), "rows": [],
        "quality": {},
    })
    for label, output in outputs.items():
        errors = output[missing] - truth[missing]
        report["quality"][label] = {
            "scored_cells": int(missing.sum()),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae": float(np.mean(np.abs(errors))),
        }

    arrays = {"train": train, "query": query, "truth": truth, "missing": missing,
              "knn_output": outputs["knn"], "faiss_output": outputs["faiss"]}
    arrays_path = args.output.with_suffix(".npz")
    np.savez_compressed(arrays_path, **arrays)
    report["arrays"] = {
        "file": arrays_path.name, "sha256": sha256(arrays_path.read_bytes()).hexdigest(),
    }
    captured, selections, searches, matching_rows = {}, {}, {}, {}
    if wanted:
        traced_knn, captured, selections = trace_knn_selections(models["knn"], train, query, wanted)
        traced_faiss, searches, matching_rows = trace_faiss_searches(models["faiss"], train, query, wanted)
        np.testing.assert_array_equal(traced_knn, outputs["knn"])
        np.testing.assert_array_equal(traced_faiss, outputs["faiss"])
        require((array_digest(train), array_digest(query)) == input_hashes, "Tracing modified inputs")
        report["traced_outputs_unchanged"] = True
        arrays["traced_query_rows"] = np.asarray(wanted, dtype=np.int64)
        arrays["knn_distance_rows"] = np.stack([captured[row] for row in wanted])
        np.savez_compressed(arrays_path, **arrays)
        report["arrays"]["sha256"] = sha256(arrays_path.read_bytes()).hexdigest()

    fractions = [
        [None if np.isnan(value) else Fraction.from_float(float(value)) for value in row]
        for row in train
    ] if wanted else []
    for row in wanted:
        exact = exact_distances(fractions, query[row])
        direct, counts = direct_squared_distances(train, query[row])
        query_key = array_digest(query[row])
        logs = searches[query_key]
        detail = {
            "query_row_index": row, "standardized_query": query[row],
            "max_abs_output_difference": float(row_maxima[row]),
            "query_rows_with_identical_values": matching_rows[query_key],
            "faiss_finished_searches": logs, "features": [],
        }
        for column in np.flatnonzero(missing[row]):
            column = int(column)
            reference = reference_cell(train, column, exact)
            actual = selections[(row, column)]
            eligible = np.flatnonzero(~np.isnan(train[:, column]))
            faiss_ids = [
                log["features"][str(column)]["training_row_indices"] for log in logs
            ]
            same_ids = all(np.array_equal(ids, faiss_ids[0]) for ids in faiss_ids)
            feature = {
                "feature_index": column, "feature_name": names[column],
                "ground_truth": float(truth[row, column]),
                "outputs": {label: float(output[row, column]) for label, output in outputs.items()},
                "exact_reference": reference,
                "float64_direct_boundary": boundary_details(direct, eligible, train, column),
                "knn_captured_boundary": boundary_details(captured[row], eligible, train, column),
                "knn_observation": {
                    **actual, **selection_details(train, column, actual["training_row_indices"],
                                                  exact, reference, outputs["knn"][row, column]),
                },
                "faiss_query_mapping_ambiguous": not same_ids,
                "faiss_selections": [
                    selection_details(train, column, ids, exact, reference, outputs["faiss"][row, column])
                    for ids in faiss_ids
                ],
            }
            candidate_ids = set(reference["representative_ids"])
            candidate_ids.update(reference["exact_boundary_tie_ids"])
            candidate_ids.update(actual["training_row_indices"].tolist())
            for ids in faiss_ids:
                candidate_ids.update(ids.tolist())
            feature["donor_details"] = [{
                "training_row_index": index,
                "standardized_values": train[index],
                "shared_feature_count": int(counts[index]),
                "exact_squared_distance": rational(exact[index]),
                "float64_direct_squared_distance": float(direct[index]),
                "knn_captured_distance": float(captured[row][index]),
            } for index in sorted(candidate_ids)]
            detail["features"].append(feature)
        report["rows"].append(detail)
    report["threadpools"] = threadpool_info()
    require(all(pool["num_threads"] == 1 for pool in report["threadpools"]),
            "A native thread pool exceeded one thread")
    report["status"] = "ok" if report["original_reproduced"] else "baseline_output_mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=(
        ROOT / "benchmarks/results/real-data-datasets-ef04b1b/abalone.json"
    ))
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "benchmark_outputs/abalone_output_diagnostic.json"
    ))
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.max_rows < 1:
        parser.error("Use a positive finite threshold and positive max-rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "error", "original_reproduced": False, "inputs_reproduced": False}
    try:
        with threadpool_limits(limits=1), config_context(working_memory=WORKING_MEMORY_MIB):
            faiss.omp_set_num_threads(1)
            diagnose(args, report)
    except Exception as error:
        report.update(status="error", original_reproduced=False,
                      error=str(error), traceback=traceback.format_exc())
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Status: {report['status']}", flush=True)
    print(f"Results: {args.output}", flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())