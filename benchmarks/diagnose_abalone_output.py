"""Diagnose archived Abalone outputs without timing measurements."""

import argparse
from email.parser import BytesParser
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch
from zipfile import ZipFile

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

RELEASE_VERSION = "0.3.22"
RELEASE_SOURCE = "4acd09dfa1aca339f38d401bdb2a1076bf3abe8c"
RELEASE_RUN = "37099197073"
RELEASE_ARCHIVE_SHA256 = (
    "8d2e491b9f066312627e5ea8ad6b955269c1634e7ae059b71d1aecac8579b638"
)
RELEASE_JSON_SHA256 = {
    "float32": "50d92bbc54652408795939f377f862a191a1a4919bd27418e3caf9f76b347bea",
    "float64": "d4e5814a2fd5b773f0d268e940fddeccc67fdf0e459edd2dc678a43a94d089ef",
}
RELEASE_DEPENDENCIES = {
    **DEPENDENCIES,
    "cloudpickle": "3.1.2", "narwhals": "2.26.0", "packaging": "26.3",
}
CASES = ("historical", "released_float32", "released_float64")


def diagnostic_profile(name):
    if name == "historical":
        return {
            "name": name, "published": False,
            "target": dict(TARGET), "baseline_sha256": BASELINE_SHA256,
        }
    if name not in CASES:
        raise ValueError(f"Unknown Abalone diagnostic case: {name}")
    dtype = name.removeprefix("released_")
    return {
        "name": name, "published": True,
        "target": {**TARGET, "mechanism": "MCAR", "dtype": dtype, "seed": 101},
        "baseline_sha256": RELEASE_JSON_SHA256[dtype],
    }


def read_baseline(path, profile):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == profile["baseline_sha256"],
            "Archived JSON checksum mismatch")
    return json.loads(raw)


def select_records(baseline, profile):
    target = profile["target"]
    if profile["published"]:
        require(baseline.get("schema_version") == 1
                and baseline.get("benchmark") == "released_real_data"
                and baseline.get("complete") is True,
                "Expected a complete released real-data benchmark")
        environment = baseline["metadata"]
        require(environment["git_commit"] == RELEASE_SOURCE
                and environment["github_run_id"] == RELEASE_RUN
                and environment["github_run_attempt"] == "1",
                "Unexpected published benchmark provenance")
        parameters = baseline["parameters"]
        require(parameters["previous_version"] == "0.3.21"
                and parameters["current_version"] == RELEASE_VERSION
                and parameters["dtype"] == target["dtype"]
                and parameters["dataset_id"] == "abalone",
                "Unexpected published benchmark configuration")
    chosen = {}
    for label, method in METHODS.items():
        variant = "current" if label == "faiss" else "knn"
        matches = [
            (index, row) for index, row in enumerate(baseline["records"])
            if row["method"] == method and row["repeat"] == 1
            and all(row.get(field) == value for field, value in target.items())
            and (not profile["published"] or row.get("variant") == variant)
        ]
        require(len(matches) == 1, "Archived target record is missing or duplicated")
        chosen[label] = matches[0]
        record = matches[0][1]
        require(record["status"] == "ok" and record["checks_passed"] is True,
                "Archived worker failed")
        if profile["published"]:
            expected = {
                "expected_version": RELEASE_VERSION,
                "api": "fit_then_transform", "training_policy": "available",
                "features": 7, "n_neighbors": K, "missing_rate": 0.1,
                "mar_reference_rows": 1000, "mar_driver": "Length", "threads": 1,
                "sklearn_working_memory_mib": 256,
                "input_dtype": target["dtype"], "output_dtype": target["dtype"],
            }
            require(all(record.get(key) == value for key, value in expected.items()),
                    "Archived worker configuration differs from the diagnostic")
            require(record["environment"]["faiss_imputer"] == RELEASE_VERSION,
                    "Archived worker used a different package release")
            require(len(record["imputed_values"]) == record["scored_cells"],
                    "Archived hidden-value count mismatch")
    if profile["published"]:
        require(chosen["knn"][1]["case"] == chosen["faiss"][1]["case"],
                "Archived methods used different prepared cases")
    return chosen


def verify_published_wheel(provenance_path, provenance, package):
    require(provenance.get("kind") == "published-wheel"
            and provenance.get("version") == RELEASE_VERSION
            and provenance.get("archive_sha256") == RELEASE_ARCHIVE_SHA256,
            "Expected published-wheel provenance for the preserved release run")
    filename = provenance.get("wheel_filename", "")
    require(filename and Path(filename).name == filename
            and "\\" not in filename and filename.endswith(".whl"),
            "Invalid wheel filename in provenance")
    wheel = provenance_path.parent / "wheels" / filename
    require(sha256(wheel.read_bytes()).hexdigest() == provenance["wheel_sha256"],
            "Published wheel checksum mismatch")
    with ZipFile(wheel) as archive:
        candidates = [name for name in archive.namelist()
                      if name.endswith(".dist-info/METADATA")]
        require(len(candidates) == 1, "Expected one wheel METADATA file")
        wheel_metadata = BytesParser().parsebytes(archive.read(candidates[0]))
        require(wheel_metadata["Name"] == "faiss-imputer"
                and wheel_metadata["Version"] == RELEASE_VERSION,
                "Downloaded wheel is not faiss-imputer 0.3.22")
        hashes = {}
        for name in CORE_SHA256:
            installed_bytes = (package / name).read_bytes()
            require(installed_bytes == archive.read(f"faiss_imputer/{name}"),
                    f"Installed core file differs from the published wheel: {name}")
            hashes[name] = sha256(installed_bytes).hexdigest()
    return hashes


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
    """Exact squared distances for the represented binary32 or binary64 inputs."""
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
        "this_query_output_minus_rounded_exact_mean": float(observed_output) - float(mean),
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
    profile = diagnostic_profile(getattr(args, "case", "historical"))
    target = profile["target"]
    baseline = read_baseline(args.baseline, profile)
    chosen = select_records(baseline, profile)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    require(sys.version.split()[0] == baseline["metadata"]["python"],
            "Python version mismatch")
    dependencies = RELEASE_DEPENDENCIES if profile["published"] else DEPENDENCIES
    installed = {name: version(name) for name in dependencies}
    require(installed == dependencies, "Numerical dependency version mismatch")
    package = Path(inspect.getfile(type(make_model("faiss")))).parent
    if profile["published"]:
        require(args.expected_version == RELEASE_VERSION,
                "The published diagnostic requires faiss-imputer 0.3.22")
        core_hashes = verify_published_wheel(args.provenance, provenance, package)
        baseline_provenance = {
            "benchmark_source_commit": RELEASE_SOURCE,
            "github_run_id": RELEASE_RUN, "github_run_attempt": "1",
            "package_version": RELEASE_VERSION,
            "archive_sha256": RELEASE_ARCHIVE_SHA256,
        }
    else:
        require(provenance["source_commit"]
                == baseline["provenance"]["source_commit"] == SOURCE,
                "Expected the original benchmark library source")
        require(provenance["version"] == args.expected_version
                == baseline["provenance"]["version"], "Library version mismatch")
        core_hashes = {
            name: sha256((package / name).read_bytes()).hexdigest()
            for name in CORE_SHA256
        }
        require(core_hashes == CORE_SHA256,
                "Installed library code differs from archived wheel")
        baseline_provenance = baseline["provenance"]
    report.update({
        "environment": metadata(), "provenance": provenance,
        "baseline_provenance": baseline_provenance,
        "baseline_sha256": profile["baseline_sha256"], "core_sha256": core_hashes,
        "diagnostic_case": profile["name"],
        "dependencies": installed, "target": target, "threshold": args.threshold,
        "knn_source_sha256": sha256(Path(inspect.getfile(KNNImputer)).read_bytes()).hexdigest(),
        "notes": [
            "No timing measurements are taken.",
            f"Original {target['dtype']} prepared inputs are used without changing dtype.",
            "Exact rational distances describe represented input values, not ideal pre-scaling data.",
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
    if profile["published"]:
        report["notes"].extend([
            "The benchmark source commit identifies scripts, not the published library source.",
            "Installed core files are checked against the retained published wheel.",
            "The original run did not preserve its wheel hash; binary artifact identity is not claimed.",
            "Float32 output differences are computed after exact promotion to float64.",
        ])
    data, names, dataset = load_dataset(
        args.data_home, dataset_id="abalone", download_if_missing=False
    )
    require(dataset == baseline["dataset"], "Source dataset metadata mismatch")
    train, query, truth, missing, case = prepare_case(
        data, names, seed=target["seed"], mechanism=target["mechanism"],
        train_size=target["train_size"], query_size=target["query_size"],
        dtype=target["dtype"], missing_rate=0.1,
        mar_reference_rows=1000, mar_driver="Length",
    )
    for _, record in chosen.values():
        if profile["published"]:
            require(case == record["case"],
                    "Prepared inputs or metadata differ from the archived case")
        else:
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
        require(outputs[label].shape == query.shape and outputs[label].dtype == np.dtype(target["dtype"])
                and np.isfinite(outputs[label]).all(), "Invalid model output")
        np.testing.assert_array_equal(outputs[label][~missing], query[~missing])
        require((array_digest(train), array_digest(query)) == input_hashes, "Input was modified")

    report["output_sha256"] = {label: array_digest(value) for label, value in outputs.items()}
    report["baseline_output_hashes_match"] = {
        label: report["output_sha256"][label] == pair[1]["output_sha256"]
        for label, pair in chosen.items()
    }
    report["original_reproduced"] = all(report["baseline_output_hashes_match"].values())
    if profile["published"]:
        report["baseline_imputed_values_match"] = {
            label: np.array_equal(
                outputs[label][missing].astype(np.float64),
                np.asarray(record["imputed_values"], dtype=np.float64),
            )
            for label, (_, record) in chosen.items()
        }
        report["original_reproduced"] = (
            report["original_reproduced"]
            and all(report["baseline_imputed_values_match"].values())
        )
        archived_difference = np.abs(
            np.asarray(chosen["faiss"][1]["imputed_values"], dtype=np.float64)
            - np.asarray(chosen["knn"][1]["imputed_values"], dtype=np.float64)
        )
        report["archived_output_comparison"] = {
            "scored_cells": int(archived_difference.size),
            "max_abs_difference": float(archived_difference.max()),
            "cells_above_threshold": int((archived_difference > args.threshold).sum()),
        }
    difference = np.abs(
        outputs["faiss"].astype(np.float64) - outputs["knn"].astype(np.float64)
    )
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
    if profile["published"] and not report["original_reproduced"]:
        report["status"] = "baseline_output_mismatch"
        report["notes"].append(
            "Tracing was skipped because the archived outputs were not reproduced."
        )
        return
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
    parser.add_argument("--case", choices=CASES, default="historical")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = diagnostic_profile(args.case)
    if args.baseline is None:
        if profile["published"]:
            parser.error("--baseline must identify the extracted published JSON")
        args.baseline = ROOT / "benchmarks/results/real-data-datasets-ef04b1b/abalone.json"
    if args.output is None:
        filename = (
            f"abalone_released_0.3.22_{profile['target']['dtype']}_diagnostic.json"
            if profile["published"] else "abalone_output_diagnostic.json"
        )
        args.output = ROOT / "benchmark_outputs" / filename
    protected = {args.baseline.resolve(), args.provenance.resolve()}
    if {args.output.resolve(), args.output.with_suffix(".npz").resolve()} & protected:
        parser.error("Diagnostic output must not overwrite input evidence")
    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.max_rows < 1:
        parser.error("Use a positive finite threshold and positive max-rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "error", "original_reproduced": False, "inputs_reproduced": False}
    try:
        memory_mib = 256 if profile["published"] else WORKING_MEMORY_MIB
        with threadpool_limits(limits=1), config_context(working_memory=memory_mib):
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
