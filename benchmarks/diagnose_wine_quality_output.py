"""Diagnose archived Wine Quality float32 outputs without timing measurements."""

import argparse
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys
import traceback

import faiss
import numpy as np
from sklearn import config_context
from sklearn.impute import KNNImputer
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.benchmark_real_data_cases import (
    array_digest, load_dataset, prepare_case,
)
from benchmarks.benchmark_scaling_threads import (
    ROOT, WORKING_MEMORY_MIB, check_released_package, metadata,
)
from benchmarks import diagnose_abalone_output as shared_diagnostic
from benchmarks.diagnose_abalone_output import (
    CORE_SHA256, DEPENDENCIES, METHODS, SOURCE,
    exact_distances, rational, reference_cell, require,
    selection_details, trace_knn_selections,
)
from benchmarks.diagnose_real_data_float32 import (
    boundary_details, direct_squared_distances, json_safe,
    make_model, trace_faiss_searches,
)


BASELINE_SHA256 = "2c4d748590efa26607cf8f6258de0b20cd58fe0bd90247a07363cc6598aec5f1"
TARGET = {
    "dataset_id": "wine_quality_white", "train_size": 3000, "query_size": 1000,
    "mechanism": "MCAR", "dtype": "float32", "seed": 101,
}


RELEASE_PROFILES = {
    "uniform": {
        "baseline_sha256": "e4b16d9794a944afc8e63dceffe2cb04a97132ff7f7a7228be562999757a7ece",
        "archive_sha256": "0654259363cd7c1d76f467fc06e5c3fd104f24198f9a7ef865569bdd2c4833bd",
        "source": "6d533d37c8173058ce2cfb314701377ee67b4b59",
        "run": "37403070692",
    },
    "distance": {
        "baseline_sha256": "d78ed2d9f265be1363f57af372fa16b8ff65b2680ca0c7ba634253ba2aa30e9c",
        "archive_sha256": "f6f3771b1385624306d7a28b22cbaff08383b5ed89a80c73c78e33b496291ddd",
        "source": "6d533d37c8173058ce2cfb314701377ee67b4b59",
        "run": "37403259216",
    },
}
CASES = (
    "historical", "released_uniform_float32_seed101",
    "released_distance_float32_seed101", "released_distance_float32_seed202",
    "released_distance_float32_seed303",
)


def diagnostic_profile(name):
    if name == "historical":
        return {"name": name, "published": False, "target": dict(TARGET), "weights": "uniform"}
    if name not in CASES:
        raise ValueError(f"Unknown Wine diagnostic case: {name}")
    weights = name.split("_")[1]
    seed = int(name.rsplit("seed", 1)[1])
    return {
        **RELEASE_PROFILES[weights],
        "name": name, "published": True, "weights": weights,
        "target": {**TARGET, "seed": seed},
        "features": 11, "mar_driver": "alcohol",
        "trace_current_on_mismatch": True,
    }


def diagnose(args, report):
    profile = diagnostic_profile(getattr(args, "case", "historical"))
    if not profile["published"]:
        return diagnose_historical(args, report)
    shared_diagnostic.diagnose(args, report, profile=profile)
    if profile["weights"] == "uniform":
        add_uniform_rounding_residuals(report)


def add_uniform_rounding_residuals(report):
    """Preserve signed float32 rounding residuals for each observed selection."""
    for row in report.get("rows", []):
        for feature in row["features"]:
            selections = [("knn", feature["knn_observation"])]
            selections.extend(("faiss", item) for item in feature["faiss_selections"])
            for label, selection in selections:
                mean = selection["exact_mean"]
                exact_mean = Fraction(int(mean["numerator"]), int(mean["denominator"]))
                selection["output_minus_exact_mean"] = rational(
                    Fraction.from_float(feature["outputs"][label]) - exact_mean
                )


def describe_selection(train, column, ids, distances, reference, observed_output):
    """Keep float32 output rounding separate from exact donor selection."""
    output = float(observed_output)
    details = selection_details(train, column, ids, distances, reference, output)
    mean = details["exact_mean"]
    exact_mean = Fraction(int(mean["numerator"]), int(mean["denominator"]))
    details["output_minus_exact_mean"] = rational(
        Fraction.from_float(output) - exact_mean
    )
    return details


def save_arrays(path, arrays, report):
    np.savez_compressed(path, **arrays)
    report["arrays"] = {
        "file": path.name, "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def diagnose_historical(args, report):
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
    core_hashes = {
        name: sha256((package / name).read_bytes()).hexdigest()
        for name in CORE_SHA256
    }
    require(core_hashes == CORE_SHA256, "Installed library code differs from archived wheel")
    report.update({
        "environment": metadata(), "provenance": provenance,
        "baseline_provenance": baseline["provenance"],
        "baseline_sha256": BASELINE_SHA256, "core_sha256": core_hashes,
        "dependencies": installed, "target": TARGET, "threshold": args.threshold,
        "knn_source_sha256": sha256(Path(inspect.getfile(KNNImputer)).read_bytes()).hexdigest(),
        "notes": [
            "No timing measurements are taken.",
            "Models and traces use the original float32 prepared inputs and outputs.",
            "Exact rational distances describe represented binary32 values, not ideal pre-scaling data.",
            "Output differences and reconstruction errors are calculated in float64.",
            "Eligibility requires an observed target and at least one shared observed feature.",
            "Exact boundary ties can admit multiple valid neighbor sets.",
            "Floating-point boundary equality is separate from exact rational equality.",
            "KNN captured distances are unsquared; direct and exact reference distances are squared.",
            "KNN IDs come from observed argpartition results, not a reconstructed selection.",
            "Faiss selections are reconstructed from actual finished search results.",
            "Duplicate query values may have multiple Faiss traces; ambiguous ordered selections stay separate.",
            "Mean residuals include float32 output rounding; exact signed residuals are also archived.",
            "The rounded-exact-mean residual uses a float64-rounded reference mean.",
            "Trace mean_float32 and mean_float64 fields are illustrative reductions, not captured aggregation steps.",
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
        args.data_home, dataset_id=TARGET["dataset_id"], download_if_missing=False
    )
    require(dataset == baseline["dataset"], "Source dataset metadata mismatch")
    train, query, truth, missing, case = prepare_case(
        data, names, seed=TARGET["seed"], mechanism=TARGET["mechanism"],
        train_size=TARGET["train_size"], query_size=TARGET["query_size"],
        dtype=TARGET["dtype"], missing_rate=0.1,
        mar_reference_rows=1000, mar_driver="alcohol",
    )
    for _, record in chosen.values():
        require(case["fingerprints"] == record["case"]["fingerprints"],
                "Prepared inputs differ from the archived case")
    require(train.dtype == query.dtype == np.dtype("float32")
            and truth.dtype == np.dtype("float64"), "Unexpected prepared dtypes")
    driver = names.index("alcohol")
    require(not np.isnan(train[:, driver]).any() and not np.isnan(query[:, driver]).any(),
            "Expected an always-observed shared alcohol feature")
    report.update(dataset=dataset, case=case, inputs_reproduced=True,
                  baseline_record_indices={label: pair[0] for label, pair in chosen.items()})

    input_hashes = (array_digest(train), array_digest(query))
    models, outputs = {}, {}
    for label in METHODS:
        model = make_model(label)
        expected_parameters = chosen[label][1]["model_parameters"]
        require(all(model.get_params()[name] == value
                    for name, value in expected_parameters.items()),
                "Model configuration differs from the benchmark")
        models[label] = model.fit(train)
        outputs[label] = model.transform(query)
        require(outputs[label].shape == query.shape
                and outputs[label].dtype == np.float32
                and np.isfinite(outputs[label]).all(), "Invalid model output")
        np.testing.assert_array_equal(outputs[label][~missing], query[~missing])
        require((array_digest(train), array_digest(query)) == input_hashes, "Input was modified")

    report["output_sha256"] = {
        label: array_digest(value) for label, value in outputs.items()
    }
    report["baseline_output_hashes_match"] = {
        label: report["output_sha256"][label] == pair[1]["output_sha256"]
        for label, pair in chosen.items()
    }
    report["original_reproduced"] = all(report["baseline_output_hashes_match"].values())
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
        errors = output[missing].astype(np.float64) - truth[missing]
        report["quality"][label] = {
            "scored_cells": int(missing.sum()),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae": float(np.mean(np.abs(errors))),
        }

    arrays = {
        "train": train, "query": query, "truth": truth, "missing": missing,
        "knn_output": outputs["knn"], "faiss_output": outputs["faiss"],
    }
    arrays_path = args.output.with_suffix(".npz")
    save_arrays(arrays_path, arrays, report)
    captured, selections, searches, matching_rows = {}, {}, {}, {}
    if wanted:
        traced_knn, captured, selections = trace_knn_selections(
            models["knn"], train, query, wanted
        )
        traced_faiss, searches, matching_rows = trace_faiss_searches(
            models["faiss"], train, query, wanted
        )
        np.testing.assert_array_equal(traced_knn, outputs["knn"])
        np.testing.assert_array_equal(traced_faiss, outputs["faiss"])
        require((array_digest(train), array_digest(query)) == input_hashes,
                "Tracing modified inputs")
        report["traced_outputs_unchanged"] = True
        arrays["traced_query_rows"] = np.asarray(wanted, dtype=np.int64)
        arrays["knn_distance_rows"] = np.stack([captured[row] for row in wanted])
        report["knn_captured_distance_dtype"] = str(arrays["knn_distance_rows"].dtype)
        save_arrays(arrays_path, arrays, report)

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
                "outputs": {
                    label: float(output[row, column]) for label, output in outputs.items()
                },
                "exact_reference": reference,
                "float64_direct_boundary": boundary_details(direct, eligible, train, column),
                "knn_captured_boundary": boundary_details(captured[row], eligible, train, column),
                "knn_observation": {
                    **actual,
                    **describe_selection(
                        train, column, actual["training_row_indices"],
                        exact, reference, outputs["knn"][row, column],
                    ),
                },
                "faiss_query_mapping_ambiguous": not same_ids,
                "faiss_selections": [
                    describe_selection(
                        train, column, ids, exact, reference, outputs["faiss"][row, column]
                    )
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
        args.baseline = ROOT / "benchmarks/results/real-data-datasets-ef04b1b/wine-quality-white.json"
    if args.output is None:
        filename = (
            f"wine_quality_released_0.3.22_float32_{profile['weights']}_seed{profile['target']['seed']}_diagnostic.json"
            if profile["published"] else "wine_quality_output_diagnostic.json"
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
    if report.get("tracing_completed") and not report["original_reproduced"]:
        print("Current-execution tracing completed; archived outputs were not reproduced.", flush=True)
    print(f"Results: {args.output}", flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
