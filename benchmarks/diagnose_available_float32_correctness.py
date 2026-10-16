"""Trace twelve archived float32 disagreements on GitHub without changing acceptance."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gc
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303)
DTYPES = ("float32", "float64")
METHODS = ("faiss", "knn")
TARGETS = {
    101: {(697, 19), (1299, 15), (1306, 15), (2074, 9), (2353, 16)},
    202: {(213, 7), (1763, 17), (1811, 15), (2922, 1)},
    303: {(265, 18), (866, 16), (2683, 11)},
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve_worker_directory(output):
    """Reserve a new evidence directory before writing any worker logs or arrays."""
    directory = Path(output).with_name(Path(output).stem + "-workers")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def exact_mean(values):
    """Exact rational mean of the represented input values, not a float sum."""
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("Exact mean requires a nonempty list of finite values")
    return sum((Fraction.from_float(value) for value in numbers), Fraction()) / len(numbers)


def same_donor_multiset(left, right):
    return Counter(left) == Counter(right)


def classify_cell(*, archive_reproduced, trace_matches, capture_complete,
                  same_donors, faithful_reducer_replay):
    if not archive_reproduced:
        return "current_execution_only"
    if not trace_matches:
        return "trace_output_mismatch"
    if not capture_complete:
        return "trace_incomplete"
    if not faithful_reducer_replay:
        return "reducer_replay_mismatch"
    if not same_donors:
        return "different_selected_donors"
    return "same_donors_aggregation_difference"


def validate_manifest(manifest):
    expected = {
        "schema_version": 1, "source_run_id": "38638261596", "source_run_attempt": 1,
        "diagnostic_id": "available_float32_uniform_12_cells_38638261596_v1",
        "spec_revision": "validation_0.3.22_v3", "seeds": list(SEEDS),
        "donor_policy": "available", "weights": "uniform", "n_neighbors": 5,
        "target_dtype": "float32", "train_rows": 20000, "query_rows": 3000,
        "features": 20, "process_repeats": 1, "threads": 1, "working_memory_mib": 256,
        "worker_timeout_seconds": 180, "total_worker_budget_seconds": 600,
        "performance_measurements": False, "automatic_retry": False,
        "baseline_order": [f"{method} {dtype} calls 1-6" for method in METHODS for dtype in DTYPES],
        "trace_order": [f"{method} float32 full query batch" for method in METHODS],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen diagnostic field changed: {key}")
    if manifest["criterion"] != {
        "criterion_id": "knn_relative_3e-6_uniform_absolute_0_v1", "rtol": 3e-6, "atol": 0.0,
    }:
        raise ValueError("The parity criterion must not change")
    if len(manifest["targets"]) != 12:
        raise ValueError("Expected exactly twelve fixed cells")
    for seed in SEEDS:
        rows = [item for item in manifest["targets"] if item["seed"] == seed]
        if len(rows) != len(TARGETS[seed]) or {tuple(item["index"]) for item in rows} != TARGETS[seed]:
            raise ValueError("The target cells changed")
    if [item["seed"] for item in manifest["baselines"]] != list(SEEDS):
        raise ValueError("Baseline seed order changed")


def load_context(spec, manifest):
    # Resolve published modules before adding repository benchmark helpers.
    from faiss_imputer import FaissImputer
    from sklearn.impute import KNNImputer

    for distribution, expected in manifest["package_versions"].items():
        if version(distribution) != expected:
            raise ValueError(f"Dependency differs from archived verification: {distribution}")
    for relative, expected in manifest["installed_source_sha256"].items():
        module = importlib.import_module(relative[:-3].replace("/", "."))
        origin = Path(module.__file__).resolve()
        if origin.is_relative_to(ROOT) or sha256(origin) != expected:
            raise ValueError(f"Published diagnostic hook source differs: {relative}")
    for relative, expected in manifest["repository_source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Original benchmark source differs: {relative}")
    sys.path.insert(0, str(ROOT))
    try:
        verifier = importlib.import_module("benchmarks.verify_benchmark_correctness")
    finally:
        sys.path.pop(0)
    if Path(verifier.__file__).resolve() != (ROOT / "benchmarks/verify_benchmark_correctness.py").resolve():
        raise ValueError("Unexpected verifier import origin")
    verifier.validate_spec(spec)
    helper = verifier.load_helper(spec)
    return verifier, helper, FaissImputer, KNNImputer


def array_layout(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "strides_bytes": list(array.strides), "c_contiguous": bool(array.flags.c_contiguous),
            "f_contiguous": bool(array.flags.f_contiguous)}


def finite_list(array):
    return [float(value) if math.isfinite(float(value)) else None for value in array]


class SelectedCellTrace:
    """Read selected published frames at return; never replace numeric operations."""

    def __init__(self, method, model, train, targets):
        import numpy as np

        self.method = method
        self.model = model
        self.targets = set(targets)
        self.donor_rows = np.flatnonzero(~np.isnan(train).all(axis=1))
        self.code = (model._aggregate if method == "faiss" else model._calc_impute).__func__.__code__
        self.parent_name = "_transform_available_batched" if method == "faiss" else "process_chunk"
        self.snapshots = []
        self.errors = []

    def global_trace(self, frame, event, argument):
        if event == "call" and frame.f_code is self.code and frame.f_back.f_code.co_name == self.parent_name:
            frame.f_trace_lines = False
            frame.f_trace_opcodes = False
            return self.return_trace
        return None

    def return_trace(self, frame, event, argument):
        if event == "return":
            try:
                self.capture(frame, argument)
            except Exception:
                # A capture failure must not change the imputer's computation.
                self.errors.append(traceback.format_exc())
        return self.return_trace

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        if self.previous_trace is not None:
            raise RuntimeError("Another Python trace is active")
        sys.settrace(self.global_trace)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.previous_trace)
        return False

    def capture(self, frame, returned):
        import numpy as np

        local, parent = frame.f_locals, frame.f_back.f_locals
        col = int(parent["col"])
        if self.method == "faiss":
            positions = np.flatnonzero(parent["fill_rows"])
            rows = parent["batch_rows"][positions]
        else:
            rows = parent["receivers_idx"]
        selected_rows = [(i, int(row)) for i, row in enumerate(rows) if (int(row), col) in self.targets]
        if not selected_rows:
            return
        if returned is None:
            raise ValueError("Target reducer did not return values")
        snapshot = {"returned": np.asarray(returned).copy(), "cells": []}
        if self.method == "faiss":
            if local["axis"] != 1 or local["ignore_nan"] is not True:
                raise ValueError("Unexpected uniform Faiss reduction")
            # Retain the actual full matrix and its layout; an extracted row
            # could follow a different floating-point reduction path.
            snapshot["values"] = local["values"]
            for position, query_row in selected_rows:
                batch_position = int(positions[position])
                slots = np.flatnonzero(parent["chosen"][batch_position])
                ids = parent["safe_ids"][batch_position, slots]
                snapshot["cells"].append({
                    "position": position, "index": [query_row, col],
                    "donor_ids": self.donor_rows[ids].tolist(),
                    "values": finite_list(local["values"][position, slots]),
                    "distances": finite_list(parent["distances"][batch_position, slots]),
                    "distance_representation": "Faiss returned squared missingness-scaled L2 distances",
                    "selected_slots": slots.tolist(),
                    "padded_values": finite_list(local["values"][position]),
                    "mask": [False] * len(slots), "weights": [1.0] * len(slots),
                })
        else:
            snapshot["donors"] = local["donors"]
            snapshot["weights"] = local["weight_matrix"]
            for position, query_row in selected_rows:
                ids = parent["potential_donors_idx"][local["donors_idx"][position]]
                snapshot["cells"].append({
                    "position": position, "index": [query_row, col],
                    "donor_ids": ids.tolist(),
                    "values": finite_list(local["donors"].data[position]),
                    "distances": finite_list(local["donors_dist"][position]),
                    "distance_representation": "sklearn returned nan_euclidean distances (not squared)",
                    "mask": np.ma.getmaskarray(local["donors"])[position].tolist(),
                    "weights": finite_list(local["weight_matrix"][position]),
                })
        self.snapshots.append(snapshot)

    def finish(self, output):
        import numpy as np

        records = []
        for snapshot in self.snapshots:
            if self.method == "faiss":
                replay = self.model._aggregate(snapshot["values"], axis=1, ignore_nan=True)
                layout = array_layout(snapshot["values"])
                primitive = "Faiss _aggregate: np.nanmean on the captured full padded matrix"
                weight_layout = None
            else:
                replay = np.ma.average(snapshot["donors"], axis=1, weights=snapshot["weights"]).data
                layout = array_layout(snapshot["donors"].data)
                primitive = "np.ma.average on the captured full donor and weight matrices"
                weight_layout = array_layout(snapshot["weights"])
            replay_equal = bool(np.array_equal(replay, snapshot["returned"], equal_nan=True))
            for item in snapshot["cells"]:
                position = item["position"]
                row, col = item["index"]
                reduction = float(snapshot["returned"][position])
                record = {
                    **item, "method": self.method, "reducer": primitive,
                    "input_layout": layout, "weight_layout": weight_layout,
                    "axis": 1, "returned_dtype": str(snapshot["returned"].dtype),
                    "reducer_output_before_assignment": reduction,
                    "final_output": float(output[row, col]),
                    "assignment_cast_matches_output": bool(np.float32(reduction) == output[row, col]),
                    "faithful_full_matrix_reducer_replay_equal": replay_equal,
                    "faithful_reducer_replay_value": float(replay[position]),
                }
                effective = [i for i, (value, masked, weight, distance) in enumerate(zip(
                    item["values"], item["mask"], item["weights"], item["distances"]
                )) if value is not None and not masked and weight is not None and weight != 0 and distance is not None]
                record["effective_donor_ids"] = [item["donor_ids"][i] for i in effective]
                record["effective_donor_values"] = [item["values"][i] for i in effective]
                record["effective_donor_count"] = len(effective)
                record["uniform_five_donors"] = len(effective) == 5 and all(item["weights"][i] == 1 for i in effective)
                if record["uniform_five_donors"]:
                    values = record["effective_donor_values"]
                    mean = exact_mean(values)
                    final_error = Fraction.from_float(record["final_output"]) - mean
                    record["exact_mean"] = {
                        "numerator": str(mean.numerator), "denominator": str(mean.denominator),
                        "float64_approximation": float(mean),
                        "final_error_numerator": str(final_error.numerator),
                        "final_error_denominator": str(final_error.denominator),
                        "final_absolute_error": float(abs(final_error)),
                    }
                    packed = np.asarray(values, dtype=np.float32)
                    order = np.argsort(np.asarray(record["effective_donor_ids"]), kind="stable")
                    record["counterfactuals_not_used_for_acceptance"] = {
                        "packed_observed_order_float32_mean": float(np.mean(packed)),
                        "packed_donor_id_order_float32_mean": float(np.mean(packed[order])),
                        "packed_float64_mean_cast_float32": float(np.float32(np.mean(packed, dtype=np.float64))),
                        "note": "Packing, reordering or changing accumulation precision may change the reduction path. These are descriptive arithmetic experiments, not faithful replays or modified benchmark outputs.",
                    }
                records.append(record)
        return records


def baseline_matches(block, expected):
    return (
        block["status"] == "passed"
        and block["fingerprints"] == expected["fingerprints"]
        and block["calls"][0]["output_sha256"] == expected["output_sha256"]
        and block["calls"][0]["output_value_sha256"] == expected["output_value_sha256"]
        and block["first_output_file_sha256"] == expected["first_output_file_sha256"]
        and block["missing_file_sha256"] == expected["missing_file_sha256"]
    )


def trace_method(verifier, helper, seed, method, baseline, output):
    import numpy as np

    train, query, truth, missing = helper.make_data(20000, 3000, seed, "random", training_policy="available", dtype="float32")
    before = query.copy()
    marks = {name: verifier.array_hash(value) for name, value in (
        ("train", train), ("query", query), ("truth", truth), ("missing", missing),
    )}
    if marks != baseline["fingerprints"]:
        raise ValueError("Trace inputs differ from untraced baseline")
    warm_train = np.random.default_rng(7).normal(size=(32, 20)).astype(np.float32)
    warm_query = warm_train[:8].copy()
    warm_query[:, :4] = np.nan
    verifier.new_model(method, "available").fit(warm_train).transform(warm_query)
    gc.collect()
    model = verifier.new_model(method, "available").fit(train)
    gc.collect()
    with SelectedCellTrace(method, model, train, TARGETS[seed]) as trace:
        transformed = model.transform(query)
    checks = verifier.structural_checks(transformed, query, before, missing, train, marks["train"])
    baseline_output = np.load(output.parent / baseline["first_output_file"], allow_pickle=False)
    equal = bool(np.array_equal(transformed, baseline_output))
    output_path = output.with_name(f"{output.stem}-{method}-traced-float32.npy")
    np.save(output_path, transformed, allow_pickle=False)
    cells = trace.finish(transformed)
    captured = [tuple(cell["index"]) for cell in cells]
    coverage = len(captured) == len(TARGETS[seed]) and set(captured) == TARGETS[seed] and not trace.errors
    return {
        "method": method, "fingerprints": marks, "checks": checks,
        "full_output_equal_to_untraced": equal,
        "output_sha256": verifier.array_hash(transformed),
        "output_file": output_path.name, "output_file_sha256": sha256(output_path),
        "capture_complete": coverage, "capture_errors": trace.errors,
        "fallback_or_unobserved_cells": [list(key) for key in sorted(TARGETS[seed] - set(captured))],
        "fallback_note": "A missing hook capture is not proof of a fallback. Preserve it as incomplete evidence rather than inventing donor identities.",
        "cells": cells,
    }


def compare_cells(seed, archive_reproduced, traces):
    comparisons = []
    for row, col in sorted(TARGETS[seed]):
        selected = {}
        for trace in traces:
            matches = [cell for cell in trace["cells"] if cell["index"] == [row, col]]
            if len(matches) == 1:
                selected[trace["method"]] = matches[0]
        complete = len(selected) == 2 and all(trace["capture_complete"] for trace in traces)
        trace_matches = len(traces) == 2 and all(
            trace["full_output_equal_to_untraced"] and all(trace["checks"].values()) for trace in traces
        )
        same = faithful = False
        details = {}
        if complete:
            left, right = selected["faiss"], selected["knn"]
            same = same_donor_multiset(left["effective_donor_ids"], right["effective_donor_ids"])
            values_match = sorted(zip(left["effective_donor_ids"], left["effective_donor_values"])) == sorted(zip(right["effective_donor_ids"], right["effective_donor_values"]))
            faithful = all(cell["faithful_full_matrix_reducer_replay_equal"] and cell["assignment_cast_matches_output"] for cell in (left, right))
            complete = complete and left["uniform_five_donors"] and right["uniform_five_donors"]
            if same and not values_match:
                complete = False
            details = {
                "same_donor_multiset": same,
                "same_donor_order": left["effective_donor_ids"] == right["effective_donor_ids"],
                "same_values_by_donor_id": values_match,
                "faiss_final_output": left["final_output"], "knn_final_output": right["final_output"],
                "absolute_difference": abs(left["final_output"] - right["final_output"]),
                "frozen_allowed_difference": 3e-6 * abs(right["final_output"]),
                "frozen_criterion_passed": abs(left["final_output"] - right["final_output"]) <= 3e-6 * abs(right["final_output"]),
                "exact_mean_error_scope": "Each method is compared to the exact rational mean of its own selected represented values. Same-donor comparisons additionally establish a shared mathematical mean; this does not change the sklearn parity criterion.",
            }
        comparisons.append({
            "seed": seed, "index": [row, col],
            "classification": classify_cell(archive_reproduced=archive_reproduced, trace_matches=trace_matches,
                capture_complete=complete, same_donors=same, faithful_reducer_replay=faithful),
            **details,
        })
    return comparisons


def worker(spec, manifest, seed, output):
    result = {"seed": seed, "status": "running", "archived_correctness_status": "failed",
              "performance_gate_status": "not_evaluated", "baselines": [], "traces": []}
    save_json(output, result)
    try:
        verifier, helper, _, _ = load_context(spec, manifest)
        import faiss
        from sklearn import config_context
        from threadpoolctl import threadpool_info, threadpool_limits

        result["environment"] = helper.metadata()
        result["source_sha256"] = {"benchmarks/diagnose_available_float32_correctness.py": sha256(__file__),
                                   **manifest["repository_source_sha256"], **manifest["installed_source_sha256"]}
        expected = next(item for item in manifest["baselines"] if item["seed"] == seed)
        faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1), config_context(working_memory=256):
            result["threadpools"] = threadpool_info()
            result["faiss_omp_threads"] = int(faiss.omp_get_max_threads())
            if result["faiss_omp_threads"] != 1 or any(pool.get("num_threads") != 1 for pool in result["threadpools"]):
                raise ValueError("Native thread limits did not take effect")
            for method in METHODS:
                for dtype in DTYPES:
                    block = verifier.run_block(helper, {"donor_policy": "available", "seed": seed, "repeat": 1}, method, dtype, output)
                    reference = next(b for b in expected["blocks"] if b["method"] == method and b["dtype"] == dtype)
                    block["matches_archived_baseline"] = baseline_matches(block, reference)
                    result["baselines"].append(block)
                    save_json(output, result)
                    gc.collect()
            result["archive_reproduced"] = all(block["matches_archived_baseline"] for block in result["baselines"])
            for method in METHODS:
                baseline = next(b for b in result["baselines"] if b["method"] == method and b["dtype"] == "float32")
                result["traces"].append(trace_method(verifier, helper, seed, method, baseline, output))
                save_json(output, result)
                gc.collect()
        result["comparisons"] = compare_cells(seed, result["archive_reproduced"], result["traces"])
        categories = {cell["classification"] for cell in result["comparisons"]}
        if not result["archive_reproduced"]:
            result["status"] = "baseline_output_mismatch"
        elif "trace_output_mismatch" in categories:
            result["status"] = "trace_output_mismatch"
        elif categories - {"same_donors_aggregation_difference", "different_selected_donors"}:
            result["status"] = "trace_incomplete"
        else:
            result["status"] = "completed"
        result["completion_note"] = "Completed means the diagnostic evidence is available; the archived correctness failure is unchanged."
        result["process_peak_rss_mib"] = helper.peak_rss_mib()
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    save_json(output, result)
    return result


def orchestrate(spec_path, targets_path, manifest, output):
    worker_directory = reserve_worker_directory(output)
    result = {"status": "running", "manifest": manifest, "manifest_sha256": sha256(targets_path),
              "spec_sha256": sha256(spec_path), "expected_seeds": list(SEEDS), "records": [],
              "worker_directory": worker_directory.name,
              "archived_correctness_status": "failed", "performance_gate_status": "not_evaluated"}
    save_json(output, result)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    started = time.monotonic()
    for seed in SEEDS:
        remaining = 600 - (time.monotonic() - started)
        worker_output = worker_directory / f"seed{seed}.json"
        if remaining <= 0:
            record = {"seed": seed, "status": "resource_bounded", "completion_status": "not_executed"}
        else:
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--spec", str(spec_path),
                       "--targets", str(targets_path), "--output", str(worker_output), "--worker-seed", str(seed)]
            timed_out = False
            returncode = None
            before = time.monotonic()
            with (worker_directory / f"seed{seed}.log").open("x", encoding="utf-8") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=min(180, remaining), check=False)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            try:
                record = json.loads(worker_output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {"seed": seed, "status": "process_error", "reason": "missing_worker_output"}
            record["worker_wall_seconds"] = time.monotonic() - before
            record["returncode"] = returncode
            if timed_out:
                record.update(status_before_timeout=record["status"], status="resource_bounded", completion_status="not_completed")
            elif record.get("seed") != seed or returncode != int(record.get("status") != "completed"):
                record.update(status="process_error", reason="worker_identity_or_exit_mismatch")
        result["records"].append(record)
        save_json(output, result)
        print(f"seed{seed}: {record['status']}", flush=True)
    result["status"] = "completed" if all(r["status"] == "completed" for r in result["records"]) else "incomplete_or_mismatch"
    result["classification_counts"] = dict(Counter(cell["classification"] for r in result["records"] for cell in r.get("comparisons", [])))
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec_path, targets_path, output = args.spec.resolve(), args.targets.resolve(), args.output.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(targets_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if sha256(spec_path) != manifest["spec_sha256"]:
        parser.error("The frozen v3 specification differs from the archived run")
    if output.exists():
        parser.error("Output exists; preserve earlier diagnostic evidence")
    result = worker(spec, manifest, args.worker_seed, output) if args.worker_seed else orchestrate(spec_path, targets_path, manifest, output)
    print(f"Diagnostic status: {result['status']}; archived correctness status: failed (unchanged)", flush=True)
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
