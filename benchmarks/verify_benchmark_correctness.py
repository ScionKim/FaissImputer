"""Verify the frozen release benchmark criterion on GitHub, without timing imputers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DTYPES = ("float32", "float64")
POLICIES = ("complete", "available")
METHODS = ("faiss", "knn")
RTOL = 3e-6
ATOL = 0.0
CRITERION_ID = "knn_relative_3e-6_uniform_absolute_0_v1"
THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
HELPER_HASHES = {
    "benchmarks/benchmark_scaling_threads.py":
        "85c0126db9f3ff8c090e59fdea09f44893527ed90a4747e0df42f624a6a68a39",
    "benchmarks/benchmark_imputers.py":
        "3494f9afcad94ffecfb2cc3e2018bffbdb1b2191dea44f3d25aac72e2a7b753a",
}
VERIFICATION_PLAN = {
    "stage": "correctness_verification",
    "scope": "Synthetic preflight only; passing does not establish correctness at every scaling size or on real data.",
    "train_rows": 20000,
    "query_rows": 3000,
    "features": 20,
    "n_neighbors": 5,
    "weights": "uniform",
    "threads": 1,
    "working_memory_mib": 256,
    "policies_in_order": list(POLICIES),
    "seeds_in_order": [101, 202, 303],
    "repeats_per_seed": 3,
    "worker_count": 18,
    "fresh_sequential_workers": True,
    "worker_identity": ["donor_policy", "seed", "repeat"],
    "blocks_in_worker_order": [f"{method} {dtype} calls 1-6" for method in METHODS for dtype in DTYPES],
    "calls_per_block": 6,
    "dtype_boundary": "Release block arrays and model, then collect garbage; retain process imports and allocator state.",
    "reference_check": "Compare saved first outputs, with exact within-block repeat checks covering calls 2-6 for both methods.",
    "performance_measurements": False,
    "instrumented_replays": False,
    "worker_timeout_seconds": 300,
    "total_worker_budget_seconds": 1500,
    "budget_scope": "Full worker wall time. These preflight execution bounds do not replace the frozen 500k scaling budget.",
    "automatic_retry": False,
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    import numpy as np

    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def output_value_hash(array):
    """Hash finite output values with signed zeros treated as numerically equal."""
    import numpy as np

    normalized = np.ascontiguousarray(array).copy()
    normalized[normalized == 0] = 0
    return array_hash(normalized)


def validate_spec(spec):
    if spec.get("revision") != "validation_0.3.22_v3":
        raise ValueError("Expected the frozen validation_0.3.22_v3 specification")
    package = spec["package"]
    if (package["version"], package["source"], package["python_minor"]) != ("0.3.22", "published_wheel", "3.12"):
        raise ValueError("Published release configuration changed")
    if package["dependency_pins"] != {"numpy": "2.5.3", "scikit-learn": "1.9.1", "faiss-cpu": "1.15.1"}:
        raise ValueError("Frozen dependency versions changed")
    criterion = spec["benchmark_correctness_criterion"]
    acceptance = criterion["numerical_acceptance"]
    if (criterion["criterion_id"], criterion["default_rtol"], criterion["current_uniform_atol"]) != (CRITERION_ID, RTOL, ATOL):
        raise ValueError("Frozen correctness criterion changed")
    if (acceptance["rtol"], acceptance["atol"], acceptance["equal_nan"]) != (RTOL, ATOL, False):
        raise ValueError("Frozen numerical acceptance changed")
    if criterion["change_control"]["post_result_relaxation_allowed"] is not False:
        raise ValueError("Post-result tolerance relaxation is forbidden")
    plan = spec["scaling_plan"]
    for key, expected in {
        "query_rows": 3000, "features": 20, "n_neighbors": 5,
        "weights": "uniform", "threads": 1, "seeds": [101, 202, 303],
        "repeats_per_seed": 3,
    }.items():
        if plan[key] != expected:
            raise ValueError(f"Frozen workload changed: {key}")
    expected_series = [(policy, dtype) for policy in POLICIES for dtype in DTYPES]
    if [(s["donor_policy"], s["dtype"]) for s in plan["series"]] != expected_series:
        raise ValueError("Frozen series or dtype order changed")
    for series in plan["series"]:
        policy = series["donor_policy"]
        if (series["rtol"], series["atol"], series["correctness_criterion_id"]) != (RTOL, ATOL, CRITERION_ID):
            raise ValueError("A series changed the frozen tolerance")
        if series["workload_id"] != f"{policy}_random" or series["train_missing_fraction"] != (0.0 if policy == "complete" else 0.1):
            raise ValueError("A series changed donor policy or missingness")
    protocol = spec["frozen_measurement_protocol"]
    if protocol["dtype_block_order"] != list(DTYPES) or protocol["transform_calls_recorded"] != list(range(1, 7)):
        raise ValueError("Frozen call order changed")


def worker_schedule(spec):
    return [
        {"donor_policy": policy, "seed": seed, "repeat": repeat}
        for policy in POLICIES for seed in spec["scaling_plan"]["seeds"]
        for repeat in range(1, spec["scaling_plan"]["repeats_per_seed"] + 1)
    ]


def compare_outputs(actual, reference, missing, *, rtol=RTOL, atol=ATOL):
    """Use KNN as reference, score every missing cell, and retain violation evidence."""
    import numpy as np

    result = {
        "status": "failed", "rtol": rtol, "atol": atol,
        "compared_cells": 0, "violation_count": None,
        "max_abs_difference": None, "zero_reference_violation_count": None,
        "worst_violation": None,
    }
    if not (np.isfinite(rtol) and np.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("Tolerances must be finite and nonnegative")
    if actual.ndim != 2 or actual.shape != reference.shape or actual.shape != missing.shape or missing.dtype != np.dtype(bool):
        return {**result, "reason": "shape_or_mask_mismatch"}
    result["compared_cells"] = int(missing.sum())
    if not result["compared_cells"]:
        return {**result, "reason": "no_missing_cells"}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {**result, "reason": "nonfinite_output"}
    values = actual[missing].astype(np.float64)
    expected = reference[missing].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(values - expected)
        allowed = atol + rtol * np.abs(expected)
    if not np.isfinite(errors).all() or not np.isfinite(allowed).all():
        return {**result, "reason": "nonfinite_comparison_arithmetic"}
    violations = errors > allowed
    result.update({
        "status": "failed" if violations.any() else "passed",
        "violation_count": int(violations.sum()),
        "max_abs_difference": float(errors.max()),
        "zero_reference_violation_count": int(np.count_nonzero(violations & (expected == 0))),
    })
    if violations.any():
        position = int(np.argmax(np.where(violations, errors - allowed, -np.inf)))
        row, column = np.argwhere(missing)[position]
        result["worst_violation"] = {
            "index": [int(row), int(column)], "actual": float(values[position]),
            "reference": float(expected[position]), "allowed_error": float(allowed[position]),
            "absolute_error": float(errors[position]),
        }
    return result


def structural_checks(output, query, before, missing, train, train_sha256,
                      first_copy=None, retained_first=None):
    import numpy as np

    same_shape = output.shape == before.shape
    checks = {
        "shape_preserved": same_shape,
        "dtype_preserved": output.dtype == before.dtype,
        "finite": bool(np.isfinite(output).all()),
        "observed_values_preserved": bool(same_shape and np.array_equal(output[~missing], before[~missing])),
        "query_unchanged": bool(np.array_equal(query, before, equal_nan=True)),
        "training_unchanged": array_hash(train) == train_sha256,
        "separate_output": not np.shares_memory(output, query),
    }
    if first_copy is not None:
        checks["identical_to_call_1"] = bool(np.array_equal(output, first_copy, equal_nan=True))
        checks["retained_first_output_unchanged"] = bool(
            retained_first is not None and np.array_equal(retained_first, first_copy, equal_nan=True)
        )
    return {key: bool(value) for key, value in checks.items()}


def summarize_records(expected, records):
    def identity(record):
        return tuple(record.get(key) for key in ("donor_policy", "seed", "repeat"))

    expected_keys = [identity(item) for item in expected]
    received_keys = [identity(item) for item in records]
    result = {
        "status": "not_evaluated", "expected_workers": len(expected), "received_workers": len(records),
        "passed_workers": 0, "failed_workers": 0, "unevaluated_workers": 0,
    }
    if not expected or len(set(expected_keys)) != len(expected_keys) or len(set(received_keys)) != len(received_keys) or not set(received_keys).issubset(expected_keys):
        return {**result, "status": "failed", "reason": "invalid_or_duplicate_worker_identity"}
    for record in records:
        status = record.get("status")
        if record.get("observed_failure") or status in ("failed", "error", "process_error", "invalid_evidence", "input_mismatch"):
            result["failed_workers"] += 1
        elif status == "passed" and record.get("evidence_validated") is True:
            result["passed_workers"] += 1
        else:
            result["unevaluated_workers"] += 1
    result["unevaluated_workers"] += len(expected) - len(records)
    if result["failed_workers"]:
        result["status"] = "failed"
    elif result["passed_workers"] == len(expected):
        result["status"] = "passed"
    return result


def load_helper(spec):
    # Import the installed wheel before making repository helpers importable.
    import faiss_imputer

    origin = Path(faiss_imputer.__file__).resolve()
    if origin.is_relative_to(ROOT):
        raise RuntimeError("Imported checkout instead of published faiss-imputer")
    for name, expected in {"faiss-imputer": "0.3.22", **spec["package"]["dependency_pins"]}.items():
        if version(name) != expected:
            raise ValueError(f"Installed version mismatch: {name}")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the frozen Python 3.12 environment")
    for relative, expected in HELPER_HASHES.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Supplied data-generator source changed: {relative}")
    sys.path.insert(0, str(ROOT))
    try:
        helper = importlib.import_module("benchmarks.benchmark_scaling_threads")
    finally:
        sys.path.pop(0)
    if Path(helper.__file__).resolve() != (ROOT / "benchmarks/benchmark_scaling_threads.py").resolve():
        raise RuntimeError("Unexpected synthetic helper origin")
    helper.check_released_package("0.3.22")
    return helper


def new_model(method, policy):
    if method == "knn":
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean", copy=True)
    from faiss_imputer import FaissImputer

    return FaissImputer(n_neighbors=5, weights="uniform", metric="l2", strategy="mean",
                        index_factory="Flat", donor_policy=policy, copy=True)


def run_block(helper, identity, method, dtype, output):
    import numpy as np

    policy = identity["donor_policy"]
    train, query, truth, missing = helper.make_data(
        20000, 3000, identity["seed"], "random", training_policy=policy, dtype=dtype,
    )
    if train.shape != (20000, 20) or query.shape != (3000, 20) or truth.shape != query.shape:
        raise ValueError("Unexpected prepared case shape")
    if any(array.dtype != np.dtype(dtype) for array in (train, query, truth)):
        raise ValueError("Prepared case dtype changed")
    if not np.isfinite(truth).all() or np.isinf(train).any() or not np.array_equal(np.isnan(query), missing):
        raise ValueError("Invalid prepared values or missingness mask")
    if not np.all(missing.sum(axis=1) == 4) or not np.array_equal(query[~missing], truth[~missing]):
        raise ValueError("Held-out query preparation changed")
    if (policy == "complete" and not np.isfinite(train).all()) or (policy == "available" and not np.isnan(train).any()):
        raise ValueError("Donor-policy missingness changed")
    if np.any(np.isfinite(train).sum(axis=0) < 5):
        raise ValueError("Fewer than k observed donors in a training feature")
    before = query.copy()
    fingerprints = {name: array_hash(array) for name, array in (
        ("train", train), ("query", query), ("truth", truth), ("missing", missing),
    )}
    warm_train = np.random.default_rng(7).normal(size=(32, 20)).astype(dtype)
    warm_query = warm_train[:8].copy()
    warm_query[:, :4] = np.nan
    new_model(method, policy).fit(warm_train).transform(warm_query)
    gc.collect()
    model = new_model(method, policy).fit(train)
    gc.collect()
    first_output = model.transform(query)
    first_copy = first_output.copy()
    calls = []
    for number in range(1, 7):
        # Retain the previous repeat until the next transform returns.
        if number == 1:
            current = first_output
        else:
            current = model.transform(query)
        checks = structural_checks(
            current, query, before, missing, train, fingerprints["train"],
            first_copy=first_copy if number > 1 else None,
            retained_first=first_output if number > 1 else None,
        )
        calls.append({"call": number, "checks": checks, "output_sha256": array_hash(current),
                      "output_value_sha256": output_value_hash(current)})
    prefix = f"{output.stem}-{method}-{dtype}"
    output_path = output.with_name(prefix + "-call1.npy")
    mask_path = output.with_name(prefix + "-missing.npy")
    np.save(output_path, first_copy, allow_pickle=False)
    np.save(mask_path, missing, allow_pickle=False)
    series_id = f"{policy}_{'f32' if dtype == 'float32' else 'f64'}"
    return {
        "method": method, "dtype": dtype, "series_id": series_id,
        "workload_id": f"{policy}_random", "donor_policy": policy,
        "n_neighbors": 5, "weights": "uniform", "rtol": RTOL, "atol": ATOL,
        "status": "passed" if all(all(call["checks"].values()) for call in calls) else "failed",
        "fingerprints": fingerprints, "train_missing_fraction_observed": float(np.isnan(train).mean()),
        "calls": calls, "first_output_file": output_path.name, "first_output_file_sha256": file_hash(output_path),
        "missing_file": mask_path.name, "missing_file_sha256": file_hash(mask_path),
    }


def compare_saved_pair(blocks, dtype, directory):
    import numpy as np

    pair = {block["method"]: block for block in blocks if block["dtype"] == dtype}
    if set(pair) != set(METHODS) or any("first_output_file" not in block for block in pair.values()):
        return {"dtype": dtype, "status": "not_evaluated", "reason": "missing_method_output"}
    left, right = pair["faiss"], pair["knn"]
    if left["fingerprints"] != right["fingerprints"]:
        return {"dtype": dtype, "status": "failed", "reason": "method_input_mismatch"}
    actual = np.load(directory / left["first_output_file"], allow_pickle=False)
    reference = np.load(directory / right["first_output_file"], allow_pickle=False)
    missing = np.load(directory / left["missing_file"], allow_pickle=False)
    reference_missing = np.load(directory / right["missing_file"], allow_pickle=False)
    if not np.array_equal(missing, reference_missing):
        return {"dtype": dtype, "status": "failed", "reason": "method_mask_mismatch"}
    return {"dtype": dtype, "criterion_id": CRITERION_ID, **compare_outputs(actual, reference, missing)}


def worker(spec, spec_path, identity, output):
    result = {
        **identity, "status": "running", "evidence_validated": False,
        "spec_revision": spec["revision"], "spec_sha256": file_hash(spec_path),
        "criterion_id": CRITERION_ID, "rtol": RTOL, "atol": ATOL,
        "verification_plan": VERIFICATION_PLAN, "pid": os.getpid(),
        "blocks": [], "comparisons": [],
    }
    save_json(output, result)
    try:
        helper = load_helper(spec)
        import faiss
        from sklearn import config_context
        from threadpoolctl import threadpool_info, threadpool_limits

        result["environment"] = helper.metadata()
        result["source_sha256"] = {name: file_hash(ROOT / name) for name in HELPER_HASHES}
        result["source_sha256"]["benchmarks/verify_benchmark_correctness.py"] = file_hash(__file__)
        faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1), config_context(working_memory=256):
            result["threadpools"] = threadpool_info()
            result["faiss_omp_threads"] = int(faiss.omp_get_max_threads())
            if result["faiss_omp_threads"] != 1 or any(pool.get("num_threads") != 1 for pool in result["threadpools"]):
                raise ValueError("Native thread limits did not take effect")
            for method in METHODS:
                for dtype in DTYPES:
                    try:
                        block = run_block(helper, identity, method, dtype, output)
                    except Exception:
                        block = {"method": method, "dtype": dtype, "status": "error", "error": traceback.format_exc()}
                    result["blocks"].append(block)
                    result["observed_failure"] = any(b["status"] != "passed" for b in result["blocks"])
                    save_json(output, result)
                    gc.collect()
            result["comparisons"] = [compare_saved_pair(result["blocks"], dtype, output.parent) for dtype in DTYPES]
        result["status"] = "passed" if all(
            item["status"] == "passed" for item in result["blocks"] + result["comparisons"]
        ) else "failed"
        result["process_peak_rss_mib"] = helper.peak_rss_mib()
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    save_json(output, result)
    return result


def audit_worker(record, identity, spec_hash, directory):
    """Recompute acceptance from saved arrays before accepting a worker record."""
    import numpy as np

    if any(record.get(key) != value for key, value in identity.items()) or record.get("spec_sha256") != spec_hash:
        raise ValueError("Worker identity or specification differs")
    if (record.get("criterion_id"), record.get("rtol"), record.get("atol")) != (CRITERION_ID, RTOL, ATOL):
        raise ValueError("Worker criterion differs")
    blocks = record.get("blocks", [])
    if [(block["method"], block["dtype"]) for block in blocks] != [(method, dtype) for method in METHODS for dtype in DTYPES]:
        raise ValueError("Missing, duplicate or reordered method/dtype blocks")
    for block in blocks:
        if block["status"] not in ("passed", "failed"):
            raise ValueError("Incomplete block evidence")
        calls = block["calls"]
        if [call["call"] for call in calls] != list(range(1, 7)):
            raise ValueError("Missing or duplicate actual calls")
        for key in ("first_output_file", "missing_file"):
            name = block[key]
            if Path(name).name != name or file_hash(directory / name) != block[key + "_sha256"]:
                raise ValueError("Changed output artifact or invalid path")
        first = np.load(directory / block["first_output_file"], allow_pickle=False)
        if array_hash(first) != calls[0]["output_sha256"]:
            raise ValueError("Saved first output differs from its call evidence")
        first_value_hash = output_value_hash(first)
        if first_value_hash != calls[0]["output_value_sha256"]:
            raise ValueError("Saved first output differs from its numerical-value hash")
        for call in calls:
            required = {"shape_preserved", "dtype_preserved", "finite", "observed_values_preserved",
                        "query_unchanged", "training_unchanged", "separate_output"}
            if call["call"] > 1:
                required |= {"identical_to_call_1", "retained_first_output_unchanged"}
            if set(call["checks"]) != required or any(type(flag) is not bool for flag in call["checks"].values()):
                raise ValueError("Missing structural or repeat check")
            if call["checks"].get("identical_to_call_1") and call["output_value_sha256"] != first_value_hash:
                raise ValueError("Repeat-equality claim disagrees with numerical-value hashes")
        calculated = "passed" if all(all(call["checks"].values()) for call in calls) else "failed"
        if block["status"] != calculated:
            raise ValueError("Block status disagrees with its checks")
    comparisons = [compare_saved_pair(blocks, dtype, directory) for dtype in DTYPES]
    if comparisons != record.get("comparisons"):
        raise ValueError("Stored acceptance differs from saved-array arithmetic")
    calculated = "passed" if all(item["status"] == "passed" for item in blocks + comparisons) else "failed"
    if record["status"] != calculated:
        raise ValueError("Worker status disagrees with its evidence")


def orchestrate(spec, spec_path, output):
    expected = worker_schedule(spec)
    spec_hash = file_hash(spec_path)
    result = {
        "status": "running", "specification": spec, "spec_sha256": spec_hash,
        "verification_plan": VERIFICATION_PLAN, "expected_workers": expected,
        "criterion_id": CRITERION_ID, "rtol": RTOL, "atol": ATOL,
        "performance_gate_status": "not_evaluated", "records": [],
    }
    save_json(output, result)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({name: "1" for name in THREAD_ENV})
    started = time.monotonic()
    input_fingerprints = {}
    for identity in expected:
        remaining = VERIFICATION_PLAN["total_worker_budget_seconds"] - (time.monotonic() - started)
        stem = f"{identity['donor_policy']}-seed{identity['seed']}-repeat{identity['repeat']}"
        worker_output = output.with_name(stem + ".json")
        if remaining <= 0:
            record = {**identity, "status": "resource_bounded", "evidence_validated": False,
                      "completion_status": "not_executed", "reason": "total_preflight_worker_budget_exhausted"}
        else:
            command = [sys.executable, "-I", str(Path(__file__).resolve()),
                       "--spec", str(spec_path), "--output", str(worker_output), "--worker",
                       "--policy", identity["donor_policy"], "--seed", str(identity["seed"]),
                       "--repeat", str(identity["repeat"])]
            worker_started = time.monotonic()
            timed_out = False
            returncode = None
            with output.with_name(stem + ".log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=min(300, remaining), check=False)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            try:
                record = json.loads(worker_output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {**identity, "status": "process_error", "reason": "missing_or_invalid_worker_json"}
            record["evidence_validated"] = False
            record["worker_wall_seconds"] = time.monotonic() - worker_started
            record["returncode"] = returncode
            if timed_out:
                record.update(status_before_timeout=record["status"], status="resource_bounded",
                              completion_status="not_completed", reason="preflight_worker_wall_budget_exceeded")
            elif record.get("status") in ("passed", "failed") and returncode == int(record["status"] != "passed"):
                try:
                    audit_worker(record, identity, spec_hash, output.parent)
                    for block in record["blocks"]:
                        key = (identity["donor_policy"], identity["seed"], block["dtype"])
                        if input_fingerprints.setdefault(key, block["fingerprints"]) != block["fingerprints"]:
                            raise ValueError("Prepared inputs differ across methods or process repeats")
                    record["evidence_validated"] = True
                except Exception:
                    record.update(status="invalid_evidence", audit_error=traceback.format_exc())
            elif record.get("status") not in ("error", "process_error"):
                record.update(status="process_error", reason="worker_status_exit_mismatch")
        result["records"].append(record)
        result["summary"] = summarize_records(expected, result["records"])
        save_json(output, result)
        print(f"{stem}: {record['status']}", flush=True)
    result["status"] = result["summary"]["status"]
    result["preflight_wall_seconds"] = time.monotonic() - started
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=POLICIES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec_path, output = args.spec.resolve(), args.output.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(spec)
    if output.exists():
        parser.error("Output already exists; preserve the previous run instead of overwriting it")
    if args.worker:
        identity = {"donor_policy": args.policy, "seed": args.seed, "repeat": args.repeat}
        if identity not in worker_schedule(spec):
            parser.error("Worker identity outside the frozen schedule")
        result = worker(spec, spec_path, identity, output)
    else:
        result = orchestrate(spec, spec_path, output)
    print(f"Correctness status: {result['status']}; performance gate: not_evaluated", flush=True)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
