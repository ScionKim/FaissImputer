"""Benchmark-only batch-budget experiment; product files are unchanged."""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from benchmarks import benchmark_scaling_threads as base

BUDGETS = {
    "KNNImputer": None,
    "FaissImputer[available]": 16,
    "FaissImputer[available,64MiB]": 64,
    "FaissImputer[available,128MiB]": 128,
}
RTOL = 1e-6
ATOL = 1e-7


def key(config, method=None):
    return (
        config["size"], config["seed"], config["pattern"],
        config["threads"], config["repeat"],
        config["method"] if method is None else method,
    )


def worker(config):
    budget = BUDGETS[config["method"]]
    original = base.FaissImputer._transform_available_batched
    source = textwrap.dedent(inspect.getsource(original))
    tokens = [
        f"({value} * 1024 * 1024)"
        for value in (16, 128)
        if f"({value} * 1024 * 1024)" in source
    ]
    if len(tokens) != 1 or source.count(tokens[0]) != 1:
        raise RuntimeError("Batch formula changed; review this experiment first")
    token = tokens[0]

    effective = source
    cls = base.FaissImputer
    if budget is not None:
        effective = source.replace(token, f"({budget} * 1024 * 1024)", 1)
        namespace = original.__globals__.copy()
        exec(compile(effective, "<batch-budget-experiment>", "exec"), namespace)
        cls = type(
            f"BatchBudget{budget}Imputer",
            (base.FaissImputer,),
            {"_transform_available_batched": namespace[original.__name__]},
        )

    old_factory = base.make_model
    last_model = []

    def factory(method):
        if method == "KNNImputer":
            model = old_factory(method)
        else:
            model = cls(
                n_neighbors=base.NEIGHBORS,
                metric="l2", strategy="mean", index_factory="Flat",
                donor_policy="available",
            )
        last_model[:] = [model]
        return model

    base.make_model = factory
    try:
        record = base.worker(config)
    finally:
        base.make_model = old_factory

    model = last_model[0]
    donor_count = None if budget is None else len(model.donors_)
    batch_rows = (
        None if budget is None else
        max(1, min(256, (budget * 1024 * 1024) // (12 * donor_count)))
    )
    record.update({
        "experiment_batch_budget_mib": budget,
        "batch_row_limit": batch_rows,
        "available_donors": donor_count,
        "model_class": type(model).__name__,
        "original_method_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "effective_method_sha256": (
            None if budget is None else
            hashlib.sha256(effective.encode()).hexdigest()
        ),
    })
    return record


def run_worker(config, timeout):
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        env[name] = str(config["threads"])
    command = [
        sys.executable, "-u", "-m", "benchmarks.benchmark_available_batches",
        "--worker", json.dumps(config),
    ]
    started = base.time.perf_counter()
    try:
        process = subprocess.run(
            command, cwd=base.ROOT, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout,
                "worker_peak_rss_mib": None}
    if process.returncode:
        return {"status": "process_error", "returncode": process.returncode,
                "error": process.stderr[-2000:], "worker_peak_rss_mib": None}
    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
        if not isinstance(record, dict) or "status" not in record:
            raise ValueError("Missing status")
    except (ValueError, IndexError):
        return {"status": "invalid_worker_output", "worker_peak_rss_mib": None}
    record["worker_wall_seconds"] = base.time.perf_counter() - started
    if process.stderr.strip():
        record["warnings"] = process.stderr[-2000:]
    return record


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker")
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    if args.worker:
        try:
            result = worker(json.loads(args.worker))
        except MemoryError:
            result = {"status": "memory_error", "worker_peak_rss_mib": base.peak_rss_mib()}
        except Exception as error:
            result = {
                "status": "error", "error": f"{type(error).__name__}: {error}",
                "worker_peak_rss_mib": base.peak_rss_mib(),
            }
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0

    if args.output is None:
        args.output = base.ROOT / "benchmark_outputs/available_batches.json"
        sys.argv.extend(["--output", str(args.output)])

    outputs = {}

    def measured(config, timeout):
        record = run_worker(config, timeout)
        if record.get("status") == "ok":
            outputs[key(config)] = np.asarray(record["_values"], dtype=np.float64)
        return record

    base.METHODS = tuple(BUDGETS)
    base.run_worker = measured
    exit_code = base.main()
    results = json.loads(args.output.read_text(encoding="utf-8"))
    results["parameters"].update({
        "experiment_batch_budgets_mib": [16, 64, 128],
        "comparison_rtol": RTOL,
        "comparison_atol": ATOL,
    })
    results["notes"] = [
        note for note in results["notes"]
        if not note.startswith(("Complete policy", "One seed and one repeat"))
    ]
    results["notes"].extend([
        "Experimental subclasses change only the batch-budget constant in memory.",
        "The released implementation and product files are unchanged.",
        "16/64/128 MiB are batch-sizing budgets, NOT total process memory caps.",
        "FaissImputer[available] denotes the historical 16 MiB baseline in this experiment.",
        "Each candidate is compared with that 16 MiB baseline in the same repeat.",
        "original_method_sha256 identifies the current product method, not the 16 MiB baseline.",
        "Numerical tolerance is a diagnostic, not proof of general equivalence.",
        "A candidate outside tolerance makes this experiment fail visibly.",
        "Repeated timings on synthetic data do not establish general performance.",
    ])
    valid_keys = {
        key(record) for record in results["records"] if record["status"] == "ok"
    }
    for record in results["records"]:
        if record["status"] != "ok":
            continue
        reference_key = key(record, "FaissImputer[available]")
        reference = outputs.get(reference_key) if reference_key in valid_keys else None
        record["max_abs_difference_from_original"] = None
        record["exactly_changed_cells"] = None
        record["cells_outside_tolerance"] = None
        if reference is None:
            continue
        values = outputs[key(record)]
        record["max_abs_difference_from_original"] = float(
            np.max(np.abs(values - reference))
        )
        record["exactly_changed_cells"] = int(np.count_nonzero(values != reference))
        outside = int(np.count_nonzero(
            ~np.isclose(values, reference, rtol=RTOL, atol=ATOL)
        ))
        record["cells_outside_tolerance"] = outside
        if BUDGETS[record["method"]] in (64, 128) and outside:
            record["status"] = "output_difference"
            exit_code = 1

    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Batch comparison saved: {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
