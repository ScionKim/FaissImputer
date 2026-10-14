"""Bounded ordered-call diagnosis of an installed release; run on GitHub only."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from copy import deepcopy
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DTYPES = ("float32", "float64")
STAGES = ("factory", "train", "add", "search")
THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spec(spec):
    """Reject a different experiment rather than silently changing run 1."""
    if spec.get("revision") != "validation_0.3.22_v1":
        raise ValueError("Unrecognized frozen specification revision")
    if spec["package"].get("version") != "0.3.22":
        raise ValueError("This diagnosis evaluates the published 0.3.22 release")
    expected = {
        "train_rows": 20000, "query_rows": 3000, "features": 20,
        "n_neighbors": 5, "weights": "uniform", "donor_policy": "complete",
        "pattern": "random", "threads": 1, "working_memory_mib": 256,
        "seeds": [101, 202, 303], "repeats_per_seed": 3,
        "ordered_dtypes": list(DTYPES), "calls_per_dtype": 6,
        "worker_timeout_seconds": 180, "run_number": 1, "outer_parallelism": 1,
    }
    for name, value in expected.items():
        if spec["diagnostic"].get(name) != value:
            raise ValueError(f"Frozen diagnostic field changed: {name}")
    warmup = spec["diagnostic"]["warmup"]
    for name, value in {
        "train_rows": 32, "query_rows": 8, "features": 20, "seed": 7,
        "missing_query_columns": [0, 1, 2, 3], "per_dtype": True,
        "separate_model": True, "enabled": True,
    }.items():
        if warmup.get(name) != value:
            raise ValueError(f"Frozen warmup field changed: {name}")
    isolation = spec["diagnostic"]["process_isolation"]
    for name, value in {
        "worker_count": 9, "fresh_subprocess": True, "sequential_workers": True,
        "dtype_blocks_same_worker": True, "same_seed_and_repeat_for_both_dtypes": True,
        "alternate_order": False, "automatic_retry": False,
    }.items():
        if isolation.get(name) != value:
            raise ValueError(f"Frozen process field changed: {name}")
    missingness = spec["diagnostic"]["missingness"]
    for name, value in {
        "train_fraction": 0.0, "query_missing_features_per_row": 4,
        "query_fraction": 0.2, "query_selection": "random_without_replacement_per_row",
    }.items():
        if missingness.get(name) != value:
            raise ValueError(f"Frozen diagnostic missingness changed: {name}")
    if spec["diagnostic"]["reference"].get("tolerance") is not None:
        raise ValueError("An unsourced tolerance cannot establish KNN equivalence")


def worker_schedule(spec):
    diagnostic = spec["diagnostic"]
    return [
        {"seed": seed, "repeat": repeat}
        for seed in diagnostic["seeds"]
        for repeat in range(1, diagnostic["repeats_per_seed"] + 1)
    ]


def run_protocol(diagnostic, run_timed, run_reference, run_trace, checkpoint):
    """Keep both untraced dtype blocks ahead of reference and trace work."""
    result = {"blocks": [], "references": [], "traces": []}
    checkpoint(deepcopy(result))
    for key, callback in (
        ("blocks", run_timed), ("references", run_reference), ("traces", run_trace)
    ):
        for dtype in diagnostic["ordered_dtypes"]:
            result[key].append(callback(dtype))
            checkpoint(deepcopy(result))
    return result


class FaissStageTimer(AbstractContextManager):
    """Time factory-created indexes in a separate replay, preserving their types.

    Patch methods on the concrete Python index class, not each index instance:
    a closure holding an instance's bound method would retain index memory.
    These hooks never run during the primary ordered-call measurements.
    """

    def __init__(self, module, clock=time.perf_counter):
        self.module = module
        self.clock = clock
        self.original_factory = None
        self.patches = []
        self.patched_types = set()
        self.index_types = set()
        self.reset()

    def reset(self):
        self.totals = {name: {"calls": 0, "seconds": 0.0} for name in STAGES}

    def measure(self, name, operation, *args, **kwargs):
        started = self.clock()
        try:
            return operation(*args, **kwargs)
        finally:
            self.totals[name]["calls"] += 1
            self.totals[name]["seconds"] += self.clock() - started

    def attach(self, index):
        cls = type(index)
        self.index_types.add(f"{cls.__module__}.{cls.__name__}")
        if cls in self.patched_types:
            return index
        for name in ("train", "add", "search"):
            original = getattr(cls, name)
            if getattr(original, "_ordered_timer", None) is self:
                continue
            existed = name in cls.__dict__
            previous = cls.__dict__.get(name)

            def measured(instance, *args, _name=name, _original=original, **kwargs):
                return self.measure(_name, _original, instance, *args, **kwargs)

            measured._ordered_timer = self
            setattr(cls, name, measured)
            self.patches.append((cls, name, existed, previous))
        self.patched_types.add(cls)
        return index

    def __enter__(self):
        self.original_factory = self.module.index_factory

        def factory(*args, **kwargs):
            index = self.measure("factory", self.original_factory, *args, **kwargs)
            return self.attach(index)

        self.module.index_factory = factory
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.module.index_factory = self.original_factory
        for cls, name, existed, previous in reversed(self.patches):
            if existed:
                setattr(cls, name, previous)
            else:
                delattr(cls, name)
        self.patches.clear()
        self.patched_types.clear()
        return False


def array_context(array):
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "strides_bytes": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "writeable": bool(array.flags.writeable),
    }


def array_hash(array):
    import numpy as np

    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def difference(actual, expected):
    """Descriptive evidence only: no invented sklearn acceptance tolerance."""
    import numpy as np

    if actual.shape != expected.shape:
        return {"comparable": False, "reason": "shape_mismatch"}
    equal = bool(np.array_equal(actual, expected, equal_nan=True))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    result = {
        "comparable": finite,
        "array_equal": equal,
        "different_elements": int(np.count_nonzero(actual != expected)),
        "max_abs_difference": None,
    }
    if finite and equal:
        result["max_abs_difference"] = 0.0
    elif finite:
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        result["max_abs_difference"] = float(np.max(np.abs(delta), initial=0.0))
    return result


def output_checks(output, query, before, missing, train, train_hash, first=None):
    import numpy as np

    same_shape = output.shape == before.shape
    checks = {
        "shape_preserved": same_shape,
        "dtype_preserved": output.dtype == before.dtype,
        "finite": bool(np.isfinite(output).all()),
        "observed_values_preserved": bool(
            same_shape and np.array_equal(output[~missing], before[~missing])
        ),
        "query_unchanged": bool(np.array_equal(query, before, equal_nan=True)),
        "training_unchanged": array_hash(train) == train_hash,
        "separate_output": not np.shares_memory(output, query),
    }
    if first is not None:
        checks["identical_to_call_1"] = bool(
            np.array_equal(output, first, equal_nan=True)
        )
    return {name: bool(value) for name, value in checks.items()}


def load_helper():
    # With -I, import the installed release BEFORE exposing repository helpers.
    import faiss_imputer

    sys.path.insert(0, str(ROOT))
    try:
        helper = importlib.import_module("benchmarks.benchmark_scaling_threads")
    finally:
        sys.path.pop(0)
    expected = ROOT / "benchmarks/benchmark_scaling_threads.py"
    if Path(helper.__file__).resolve() != expected.resolve():
        raise RuntimeError("Unexpected synthetic helper source")
    helper.check_released_package("0.3.22")
    if (helper.FEATURES, helper.NEIGHBORS, helper.WORKING_MEMORY_MIB) != (20, 5, 256):
        raise ValueError("Supplied synthetic helper configuration has changed")
    return helper


def make_case(helper, diagnostic, seed, dtype):
    import numpy as np

    train, query, truth, missing = helper.make_data(
        diagnostic["train_rows"], diagnostic["query_rows"], seed,
        diagnostic["pattern"], training_policy="complete", dtype=dtype,
    )
    if train.shape != (20000, 20) or query.shape != (3000, 20):
        raise ValueError("Synthetic helper returned unexpected shapes")
    if train.dtype != np.dtype(dtype) or query.dtype != np.dtype(dtype):
        raise ValueError("Synthetic helper returned an unexpected dtype")
    if not np.isfinite(train).all() or not np.isfinite(truth).all():
        raise ValueError("Complete-policy training/truth must be fully observed")
    if not np.array_equal(np.isnan(query), missing):
        raise ValueError("Query mask and missing values disagree")
    if not np.all(missing.sum(axis=1) == 4):
        raise ValueError("Original workload requires four missing features per query")
    return train, query, truth, missing


def new_model(method):
    if method == "knn":
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
    from faiss_imputer import FaissImputer

    return FaissImputer(
        n_neighbors=5, weights="uniform", metric="l2", strategy="mean",
        index_factory="Flat", donor_policy="complete",
    )


def warm_up(dtype):
    import numpy as np

    train = np.random.default_rng(7).normal(size=(32, 20)).astype(dtype)
    query = train[:8].copy()
    query[:, :4] = np.nan
    started = time.perf_counter()
    new_model("faiss").fit(train).transform(query)
    return train, query, time.perf_counter() - started


def measure_block(helper, diagnostic, seed, dtype, output_path, traced=False):
    import numpy as np

    train, query, truth, missing = make_case(helper, diagnostic, seed, dtype)
    before = query.copy()
    train_hash = array_hash(train)
    marks = {"train": train_hash, "query": array_hash(query), "truth": array_hash(truth)}
    # Retain warmup arrays through the block, as in the supplied worker.
    warm_train, warm_query, warm_seconds = warm_up(dtype)
    model = new_model("faiss")
    timer = None
    if traced:
        import faiss

        timer = FaissStageTimer(faiss)
        timer.__enter__()
    try:
        gc.collect()
        rss_before = helper.current_rss_mib()
        started = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - started
        fit_stages = deepcopy(timer.totals) if timer else None
        gc.collect()
        rss_after = helper.current_rss_mib()
        calls = []

        def record_call(number, elapsed, output, first):
            checks = output_checks(output, query, before, missing, train, train_hash, first)
            calls.append({
                "call": number, "elapsed_seconds": elapsed, "dtype": dtype,
                "input": array_context(query), "output": array_context(output),
                "output_sha256": array_hash(output), "checks": checks,
                "checks_passed": all(checks.values()),
                "difference_from_call_1": None if first is None else difference(output, first),
                "faiss_stages": deepcopy(timer.totals) if timer else None,
            })

        if timer:
            timer.reset()
        started = time.perf_counter()
        output = model.transform(query)
        elapsed = time.perf_counter() - started
        record_call(1, elapsed, output, None)
        first_output = output.copy()
        for number in range(2, diagnostic["calls_per_dtype"] + 1):
            if timer:
                timer.reset()
            started = time.perf_counter()
            repeated_output = model.transform(query)
            elapsed = time.perf_counter() - started
            record_call(number, elapsed, repeated_output, first_output)
            retained_unchanged = bool(np.array_equal(output, first_output, equal_nan=True))
            calls[-1]["checks"]["retained_first_output_unchanged"] = retained_unchanged
            calls[-1]["checks_passed"] = calls[-1]["checks_passed"] and retained_unchanged

        if traced:
            untraced = np.load(output_path, allow_pickle=False)
            replay_comparison = difference(first_output, untraced)
        else:
            # Save only after all six calls: no output-file I/O between calls.
            np.save(output_path, first_output, allow_pickle=False)
            replay_comparison = None
        checks_passed = all(call["checks_passed"] for call in calls)
        if traced:
            checks_passed = checks_passed and replay_comparison["array_equal"]
        return {
            "status": "ok" if checks_passed else "validation_failed",
            "dtype": dtype, "donor_policy": "complete", "seed": seed,
            "instrumented_replay": traced,
            "train": array_context(train), "query": array_context(query),
            "fingerprints": marks, "warmup_seconds_excluded": warm_seconds,
            "fit_seconds": fit_seconds,
            "fit_plus_first_transform_seconds": fit_seconds + calls[0]["elapsed_seconds"],
            "fit_faiss_stages": fit_stages, "calls": calls,
            "rss_before_fit_mib": rss_before, "rss_after_fit_mib": rss_after,
            "process_peak_rss_mib": helper.peak_rss_mib(),
            "first_output_file": Path(output_path).name,
            "traced_vs_untraced_output": replay_comparison,
            "stage_coverage": None if not timer else {
                "scope": "Python methods on factory-created indexes; separate replay only",
                "index_types": sorted(timer.index_types),
            },
        }
    finally:
        if timer:
            timer.__exit__(None, None, None)


def reference_block(helper, diagnostic, seed, dtype, output_path, marks):
    import numpy as np

    train, query, truth, missing = make_case(helper, diagnostic, seed, dtype)
    regenerated = {"train": array_hash(train), "query": array_hash(query), "truth": array_hash(truth)}
    if regenerated != marks:
        raise ValueError("Reference inputs differ from the timed Faiss inputs")
    before = query.copy()
    output = new_model("knn").fit(train).transform(query)
    faiss_output = np.load(output_path, allow_pickle=False)
    checks = output_checks(output, query, before, missing, train, marks["train"])
    return {
        "status": "ok" if all(checks.values()) else "validation_failed",
        "dtype": dtype, "fingerprints": regenerated,
        "sklearn_output_sha256": array_hash(output), "checks": checks,
        "faiss_vs_knn_all_output": difference(faiss_output, output),
        "faiss_vs_knn_imputed_only": difference(faiss_output[missing], output[missing]),
        "correctness_gate": {
            "status": "not_evaluated",
            "reason": "Existing sklearn-equivalence acceptance tolerance is not yet specified.",
        },
    }


def worker(spec, spec_path, identity, output):
    from importlib.metadata import version

    import faiss
    from sklearn import config_context
    from threadpoolctl import threadpool_info, threadpool_limits

    helper = load_helper()
    for distribution, expected_version in spec["package"]["dependency_pins"].items():
        if version(distribution) != expected_version:
            raise ValueError(f"Unexpected dependency version: {distribution}")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != spec["package"]["python_minor"]:
        raise ValueError("Python minor version differs from the frozen environment")
    diagnostic = spec["diagnostic"]
    faiss.omp_set_num_threads(1)
    result = {
        **identity, "status": "running", "spec_sha256": file_hash(spec_path),
        "spec_revision": spec["revision"], "pid": os.getpid(),
        "execution_order": [f"complete {dtype} calls 1-6" for dtype in DTYPES],
        "post_measurement_order": ["KNN float32", "KNN float64", "trace float32", "trace float64"],
        "metadata": helper.metadata(),
        "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (Path(__file__).resolve(), ROOT / "benchmarks/benchmark_scaling_threads.py",
                         ROOT / "benchmarks/benchmark_imputers.py")
        },
        "protocol": diagnostic, "blocks": [], "references": [], "traces": [],
    }
    save_json(output, result)

    def checkpoint(partial):
        result.update(partial)
        save_json(output, result)

    def output_path(dtype):
        return output.with_name(f"{output.stem}-{dtype}-call1.npy")

    def timed(dtype):
        block = measure_block(helper, diagnostic, identity["seed"], dtype, output_path(dtype))
        # Locals, model, and numpy outputs from float32 no longer remain live.
        # Imported modules/native pools/allocator state persist into float64.
        gc.collect()
        return block

    def reference(dtype):
        marks = next(block["fingerprints"] for block in result["blocks"] if block["dtype"] == dtype)
        return reference_block(helper, diagnostic, identity["seed"], dtype, output_path(dtype), marks)

    def trace(dtype):
        return measure_block(helper, diagnostic, identity["seed"], dtype, output_path(dtype), traced=True)

    try:
        with threadpool_limits(limits=1), config_context(working_memory=256):
            result["threadpools"] = threadpool_info()
            result["faiss_omp_threads"] = int(faiss.omp_get_max_threads())
            if result["faiss_omp_threads"] != 1 or any(
                pool.get("num_threads") != 1 for pool in result["threadpools"]
            ):
                raise ValueError("Native thread limits did not take effect")
            run_protocol(diagnostic, timed, reference, trace, checkpoint)
        result["status"] = (
            "ok" if all(item["status"] == "ok" for key in ("blocks", "references", "traces")
                        for item in result[key]) else "validation_failed"
        )
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    finally:
        result["process_peak_rss_mib"] = helper.peak_rss_mib()
        save_json(output, result)
    return result


def summarize(records):
    summaries = []
    for dtype in DTYPES:
        blocks = [block for record in records for block in record.get("blocks", [])
                  if block["dtype"] == dtype and block["status"] == "ok"]
        summaries.append({
            "dtype": dtype, "completed_valid_blocks": len(blocks),
            "call_seconds_medians": [
                median(block["calls"][index]["elapsed_seconds"] for block in blocks)
                if blocks else None for index in range(6)
            ],
            "call_3_over_call_2_ratios": [
                {"seed": record["seed"], "repeat": record["repeat"],
                 "ratio": block["calls"][2]["elapsed_seconds"] / block["calls"][1]["elapsed_seconds"]}
                for record in records for block in record.get("blocks", [])
                if block["dtype"] == dtype and block["status"] == "ok"
                and block["calls"][1]["elapsed_seconds"] > 0
            ],
            "interpretation": "Descriptive timing evidence; no automatic anomaly or correctness verdict.",
        })
    return summaries


def orchestrate(spec, spec_path, output):
    expected = worker_schedule(spec)
    result = {
        "status": "running", "specification": spec, "spec_sha256": file_hash(spec_path),
        "expected_workers": expected, "records": [],
        "correctness_gate_status": "not_evaluated",
        "notes": [
            "One diagnostic run contains nine fresh sequential workers; no automatic retries.",
            "Each worker measures float32 calls 1-6, then float64 calls 1-6, before KNN or tracing.",
            "Trace elapsed times are separate instrumented replays, not the original call timings.",
            "Peak RSS is process-lifetime RSS, including reference/tracing; not phase-specific memory.",
            "A resource timeout is retained as incomplete evidence, not an algorithm speed verdict.",
            "The 500k scaling resource policy is specified but is not exercised by this diagnostic.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({name: "1" for name in THREAD_ENV})
    save_json(output, result)
    for identity in expected:
        stem = f"seed{identity['seed']}-repeat{identity['repeat']}"
        worker_output = output.with_name(stem + ".json")
        command = [
            sys.executable, "-I", str(Path(__file__).resolve()),
            "--spec", str(spec_path), "--output", str(worker_output),
            "--worker", "--seed", str(identity["seed"]), "--repeat", str(identity["repeat"]),
        ]
        started = time.perf_counter()
        returncode = None
        failure = None
        with output.with_name(stem + ".log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=spec["diagnostic"]["worker_timeout_seconds"], check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                failure = "diagnostic_timeout"
        record = json.loads(worker_output.read_text(encoding="utf-8")) if worker_output.exists() else {
            **identity, "status": "missing_worker_output",
        }
        if failure:
            record["worker_status_before_timeout"] = record["status"]
            record["status"] = failure
        elif returncode:
            record["worker_status_before_process_exit"] = record["status"]
            record["status"] = "process_error"
        record["worker_wall_seconds"] = time.perf_counter() - started
        record["returncode"] = returncode
        result["records"].append(record)
        result["summary"] = summarize(result["records"])
        save_json(output, result)
        print(f"{stem}: {record['status']}", flush=True)
    result["status"] = "ok" if all(record["status"] == "ok" for record in result["records"]) else "incomplete_or_failed"
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    output = args.output.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_spec(spec)
    if args.worker:
        identity = {"seed": args.seed, "repeat": args.repeat}
        if identity not in worker_schedule(spec):
            parser.error("Worker identity is outside the frozen schedule")
        result = worker(spec, spec_path, identity, output)
    else:
        result = orchestrate(spec, spec_path, output)
    print(f"Status: {result['status']}; correctness gate: not_evaluated")
    return int(result["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
