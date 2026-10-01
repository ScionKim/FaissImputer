"""Trace available-donor transforms without changing the search algorithm.

These instrumented timings are diagnostic/reference-only. They are not
benchmark speedups. Each case runs in a fresh sequential process; its first,
untraced transform warms that process before a separately fitted model is
traced. No KNNImputer measurements are made here.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from unittest.mock import patch


DATASETS = ("wine_quality_white", "abalone")
DTYPES = ("float32", "float64")
SEEDS = (101, 202, 303)
WORKER_TIMEOUT_SECONDS = 300
SEARCH_CONTRACT = "available-search-query-cache-and-refinement-v1"
BATCHED_SEARCH_CONTRACT = "available-search-query-cache-and-refinement-v2"


class TimerCollector:
    """Inclusive/self accounting for nested calls using a monotonic clock.

Self time is elapsed time minus the immediate traced children's elapsed
times. Instrumentation work between child calls belongs to the parent's
self time. Inclusive entries overlap and must not be added together.
"""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.stack = []
        self.timings = {}
        self.counters = defaultdict(int)
        self.search_events = []
        self.retain_events = []

    @contextmanager
    def measure(self, name):
        frame = {"name": name, "start": self.clock(), "children": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame["start"]
            self.stack.pop()
            own = elapsed - frame["children"]
            if own < -1e-8 or elapsed < 0:
                raise RuntimeError("Invalid nested timer accounting")
            own = max(0.0, own)
            entry = self.timings.setdefault(
                name,
                {"calls": 0, "inclusive_seconds": 0.0, "self_seconds": 0.0},
            )
            entry["calls"] += 1
            entry["inclusive_seconds"] += elapsed
            entry["self_seconds"] += own
            if self.stack:
                self.stack[-1]["children"] += elapsed

    def snapshot(self):
        if self.stack:
            raise RuntimeError("Cannot summarize an active timer")
        root = self.timings.get("instrumented_transform", {})
        root_seconds = root.get("inclusive_seconds", 0.0)
        self_sum = sum(entry["self_seconds"] for entry in self.timings.values())
        return {
            "timings": dict(sorted(self.timings.items())),
            "counters": dict(sorted(self.counters.items())),
            "search_events": self.search_events,
            "retain_events": self.retain_events,
            "accounting": {
                "root_inclusive_seconds": root_seconds,
                "sum_of_all_traced_self_seconds": self_sum,
                "root_minus_self_sum_seconds": root_seconds - self_sum,
                "definition": (
                    "Self times partition only the instrumented transform "
                    "root, including tracing overhead. Inclusive times overlap. "
                    "The root self time includes validation, output formatting, "
                    "cache cleanup, and tracing work outside traced children. "
                    "The available_transform self time is residual work outside "
                    "search, aggregation, and retain calls; it is not pure "
                    "aggregation time. Do not equate these values with the "
                    "untraced reference timing."
                ),
            },
        }


def _attribute_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    )


def guard_search_contract(index):
    """Guard the source-specific full-refinement reason classification.

The current implementation invokes _direct_distances before assigning
query_ref for suspicion refinement, and afterwards for qualifying ties.
We classify by that state while the actual search is running, not by the
number of donors supplied to a distance kernel. Fail closed if this source
structure changes; the script's classification must then be reviewed.
"""
    source = textwrap.dedent(inspect.getsource(type(index).search))
    function = ast.parse(source).body[0]
    expected_test = ast.parse("queries is not self.query_ref", mode="eval").body
    first = function.body[0]
    if not isinstance(first, ast.If) or ast.dump(first.test) != ast.dump(expected_test):
        raise RuntimeError("Unsupported search cache guard; review the profiler")
    last = first.body[-1]
    expected_assignment = ast.parse("self.query_ref = queries").body[0]
    if ast.dump(last) != ast.dump(expected_assignment):
        raise RuntimeError("Unsupported query_ref assignment; review the profiler")

    initial = [n for n in ast.walk(first) if _attribute_call(n, "_direct_distances")]
    remaining = [n for stmt in function.body[1:] for n in ast.walk(stmt)
                 if _attribute_call(n, "_direct_distances")]
    if len(initial) != 1 or len(remaining) != 1:
        raise RuntimeError("Unsupported full-refinement calls; review the profiler")
    initial_loop = next(
        (node for node in ast.walk(first) if isinstance(node, ast.For)
         and initial[0] in list(ast.walk(node))
         and "suspect" in ast.unparse(node.iter)),
        None,
    )
    tie_loop = next(
        (node for stmt in function.body[1:] for node in ast.walk(stmt)
         if isinstance(node, ast.For)
         and remaining[0] in list(ast.walk(node))
         and "tied.any" in ast.unparse(node.iter)),
        None,
    )
    if initial_loop is None or tie_loop is None:
        raise RuntimeError("Unsupported refinement branches; review the profiler")
    assignments = [node for node in ast.walk(function)
                   if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                   and "self.query_ref" in ast.unparse(node).split("=")[0]]
    if len(assignments) != 1:
        raise RuntimeError("Unsupported query_ref mutation; review the profiler")
    descriptor = inspect.getattr_static(type(index), "_precise_topk")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unsupported _precise_topk binding; review the profiler")
    selected = [node for node in ast.walk(function)
                if _attribute_call(node, "_distances_to")]
    batched = [node for node in ast.walk(function)
               if _attribute_call(node, "_refine_selected")]
    if len(selected) == 1 and not batched:
        contract = SEARCH_CONTRACT
        selected_definition = "Selected kernels process one query row per call. "
    elif len(batched) == 1 and not selected:
        refine = getattr(type(index), "_refine_selected", None)
        pairs = getattr(type(index), "_selected_distances", None)
        if refine is None or pairs is None:
            raise RuntimeError("Unsupported selected batching; review the profiler")
        refine_tree = ast.parse(textwrap.dedent(inspect.getsource(refine)))
        pair_tree = ast.parse(textwrap.dedent(inspect.getsource(pairs)))
        if (sum(_attribute_call(node, "_selected_distances") for node in ast.walk(refine_tree)) != 1
                or sum(_attribute_call(node, "_distances_to") for node in ast.walk(pair_tree)) != 1):
            raise RuntimeError("Unsupported selected batch kernel; review the profiler")
        contract = BATCHED_SEARCH_CONTRACT
        selected_definition = (
            "Selected kernels process bounded pair blocks. selected_kernel_calls "
            "counts invocations; selected_row_events counts unique query row IDs "
            "within each block, so a row can repeat across candidate-column chunks. "
            "selected_distance_batch includes gathering and the nested kernel. "
        )
    else:
        raise RuntimeError("Unsupported selected refinement; review the profiler")
    return {
        "name": contract,
        "classification": (
            "During search, _direct_distances with query_ref=None is a "
            "new-matrix suspicion refinement; with query_ref set it is a "
            "qualifying-tie refinement. _distances_to called inside "
            "_direct_distances is a full-donor kernel; calls outside it are "
            "selected-candidate kernels, even when their donor count equals "
            "the full donor count. Counts are row events across calls, not "
            "unique query rows. " + selected_definition
        ),
    }


class AvailableTrace:
    """Patch actual bound methods temporarily; retain only scalar events."""

    def __init__(self, model, faiss_module, collector):
        self.model = model
        self.index = model.available_index_
        self.faiss = faiss_module
        self.collector = collector
        self._search_stack = []
        self._full_reasons = []
        self._selected_row_counts = []
        self._patches = ExitStack()
        self._originals = []

    def _patch(self, owner, name, wrapper_factory):
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        self._patches.enter_context(patch.object(owner, name, wrapper_factory(original)))

    def _timed(self, name):
        def factory(original):
            def wrapped(*args, **kwargs):
                with self.collector.measure(name):
                    return original(*args, **kwargs)
            return wrapped
        return factory

    def __enter__(self):
        try:
            self._patch(self.model, "_transform_available_batched", self._timed("available_transform"))
            self._patch(self.model, "_aggregate", self._aggregation)
            self._patch(self.index, "search", self._search)
            self._patch(self.index, "_prepare_search_matrix", self._prepare_matrix)
            self._patch(self.index, "_prepared_distances", self._prepared_distances)
            self._patch(self.index, "_direct_distances", self._full_distances)
            self._patch(self.index, "_distances_to", self._distance_kernel)
            if hasattr(self.index, "_selected_distances"):
                self._patch(self.index, "_selected_distances", self._selected_pairs)
            # This is the bound static function on the instance. Its wrapper
            # receives (distances, k), without introducing a self argument.
            self._patch(self.index, "_precise_topk", self._precise_topk)
            self._patch(self.index, "retain_queries", self._retain)
            self._patch(self.faiss, "kmin", self._kmin)
            return self
        except BaseException:
            self._patches.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        self._patches.close()
        if self._search_stack or self._full_reasons or self._selected_row_counts:
            raise RuntimeError("Unbalanced instrumentation state")
        if any(getattr(owner, name) != original for owner, name, original in self._originals):
            raise RuntimeError("Instrumentation did not restore the original methods")
        return False

    def _search(self, original):
        def wrapped(queries, k):
            cache_hit = queries is self.index.query_ref
            event = {
                "call": len(self.collector.search_events) + 1,
                "query_rows": len(queries),
                "requested_k": int(k),
                "cache_hit_before": cache_hit,
                "precise_rows_before": len(self.index.precise_rows),
                "suspect_full_row_events": 0,
                "tie_full_row_events": 0,
                "selected_row_events": 0,
                "selected_kernel_calls": 0,
                "selected_pair_batch_calls": 0,
            }
            self.collector.search_events.append(event)
            count = self.collector.counters
            count["search_calls"] += 1
            count["search_query_row_events"] += len(queries)
            count["requested_candidate_pair_events"] += len(queries) * min(int(k), len(self.index.donors64))
            if cache_hit:
                count["cache_hit_search_calls"] += 1
                count["expansion_search_calls"] += 1
            else:
                count["initial_batch_search_calls"] += 1
                count["initial_batch_query_rows"] += len(queries)
            self._search_stack.append(event)
            try:
                with self.collector.measure("search"):
                    result = original(queries, k)
                event["status"] = "ok"
                return result
            except BaseException:
                event["status"] = "error"
                raise
            finally:
                event["cache_identity_after"] = self.index.query_ref is queries
                event["precise_rows_after"] = len(self.index.precise_rows)
                self._search_stack.pop()
        return wrapped

    def _prepare_matrix(self, original):
        def wrapped(queries):
            count = self.collector.counters
            count["matrix_build_calls"] += 1
            count["matrix_build_query_rows"] += len(queries)
            count["matrix_build_donor_pairs"] += len(queries) * len(self.index.donors64)
            with self.collector.measure("prepare_search_matrix"):
                result = original(queries)
            count["matrix_suspect_query_rows"] += int(result[1].sum())
            return result
        return wrapped

    def _prepared_distances(self, original):
        def wrapped(queries):
            self.collector.counters["prepared_distance_calls"] += 1
            with self.collector.measure("prepared_matrix_distances"):
                return original(queries)
        return wrapped

    def _full_distances(self, original):
        def wrapped(query):
            if not self._search_stack:
                raise RuntimeError("Full refinement occurred outside search")
            reason = "suspect" if self.index.query_ref is None else "tie"
            event = self._search_stack[-1]
            event[f"{reason}_full_row_events"] += 1
            self.collector.counters[f"full_{reason}_row_events"] += 1
            self.collector.counters[f"full_{reason}_donor_pair_events"] += len(self.index.donors64)
            self._full_reasons.append(reason)
            try:
                with self.collector.measure(f"full_direct_distances.{reason}"):
                    return original(query)
            finally:
                self._full_reasons.pop()
        return wrapped

    def _distance_kernel(self, original):
        def wrapped(query, donors, present):
            if self._full_reasons:
                kind = f"full_{self._full_reasons[-1]}"
            else:
                if not self._search_stack:
                    raise RuntimeError("Selected-candidate distances occurred outside search")
                kind = "selected"
                row_events = self._selected_row_counts[-1] if self._selected_row_counts else 1
                self._search_stack[-1]["selected_row_events"] += row_events
                self._search_stack[-1]["selected_kernel_calls"] += 1
                self.collector.counters["selected_row_events"] += row_events
                self.collector.counters["selected_kernel_calls"] += 1
                self.collector.counters["selected_donor_pair_events"] += len(donors)
            with self.collector.measure(f"direct_distance_kernel.{kind}"):
                return original(query, donors, present)
        return wrapped

    def _selected_pairs(self, original):
        def wrapped(queries, query_rows, donor_ids):
            if not self._search_stack:
                raise RuntimeError("Selected batching occurred outside search")
            row_events = len(set(int(row) for row in query_rows))
            self.collector.counters["selected_pair_batch_calls"] += 1
            self._search_stack[-1]["selected_pair_batch_calls"] += 1
            self._selected_row_counts.append(row_events)
            try:
                with self.collector.measure("selected_distance_batch"):
                    return original(queries, query_rows, donor_ids)
            finally:
                self._selected_row_counts.pop()
        return wrapped

    def _precise_topk(self, original):
        def wrapped(distances, k):
            self.collector.counters["precise_topk_calls"] += 1
            with self.collector.measure("precise_topk"):
                return original(distances, k)
        return wrapped

    def _retain(self, original):
        def wrapped(rows):
            before = len(self.index.query_ref)
            self.collector.counters["retain_calls"] += 1
            self.collector.counters["retained_query_row_events"] += len(rows)
            with self.collector.measure("retain_queries"):
                result = original(rows)
            self.collector.retain_events.append({
                "rows_before": before,
                "rows_retained": len(rows),
                "returned_cache_identity": result is self.index.query_ref,
                "precise_rows_after": len(self.index.precise_rows),
            })
            return result
        return wrapped

    def _aggregation(self, original):
        def wrapped(values, *, axis, ignore_nan):
            self.collector.counters["aggregation_calls"] += 1
            if axis == 1:
                self.collector.counters["aggregation_row_events"] += len(values)
            with self.collector.measure("aggregate"):
                return original(values, axis=axis, ignore_nan=ignore_nan)
        return wrapped

    def _kmin(self, original):
        def wrapped(matrix, k):
            self.collector.counters["faiss_kmin_calls"] += 1
            self.collector.counters["faiss_kmin_query_row_events"] += len(matrix)
            with self.collector.measure("faiss_kmin"):
                return original(matrix, k)
        return wrapped


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_case_helper():
    path = Path(__file__).resolve().with_name("benchmark_real_data_cases.py")
    spec = importlib.util.spec_from_file_location("_available_profile_cases", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the existing case helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _installed_package(expected_version):
    # Heavy imports are confined to child processes.
    import faiss
    import numpy as np
    import sklearn
    from faiss_imputer import FaissImputer

    actual = importlib.metadata.version("faiss-imputer")
    if actual != expected_version:
        raise RuntimeError(f"Expected faiss-imputer {expected_version}, found {actual}")
    package_path = Path(inspect.getfile(FaissImputer)).resolve()
    if not any(part in ("site-packages", "dist-packages") for part in package_path.parts):
        raise RuntimeError(f"FaissImputer was not imported from an installed wheel: {package_path}")
    return FaissImputer, faiss, np, {
        "faiss_imputer": actual,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "faiss": getattr(faiss, "__version__", importlib.metadata.version("faiss-cpu")),
        "python": platform.python_version(),
        "executable": sys.executable,
        "faiss_imputer_source_path": str(package_path),
        "faiss_imputer_source_sha256": _file_sha256(package_path),
    }


def _cpu_metadata():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "processor": cpu,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity,
        "native_thread_limit": 1,
    }


def _configuration(args, dtype=None, seed=None):
    return {
        "dataset": args.dataset,
        "mechanism": args.mechanism,
        "n_train": args.train_size,
        "n_query": args.query_size,
        "dtype": dtype,
        "seed": seed,
        "target_missing_rate": 0.10,
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "l2",
        "strategy": "mean",
        "index_factory": "Flat",
        "donor_policy": "available",
        "api": "fit_then_transform",
        "timing_scope": "first transform only; fit excluded",
    }


def _cache_is_empty(index):
    return index.query_ref is None and index.matrix is None and not index.precise_rows


def _worker(args):
    record = {"configuration": _configuration(args, args.dtype, args.seed),
              "status": "error", "checks_passed": False,
              "cpu": _cpu_metadata()}
    collector = TimerCollector()
    try:
        model_type, faiss_module, np, environment = _installed_package(args.expected_version)
        record["environment"] = environment
        helper, helper_path = _load_case_helper()
        record["case_helper"] = {"path": str(helper_path), "sha256": _file_sha256(helper_path)}
        from threadpoolctl import threadpool_info, threadpool_limits

        data, names, dataset_metadata = helper.load_dataset(
            args.data_home, download_if_missing=False, dataset_id=args.dataset,
        )
        record["dataset"] = dataset_metadata
        defaults = helper.DATASET_DEFAULTS[args.dataset]
        train, query, truth, missing, case = helper.prepare_case(
            data, names, args.seed, args.mechanism,
            train_size=args.train_size, query_size=args.query_size,
            dtype=args.dtype, missing_rate=0.10,
            mar_reference_rows=defaults["mar_reference_rows"],
            mar_driver=defaults["mar_driver"],
        )
        record["case"] = case
        before = {"train": helper.array_digest(train), "query": helper.array_digest(query)}
        parameters = dict(n_neighbors=5, weights="uniform", metric="l2", strategy="mean",
                          index_factory="Flat", donor_policy="available", copy=True)
        with threadpool_limits(limits=1):
            faiss_module.omp_set_num_threads(1)
            pools = threadpool_info()
            record["threadpools_during_transform"] = pools
            if any(pool["num_threads"] != 1 for pool in pools):
                raise RuntimeError("A native thread pool did not honor the one-thread limit")
            reference = model_type(**parameters).fit(train)
            start = time.perf_counter()
            reference_output = reference.transform(query)
            reference_seconds = time.perf_counter() - start
            record["reference_transform_seconds"] = reference_seconds
            if not _cache_is_empty(reference.available_index_):
                raise RuntimeError("Reference transform left a query cache behind")
            if before != {"train": helper.array_digest(train), "query": helper.array_digest(query)}:
                raise RuntimeError("The reference pass changed an input")

            instrumented = model_type(**parameters).fit(train)
            index = instrumented.available_index_
            if not _cache_is_empty(index):
                raise RuntimeError("The separately fitted model started with a query cache")
            contract = guard_search_contract(index)
            record["source_contract"] = contract
            matrix_path = Path(inspect.getfile(type(index))).resolve()
            environment["matrix_source_path"] = str(matrix_path)
            environment["matrix_source_sha256"] = _file_sha256(matrix_path)
            with AvailableTrace(instrumented, faiss_module, collector):
                with collector.measure("instrumented_transform"):
                    output = instrumented.transform(query)

        if output.dtype != reference_output.dtype or output.shape != reference_output.shape:
            raise RuntimeError("Instrumentation changed output dtype or shape")
        if not np.array_equal(output, reference_output, equal_nan=True):
            raise RuntimeError("Instrumented and reference outputs differ")
        if output.dtype != query.dtype or output.shape != query.shape:
            raise RuntimeError("Unexpected transform output dtype or shape")
        if not np.isfinite(output).all():
            raise RuntimeError("Transform output is not finite")
        if not np.array_equal(output[~missing], query[~missing], equal_nan=True):
            raise RuntimeError("Observed query values changed")
        after = {"train": helper.array_digest(train), "query": helper.array_digest(query)}
        if after != before:
            raise RuntimeError("A fit or transform changed an input")
        if not _cache_is_empty(index):
            raise RuntimeError("Instrumented transform left a query cache behind")
        profile = collector.snapshot()
        if abs(profile["accounting"]["root_minus_self_sum_seconds"]) > 1e-8:
            raise RuntimeError("Nested timer self accounting does not balance")
        error = output[missing].astype(np.float64) - truth[missing]
        record.update({
            "status": "ok", "checks_passed": True,
            "donor_rows": len(index.donors64),
            "instrumented_transform_seconds": profile["accounting"]["root_inclusive_seconds"],
            "profile": profile,
            "checks": {
                "exact_reference_output_agreement_equal_nan": True,
                "shape_and_dtype_preserved": True,
                "observed_values_preserved": True,
                "finite_output": True,
                "inputs_unchanged": True,
                "query_cache_cleared": True,
                "patches_restored": True,
                "input_hashes_before": before,
                "input_hashes_after": after,
                "reference_output_sha256": helper.array_digest(reference_output),
                "instrumented_output_sha256": helper.array_digest(output),
            },
            "quality": {
                "scope": "masked held-out query entries against benchmark ground truth",
                "scored_cells": int(missing.sum()),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "mae": float(np.mean(np.abs(error))),
            },
        })
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        record["partial_profile"] = collector.snapshot()
    _write_json(args.output, record)
    print(json.dumps({"status": record["status"], "dtype": args.dtype, "seed": args.seed}))
    return 0 if record["status"] == "ok" else 1


def _child_arguments(args, output):
    return [sys.executable, str(Path(__file__).resolve()),
            "--data-home", str(Path(args.data_home).resolve()),
            "--expected-version", args.expected_version,
            "--provenance", str(Path(args.provenance).resolve()),
            "--dataset", args.dataset, "--mechanism", args.mechanism,
            "--train-size", str(args.train_size), "--query-size", str(args.query_size),
            "--output", str(output)]


def _validate_provenance(provenance, expected_version):
    if not isinstance(provenance, dict) or provenance.get("version") != expected_version:
        raise ValueError("Candidate provenance version differs from expected installed version")
    commit = provenance.get("source_commit", "")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Candidate provenance must identify a full source commit")
    if not expected_version.endswith(f"+bench.{commit[:12]}"):
        raise ValueError("Candidate version suffix does not identify the provenance source commit")


def _orchestrate(args):
    provenance_path = Path(args.provenance).resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _validate_provenance(provenance, args.expected_version)
    records = []
    result = {
        "schema_version": 1,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_scope": "available-donor first held-out transform",
            "timing_status": "instrumented diagnostic/reference-only; no benchmark speedups",
            "process_order": "fresh sequential case workers; dataset must already be cached",
            "reference_order": "untraced reference transform first, separately fitted instrumented model second",
            "case_grid": {"dataset": args.dataset, "mechanism": args.mechanism,
                          "dtypes": args.dtypes, "seeds": args.seeds},
            "aggregation": "No pooled or method-level benchmark aggregation; one diagnostic record per case",
            "limitations": [
                "The untraced first pass warms the process; traced/reference timings are not directly comparable.",
                "Wrappers add overhead, especially for many small rowwise calls; inclusive timing buckets overlap.",
                "Residual available_transform self time includes donor filtering, gathering, assignments, control flow, and tracing overhead.",
                "Full and selected counts are repeated row/donor-pair events, not distinct query/donor pairs.",
                "No timing result from this diagnostic replaces an archived benchmark or establishes a speedup.",
                "Matching the reference checks instrumentation fidelity for this case, not agreement with KNNImputer.",
            ],
            "expected_installed_version": args.expected_version,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "provenance": provenance,
            "provenance_sha256": _file_sha256(provenance_path),
            "profile_script_sha256": _file_sha256(__file__),
        },
        "records": records,
        "status": "in_progress",
    }
    _write_json(args.output, result)
    with tempfile.TemporaryDirectory(prefix="available-profile-") as temporary:
        temporary = Path(temporary)
        for dtype in args.dtypes:
            for seed in args.seeds:
                worker_output = temporary / f"{dtype}-{seed}.json"
                command = _child_arguments(args, worker_output) + ["--worker", "--dtype", dtype, "--seed", str(seed)]
                try:
                    completed = subprocess.run(command, text=True, capture_output=True,
                                               check=False, timeout=WORKER_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    record = {
                        "configuration": _configuration(args, dtype, seed),
                        "status": "error", "checks_passed": False,
                        "process_error": {"type": "TimeoutExpired", "timeout_seconds": WORKER_TIMEOUT_SECONDS},
                    }
                    completed = None
                if completed is not None and worker_output.is_file():
                    record = json.loads(worker_output.read_text(encoding="utf-8"))
                    if completed.returncode != 0 and record.get("status") == "ok":
                        record.update(status="error", checks_passed=False)
                        record["process_error"] = {"returncode": completed.returncode, "stderr": completed.stderr}
                elif completed is not None:
                    record = {
                        "configuration": _configuration(args, dtype, seed),
                        "status": "error", "checks_passed": False,
                        "process_error": {"returncode": completed.returncode,
                                          "stdout": completed.stdout, "stderr": completed.stderr},
                    }
                records.append(record)
                result["counts"] = {"records": len(records), "ok": sum(r.get("status") == "ok" for r in records)}
                _write_json(args.output, result)
                print(f"{args.dataset} {dtype} seed={seed}: {record['status']}", flush=True)
    result["status"] = "ok" if all(r.get("status") == "ok" and r.get("checks_passed") for r in records) else "error"
    result["counts"] = {"records": len(records), "ok": sum(r.get("status") == "ok" for r in records)}
    _write_json(args.output, result)
    return 0 if result["status"] == "ok" else 1


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="wine_quality_white")
    parser.add_argument("--mechanism", choices=("MCAR", "MAR"), default="MCAR")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--query-size", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dtypes", choices=DTYPES, nargs="+", default=list(DTYPES))
    parser.add_argument("--output", type=Path, default=Path("benchmark_outputs/available_transform_profile.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=DTYPES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.train_size < 1000 or args.query_size < 1:
        parser.error("train-size must be >= 1000 and query-size must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if len(set(args.dtypes)) != len(args.dtypes):
        parser.error("dtypes must be distinct")
    if args.worker and (args.dtype is None or args.seed is None):
        parser.error("an internal worker requires dtype and seed")
    if args.worker:
        return _worker(args)
    return _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
