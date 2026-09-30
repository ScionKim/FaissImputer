"""Regenerate reports from archived transform profiles using only the stdlib."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import math
from pathlib import Path
import re
from statistics import median
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/ScionKim/FaissImputer"
COMMIT = "a25900033ab590ae8c8982e0d996f03cbc26591c"
VERSION = "0.3.21+bench.a25900033ab5"
CPU = "AMD EPYC 7763 64-Core Processor"
BASE = "benchmarks/results/available-transform-profile-a259000"
REPORT = "docs/benchmarks/available-transform-profile-a259000.md"
SUMMARY = "benchmarks/results/available-transform-profile-a259000-summary.json"
MEMBER = "available_transform_profile.json"
DATASETS = {"wine_quality_white": "wine-quality-white.zip", "abalone": "abalone.zip"}
LABELS = {"wine_quality_white": "Wine quality (white)", "abalone": "Abalone"}
DTYPES = ("float32", "float64")
SEEDS = (101, 202, 303)
CONFIG = {
    "mechanism": "MCAR", "n_train": 3000, "n_query": 1000,
    "target_missing_rate": 0.10, "n_neighbors": 5, "weights": "uniform",
    "metric": "l2", "strategy": "mean", "index_factory": "Flat",
    "donor_policy": "available", "api": "fit_then_transform",
    "timing_scope": "first transform only; fit excluded",
}
CHECKS = (
    "exact_reference_output_agreement_equal_nan", "shape_and_dtype_preserved",
    "observed_values_preserved", "finite_output", "inputs_unchanged",
    "query_cache_cleared", "patches_restored",
)
ENVIRONMENT = (
    "faiss_imputer", "python", "numpy", "scikit_learn", "faiss",
    "faiss_imputer_source_sha256", "matrix_source_sha256",
)
STAGES = {
    "transform_seconds": "Instrumented transform",
    "full_refinement_seconds": "Full-donor refinement (suspect + tie)",
    "matrix_seconds": "Initial distance matrix",
    "selected_refinement_seconds": "Selected-candidate refinement",
    "faiss_kmin_seconds": "Faiss candidate selection",
    "aggregate_seconds": "Aggregation calls",
    "matrix_prepare_self_seconds": "Matrix preparation: self",
    "search_self_seconds": "Search: self",
    "available_transform_self_seconds": "Available transform: self",
}
COUNTS = (
    "search_calls", "initial_batch_query_rows", "full_suspect_row_events",
    "full_tie_row_events", "full_row_events", "selected_row_events",
    "selected_donor_pair_events", "expansion_search_calls", "retain_calls",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, integer=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f"Invalid {name}: {value!r}")
    if integer:
        require(isinstance(value, int), f"Expected integer {name}")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA-256 string")
    return value


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def read_archive(path, dataset):
    raw = path.read_bytes()
    with ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), f"Duplicate ZIP members: {path.name}")
        raw_json = archive.read(MEMBER)
        provenance_bytes = archive.read("candidate-build.json")
        provenance = decode(provenance_bytes)
        require(provenance["source_commit"] == COMMIT, "Unexpected source commit")
        require(provenance["version"] == VERSION, "Unexpected candidate version")
        wheel = archive.read("wheels/" + provenance["wheel_filename"])
        require(sha256(wheel).hexdigest() == provenance["wheel_sha256"],
                f"Candidate wheel hash mismatch: {dataset}")
        dependency_hash = sha256(archive.read("recorded-dependencies.txt")).hexdigest()
        require(dependency_hash == provenance["recorded_dependencies_sha256"],
                f"Dependency list hash mismatch: {dataset}")
        dataset_metadata = decode(archive.read("dataset.json"))
        install_report = decode(archive.read("install-report.json"))
        installed = {item["metadata"]["name"].lower().replace("_", "-"): item["metadata"]["version"]
                     for item in install_report["install"]}
        freeze_bytes = archive.read("environment.txt")
        frozen = dict(line.split("==", 1) for line in freeze_bytes.decode("utf-8").splitlines()
                      if "==" in line)
        require(installed.get("faiss-imputer") == VERSION, "Installed candidate version differs")
    data = decode(raw_json)
    require(data["schema_version"] == 1 and data["status"] == "ok",
            f"Incomplete or unsupported profile: {dataset}")
    metadata = data["metadata"]
    require(metadata["provenance"] == provenance, "Embedded provenance differs")
    require(metadata["provenance_sha256"] == sha256(provenance_bytes).hexdigest(),
            "Provenance file hash mismatch")
    require(metadata["expected_installed_version"] == VERSION,
            "Unexpected installed version")
    grid = metadata["case_grid"]
    require(grid["dataset"] == dataset and grid["mechanism"] == "MCAR"
            and sorted(grid["dtypes"]) == sorted(DTYPES)
            and sorted(grid["seeds"]) == sorted(SEEDS), "Unexpected case grid")
    require(data["counts"] == {"records": 6, "ok": 6}, "Unexpected record counts")
    with ZipFile(BytesIO(wheel)) as package:
        for record in data["records"]:
            require(record["dataset"] == dataset_metadata, "Archived dataset metadata differs")
            require(record["case"]["fingerprints"]["dataset"] == dataset_metadata["dataset_sha256"],
                    "Prepared case refers to a different source dataset")
            require(record["case"]["feature_names"] == dataset_metadata["feature_names"]
                    and record["case"]["n_train"] == CONFIG["n_train"]
                    and record["case"]["n_query"] == CONFIG["n_query"]
                    and record["case"]["input_dtype"] == record["configuration"]["dtype"],
                    "Prepared case configuration differs")
            environment = record["environment"]
            for project, field in (("numpy", "numpy"), ("scikit-learn", "scikit_learn"), ("faiss-cpu", "faiss")):
                require(installed[project] == frozen[project] == environment[field],
                        f"Recorded dependency version differs: {project}")
            for source_name in ("faiss_imputer", "matrix"):
                location = environment[source_name + "_source_path"].replace("\\", "/")
                require("/site-packages/" in location, "Unexpected installed source path")
                member = location.split("/site-packages/", 1)[1]
                require(sha256(package.read(member)).hexdigest() == digest(environment[source_name + "_source_sha256"]),
                        "Installed module hash differs from archived wheel")
    return data, {
        "dataset": dataset, "path": f"{BASE}/{DATASETS[dataset]}",
        "archive_sha256": sha256(raw).hexdigest(),
        "json_member": MEMBER, "json_sha256": sha256(raw_json).hexdigest(),
        "provenance": provenance, "profile_script_sha256": metadata["profile_script_sha256"],
        "recorded_dependencies_sha256": dependency_hash,
        "dataset_metadata": dataset_metadata,
        "environment_sha256": sha256(freeze_bytes).hexdigest(),
    }


def derive(record):
    profile = record["profile"]
    timers, counters, events = (profile[key] for key in ("timings", "counters", "search_events"))
    for name, entry in timers.items():
        number(entry["calls"], name + ".calls", integer=True)
        number(entry["inclusive_seconds"], name + ".inclusive_seconds")
        number(entry["self_seconds"], name + ".self_seconds")
        require(entry["self_seconds"] <= entry["inclusive_seconds"] + 1e-8,
                f"Self time exceeds inclusive time: {name}")
    for name, value in counters.items():
        number(value, name, integer=True)

    def count(name):
        return counters.get(name, 0)

    def timer(name, kind="inclusive_seconds"):
        return timers.get(name, {}).get(kind, 0.0)

    require(bool(events) and count("search_calls") == len(events), "Search count mismatch")
    for event in events:
        require(event["status"] == "ok" and event["cache_identity_after"] is True,
                "Unsuccessful search event")
        for field in ("query_rows", "requested_k", "suspect_full_row_events",
                      "tie_full_row_events", "selected_row_events"):
            number(event[field], "search event " + field, integer=True)
        require(event["query_rows"] > 0 and event["requested_k"] > 0,
                "Empty search event")
        require(isinstance(event["cache_hit_before"], bool), "Invalid cache-hit flag")
    for field, counter, bucket in (
        ("suspect_full_row_events", "full_suspect_row_events", "full_direct_distances.suspect"),
        ("tie_full_row_events", "full_tie_row_events", "full_direct_distances.tie"),
        ("selected_row_events", "selected_row_events", "direct_distance_kernel.selected"),
    ):
        total = sum(event[field] for event in events)
        require(count(counter) == total, f"Row event count mismatch: {counter}")
        require(timers.get(bucket, {}).get("calls", 0) == total,
                f"Timer call count mismatch: {bucket}")
    expansions = sum(event["cache_hit_before"] for event in events)
    require(count("expansion_search_calls") == expansions
            and count("cache_hit_search_calls") == expansions, "Expansion count mismatch")
    require(count("initial_batch_query_rows") == sum(
        event["query_rows"] for event in events if not event["cache_hit_before"]),
        "Initial query row count mismatch")
    require(count("search_query_row_events") == sum(event["query_rows"] for event in events),
            "Search query row count mismatch")
    require(count("requested_candidate_pair_events") == sum(
        event["query_rows"] * min(event["requested_k"], record["donor_rows"]) for event in events),
        "Requested candidate pair count mismatch")
    require(count("retain_calls") == len(profile["retain_events"]), "Retain count mismatch")
    require(timers.get("retain_queries", {}).get("calls", 0) == count("retain_calls")
            and count("retained_query_row_events") == sum(event["rows_retained"] for event in profile["retain_events"]),
            "Retain timer or retained row count mismatch")
    selected_rows = count("selected_row_events")
    selected_pairs = count("selected_donor_pair_events")
    require((selected_rows == 0 and selected_pairs == 0)
            or (selected_rows > 0 and selected_rows <= selected_pairs <= selected_rows * record["donor_rows"]),
            "Selected candidate pair count is missing or inconsistent")
    require(timers["instrumented_transform"]["calls"] == 1, "Unexpected transform count")
    root = record["instrumented_transform_seconds"]
    number(root, "instrumented_transform_seconds")
    require(root > 0 and root == timer("instrumented_transform"), "Root timing mismatch")
    accounting = profile["accounting"]
    self_sum = sum(entry["self_seconds"] for entry in timers.values())
    require(abs(root - self_sum) <= 1e-8
            and accounting["root_inclusive_seconds"] == root
            and abs(accounting["sum_of_all_traced_self_seconds"] - self_sum) <= 1e-8
            and abs(accounting["root_minus_self_sum_seconds"] - (root - self_sum)) <= 1e-8,
            "Unbalanced nested timer accounting")
    for required in ("available_transform", "search", "prepare_search_matrix",
                     "prepared_matrix_distances", "faiss_kmin", "aggregate"):
        require(required in timers, f"Missing required timer: {required}")
    values = {
        "transform_seconds": root,
        "full_refinement_seconds": timer("full_direct_distances.suspect") + timer("full_direct_distances.tie"),
        "matrix_seconds": timer("prepared_matrix_distances"),
        "selected_refinement_seconds": timer("direct_distance_kernel.selected"),
        "faiss_kmin_seconds": timer("faiss_kmin"), "aggregate_seconds": timer("aggregate"),
        "matrix_prepare_self_seconds": timer("prepare_search_matrix", "self_seconds"),
        "search_self_seconds": timer("search", "self_seconds"),
        "available_transform_self_seconds": timer("available_transform", "self_seconds"),
    }
    values.update({key.removesuffix("_seconds") + "_fraction": value / root
                   for key, value in list(values.items())})
    values["reference_transform_seconds"] = number(record["reference_transform_seconds"], "reference time")
    values.update({name: count(name) for name in COUNTS if name != "full_row_events"})
    values["full_row_events"] = count("full_suspect_row_events") + count("full_tie_row_events")
    for field in ("rmse", "mae", "scored_cells"):
        values[field] = number(record["quality"][field], field, integer=field == "scored_cells")
    require(values["scored_cells"] > 0, "No scored ground-truth cells")
    return values


def analyze(input_dir):
    sources, samples, common = [], [], None
    for dataset, filename in DATASETS.items():
        data, source = read_archive(input_dir / filename, dataset)
        records = data["records"]
        keys = [(record["configuration"]["dtype"], record["configuration"]["seed"]) for record in records]
        require(len(keys) == 6 and set(keys) == set(product(DTYPES, SEEDS)),
                f"Duplicate, missing, or unexpected records: {dataset}")
        for record in sorted(records, key=lambda value: (value["configuration"]["dtype"], value["configuration"]["seed"])):
            config, checks, environment = (record[key] for key in ("configuration", "checks", "environment"))
            require(config["dataset"] == dataset and all(config[key] == value for key, value in CONFIG.items()),
                    f"Unexpected configuration: {dataset}")
            require(record["status"] == "ok" and record["checks_passed"] is True,
                    "Failed record cannot be summarized")
            require(all(checks[key] is True for key in CHECKS), "Failed instrumentation check")
            require(checks["input_hashes_before"] == checks["input_hashes_after"], "Input hashes differ")
            require(checks["reference_output_sha256"] == checks["instrumented_output_sha256"], "Output hashes differ")
            for value in (*checks["input_hashes_before"].values(), checks["reference_output_sha256"]):
                digest(value)
            require(environment["faiss_imputer"] == VERSION and record["donor_rows"] == CONFIG["n_train"],
                    "Unexpected package version or donor count")
            require(record["cpu"]["processor"] == CPU and record["cpu"]["native_thread_limit"] == 1
                    and all(pool["num_threads"] == 1 for pool in record["threadpools_during_transform"]),
                    "Unexpected recorded native thread count")
            identity = {key: environment[key] for key in ENVIRONMENT}
            identity.update(cpu_model=record["cpu"]["processor"],
                            case_helper_sha256=digest(record["case_helper"]["sha256"]),
                            profile_script_sha256=digest(source["profile_script_sha256"]),
                            recorded_dependencies_sha256=source["recorded_dependencies_sha256"])
            if common is None:
                common = identity
            require(identity == common, "Recorded CPU model, source, or dependencies differ")
            require(record["source_contract"]["name"] == "available-search-query-cache-and-refinement-v1",
                    "Unexpected profiling source contract")
            samples.append({
                "configuration": config, "values": derive(record),
                "timings": record["profile"]["timings"], "counters": record["profile"]["counters"],
                "search_events": record["profile"]["search_events"], "retain_events": record["profile"]["retain_events"],
                "accounting": record["profile"]["accounting"], "checks": checks,
                "case": record["case"], "quality": record["quality"], "cpu": record["cpu"],
            })
        sources.append(source)
    cells = []
    for dataset, dtype in product(DATASETS, DTYPES):
        selected = [row for row in samples if row["configuration"]["dataset"] == dataset
                    and row["configuration"]["dtype"] == dtype]
        cells.append({"dataset": dataset, "dtype": dtype, "records": len(selected), "seeds": list(SEEDS),
                      "statistics": {key: stats([row["values"][key] for row in selected])
                                     for key in selected[0]["values"]}})
    return {"schema_version": 1, "source_commit": COMMIT, "common_environment": common,
            "configuration": CONFIG, "sources": sources, "cells": cells, "samples": samples,
            "aggregation": {
                "cell": "dataset and dtype; three seeds, one traced execution per seed; no API/dtype pooling",
                "statistics": "median, min, max across the three seed records; no rounding in JSON",
                "stage_fraction": "stage seconds / instrumented_transform_seconds for each record, then median/min/max",
                "full_refinement_seconds": "sum of sibling full_direct_distances.suspect and .tie inclusive_seconds",
                "timer_hierarchy": "inclusive buckets overlap; do not add full kernels to their full-refinement parents",
                "timing_status": "instrumented diagnostic/reference-only; fit excluded; no benchmark speedups",
                "quality": "RMSE/MAE against masked standardized held-out ground truth, not a KNN comparison",
            }}


def format_stat(value, scale=1.0, digits=2):
    return (f"{value['median'] * scale:.{digits}f} "
            f"[{value['min'] * scale:.{digits}f}\u2013{value['max'] * scale:.{digits}f}]")


def render_report(summary):
    cells, environment = summary["cells"], summary["common_environment"]
    lines = ["# Available-transform profiles: Wine quality and Abalone", "",
             f"Source: [`{COMMIT[:7]}`]({REPO}/commit/{COMMIT}); candidate `{VERSION}`.", "",
             f"Recorded CPU model: **{environment['cpu_model']}**. Matching CPU labels do not identify the same physical runner.", "",
             f"Recorded environment: Python {environment['python']}, NumPy {environment['numpy']}, "
             f"scikit-learn {environment['scikit_learn']}, Faiss {environment['faiss']}; native threads: 1.", "",
             "## Configuration and aggregation", "",
             "Both datasets use 3,000 training rows, 1,000 held-out queries, k=5, 10% target MCAR missingness, "
             "uniform weights, l2 distance, mean aggregation, a Flat index, and available donors.", "",
             "The API is `fit_then_transform`; only the first transform is timed, with fit excluded. "
             "Each dataset/dtype cell has **3 records: seeds 101, 202, 303, with one traced execution per seed**. "
             "The datasets and dtypes are kept separate. Ranges reflect different seeded cases, not repeated timing trials.", "",
             "Every table summary is median [min\u2013max] across those three records. Time values are converted "
             "from seconds to milliseconds only for display. For a stage's percentage, divide its seconds by "
             "`instrumented_transform_seconds` within each record, then summarize the three fractions. "
             "Percentages are not ratios of displayed median times.", "",
             "The untraced reference transform runs first and warms the process; a separately fitted model is "
             "then traced. Wrappers add overhead. These diagnostic times do not establish benchmark speedups, "
             "and reference/traced timing ratios are not reported.", "",
             "### Timing field definitions", "",
             "All timer fields below are under `records[].profile.timings`. Except the root, fields use "
             "`inclusive_seconds` unless marked `self_seconds`.", "",
             "| Report field | Raw field or calculation |", "|---|---|",
             "| Instrumented transform | `records[].instrumented_transform_seconds`, equal to `instrumented_transform.inclusive_seconds` |",
             "| Full-donor refinement | `full_direct_distances.suspect.inclusive_seconds + full_direct_distances.tie.inclusive_seconds` |",
             "| Initial distance matrix | `prepared_matrix_distances.inclusive_seconds` |",
             "| Selected-candidate refinement | `direct_distance_kernel.selected.inclusive_seconds` |",
             "| Faiss candidate selection | `faiss_kmin.inclusive_seconds` |",
             "| Aggregation calls | `aggregate.inclusive_seconds` |",
             "| Matrix preparation: self | `prepare_search_matrix.self_seconds` |",
             "| Search: self | `search.self_seconds` |",
             "| Available transform: self | `available_transform.self_seconds` |", "",
             "Inclusive buckets overlap. Full-donor refinement already contains its direct-distance kernel; "
             "that child is not added again. Self time subtracts immediate traced children and includes tracing "
             "overhead and surrounding production work. The selected stages below are not a complete additive breakdown. "
             "Absent optional refinement buckets count as zero only when matching recorded row events are zero.", ""]
    lines += ["Dataset feature counts: " + "; ".join(
        f"{LABELS[source['dataset']]} {source['dataset_metadata']['features']}"
        for source in summary["sources"]) + ".", ""]
    for title, fraction in (("Stage times (ms)", False), ("Stage shares of instrumented transform (%)", True)):
        lines += [f"### {title}", "", "| Stage | Wine f32 | Wine f64 | Abalone f32 | Abalone f64 |",
                  "|---|---:|---:|---:|---:|"]
        for key, label in STAGES.items():
            field = key.removesuffix("_seconds") + "_fraction" if fraction else key
            values = [format_stat(cell["statistics"][field], 100 if fraction else 1000) for cell in cells]
            lines.append("| " + label + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Search and refinement events", "",
              "Counts are row events across search calls, not generally unique query rows. Full = suspect + tie. "
              "Expansion calls are cache-hit search calls. Values come from `profile.counters`, checked against "
              "`profile.search_events`; `requested_k` lists every call in order.", "",
              "| Dataset | dtype | Seed | Initial query rows | Suspect | Tie | Full | Selected rows | Selected pairs | Searches | Expansion | Retain | requested_k |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["samples"]:
        config, values = row["configuration"], row["values"]
        fields = ("initial_batch_query_rows", "full_suspect_row_events", "full_tie_row_events",
                  "full_row_events", "selected_row_events", "selected_donor_pair_events",
                  "search_calls", "expansion_search_calls", "retain_calls")
        entries = [LABELS[config["dataset"]], config["dtype"], str(config["seed"])]
        entries += [str(values[key]) for key in fields]
        entries.append(", ".join(str(event["requested_k"]) for event in row["search_events"]))
        lines.append("| " + " | ".join(entries) + " |")
    lines += ["", "## Instrumentation fidelity and reconstruction error", "",
              "All 12 archived records have status `ok`, `checks_passed=true`, matching stored reference/traced "
              "output hashes, unchanged input hashes, and passing shape/dtype, observed-value, finite-output, "
              "cache-cleanup, and wrapper-restoration checks. These checks establish instrumentation fidelity "
              "for the recorded fixtures, not general algorithmic correctness or agreement with KNNImputer.", "",
              "RMSE and MAE are `records[].quality.rmse` and `.mae`: reconstruction error against masked "
              "held-out ground truth, standardized using observed training values. Each metric below is "
              "median [min\u2013max] across seeds. Scored cells vary by seed; errors are not pooled across cells.", "",
              "| Dataset | dtype | RMSE | MAE | Scored cells |", "|---|---|---:|---:|---:|"]
    for cell in cells:
        stat = cell["statistics"]
        lines.append(f"| {LABELS[cell['dataset']]} | {cell['dtype']} | {format_stat(stat['rmse'], digits=9)} "
                     f"| {format_stat(stat['mae'], digits=9)} | {format_stat(stat['scored_cells'], digits=0)} |")
    lines += ["", "## Interpretation and next investigation", "",
              "The largest listed main-stage bucket in each cell is determined from its median time:", ""]
    for cell in cells:
        stat = cell["statistics"]
        key = max(("full_refinement_seconds", "matrix_seconds", "selected_refinement_seconds",
                   "faiss_kmin_seconds", "aggregate_seconds"), key=lambda field: stat[field]["median"])
        fraction = key.removesuffix("_seconds") + "_fraction"
        lines.append(f"- {LABELS[cell['dataset']]} {cell['dtype']}: {STAGES[key]}, "
                     f"{stat[key]['median'] * 1000:.2f} ms median; "
                     f"{stat[fraction]['median'] * 100:.2f}% median per-record share.")
    expansions = sum(row["values"]["expansion_search_calls"] for row in summary["samples"])
    requested = sorted({event["requested_k"] for row in summary["samples"] for event in row["search_events"]})
    lines += ["", f"Recorded expansion search calls: {expansions}. Requested candidate counts across all calls: "
              + ", ".join(str(value) for value in requested) + ".", "",
              "Read the costs separately for each dataset/dtype. Bounded batching of selected-candidate "
              "float64 distance recomputation is a shared optimization candidate. Full-donor refinement "
              "needs separate attention wherever its recorded share is large. Preserve numerical guards, "
              "overflow repair, distance semantics, and deterministic neighbor selection. The profiles do "
              "not predict a speedup: evaluation requires correctness checks and uninstrumented measurements.", "",
              "## Evidence and reproduction", "",
              f"[Analysis script]({REPO}/blob/main/benchmarks/analyze_available_transform_profile.py) \u00b7 "
              f"[Exact summary]({REPO}/blob/main/{SUMMARY}) \u00b7 "
              f"[Analysis workflow]({REPO}/blob/main/.github/workflows/analyze-available-transform-profile.yml)", "",
              "Run **Analyze available-transform profiles** in GitHub Actions. It reads the two preserved "
              "ZIPs using the Python standard library and writes this report plus the JSON summary. "
              "The summary retains unrounded binary64 calculation values, per-seed derived measurements, "
              "original timer buckets/counters/search events, checks, case fingerprints, and provenance. "
              "No imputer or dataset preparation is executed and the raw ZIPs are never modified.", ""]
    for source in summary["sources"]:
        provenance = source["provenance"]
        lines += [f"### {LABELS[source['dataset']]}", "",
                  f"[Preserved ZIP]({REPO}/blob/main/{source['path']}) \u00b7 "
                  f"[Profiling run]({REPO}/actions/runs/{provenance['github_run_id']}) (attempt {provenance['github_run_attempt']})", "",
                  f"- ZIP SHA-256: `{source['archive_sha256']}`",
                  f"- `{MEMBER}` SHA-256: `{source['json_sha256']}`",
                  f"- Candidate wheel SHA-256: `{provenance['wheel_sha256']}`", ""]
    lines += ["Source hashes shared by the archived records:", ""]
    for key in ("faiss_imputer_source_sha256", "matrix_source_sha256", "case_helper_sha256",
                "profile_script_sha256", "recorded_dependencies_sha256"):
        lines.append(f"- `{key}`: `{environment[key]}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / BASE)
    parser.add_argument("--report", type=Path, default=ROOT / REPORT)
    parser.add_argument("--summary", type=Path, default=ROOT / SUMMARY)
    args = parser.parse_args()
    summary = analyze(args.input_dir)
    report = render_report(summary)
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for path, text in ((args.report, report), (args.summary, serialized)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(f"Validated 12 records; wrote {args.report} and {args.summary}")


if __name__ == "__main__":
    main()
