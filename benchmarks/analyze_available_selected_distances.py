"""Regenerate selected-distance comparisons from preserved JSONs using stdlib."""

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
STEM = "available-selected-distances-c02b71d"
SPECS = {
    "before_after": {
        "title": "Before/after", "file": f"{STEM}.zip",
        "sha256": "688f3256174668a05cffda7daaa1a1b28d8f8c0b38bdb5b87ceb9bde57269e20",
        "previous": "a25900033ab590ae8c8982e0d996f03cbc26591c",
        "current": "c02b71d3290245d7a131a25e82f80fa028c086bf", "run": "36788192361",
    },
    "same_code": {
        "title": "Same-code control", "file": "available-selected-distances-aa-6d63ae4.zip",
        "sha256": "8e5d83b344ad134a412030f693aa90b90dfbd6b5f45a5ad45bc88b0fb4b7b7a5",
        "previous": "c02b71d3290245d7a131a25e82f80fa028c086bf",
        "current": "6d63ae4e2b5847c92e18b50e05a7a2de05633f6a", "run": "36796444560",
    },
}
POLICIES, DTYPES = ("complete", "available"), ("float32", "float64")
VARIANTS, SEEDS, REPEATS = ("knn", "previous", "current"), (101, 202, 303), (1, 2, 3)
QUERIES = (300, 1000, 3000)
TIMINGS = ("fit_seconds", "transform_seconds", "total_seconds",
           "repeated_transform_median_seconds", "second_transform_seconds",
           "third_transform_seconds")
MEASURES = TIMINGS + ("worker_peak_rss_mib", "rmse", "mae", "scored_cells", "complete_donors")
ENVIRONMENT = ("git_commit", "github_run_id", "github_run_attempt", "python", "platform",
               "cpu_model", "logical_cpus", "affinity_cpus", "numpy", "scikit_learn", "faiss")
CONFIG = {
    "train_size": 20000, "features": 20, "n_neighbors": 5,
    "metric": "l2 / nan_euclidean", "weights": "uniform",
    "training_policies": list(POLICIES),
    "training_missing_rates": {"complete": 0.0, "available": 0.1},
    "dtypes": list(DTYPES), "query_pattern": "random", "threads": 1,
    "seeds": list(SEEDS), "repeats": 3, "repeated_transforms": 2,
    "sklearn_working_memory_mib": 256, "expected_workers": 108,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def finite(value, name, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid {name}: {value!r}")
    return value


def stats(values):
    values = list(values)
    return {"median": median(values), "min": min(values), "max": max(values)}


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "Invalid SHA-256 string")


def read_archive(directory, spec):
    candidates = [directory / name for name in (spec["file"], spec["file"] + ".zip")]
    found = [path for path in candidates if path.is_file()]
    require(len(found) == 1, f"Expected exactly one archive: {spec['file']} (or .zip.zip)")
    path = found[0].resolve()
    relative = path.relative_to(ROOT.resolve()).as_posix()
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == spec["sha256"], f"Archive hash mismatch: {relative}")
    with ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {f"version_comparison_q{q}.json" for q in QUERIES} | {
            "baseline-build.json", "candidate-build.json", "current-environment.txt",
            "previous-environment.txt", "shared-dependencies.txt",
        }
        require(len(names) == len(set(names)) and set(names) == expected,
                f"Unexpected/duplicate ZIP members: {relative}")
        members = {name: archive.read(name) for name in sorted(names)}
    versions = {role: "0.3.21+bench." + spec[role][:12] for role in ("previous", "current")}
    versions["knn"] = versions["current"]
    for role, name, field in (("previous", "baseline-build.json", "baseline_wheel_version"),
                              ("current", "candidate-build.json", "candidate_wheel_version")):
        build = decode(members[name])
        require(build["source_commit"] == spec[role] and build[field] == versions[role]
                and build["declared_version"] == "0.3.21", f"Invalid build provenance: {name}")
    freezes = {name: members[name].decode("utf-8").splitlines()
               for name in ("shared-dependencies.txt", "previous-environment.txt", "current-environment.txt")}
    shared = freezes["shared-dependencies.txt"]
    require(len(shared) == len(set(shared)) and all("==" in line for line in shared),
            "Invalid shared dependency freeze")
    dependencies = dict(line.split("==", 1) for line in shared)
    for role in ("previous", "current"):
        frozen = freezes[f"{role}-environment.txt"]
        wheel = [line for line in frozen if line.startswith("faiss-imputer @ ")]
        require(len(wheel) == 1 and "%2Bbench." + spec[role][:12] in wheel[0]
                and re.search(r"#sha256=[0-9a-f]{64}$", wheel[0]), "Unexpected installed wheel")
        require(sorted(line for line in frozen if line != wheel[0]) == sorted(shared),
                f"Dependency mismatch: {role}")
    provenance = {"path": relative, "archive_sha256": spec["sha256"],
                  "member_sha256": {name: sha256(raw).hexdigest() for name, raw in members.items()},
                  "builds": {name: decode(members[name]) for name in
                             ("baseline-build.json", "candidate-build.json")},
                  "environment_files": freezes}
    return members, versions, dependencies, provenance


def validate_member(data, query_count, spec, versions, dependencies):
    params, metadata, records = data["parameters"], data["metadata"], data["records"]
    require(data["schema_version"] == 1, "Unexpected raw schema")
    for field, value in {**CONFIG, "queries": query_count}.items():
        require(params[field] == value, f"Unexpected parameter: {field}")
    require(params["faiss_imputer_environment_versions"] == versions, "Version grid mismatch")
    require(metadata["git_commit"] == spec["current"] and metadata["github_run_id"] == spec["run"]
            and metadata["github_run_attempt"] == "1", "Run provenance mismatch")
    require(metadata["faiss_imputer"] == versions["current"], "Coordinator version mismatch")
    for field, package in (("numpy", "numpy"), ("scikit_learn", "scikit-learn"), ("faiss", "faiss-cpu")):
        require(metadata[field] == dependencies[package], f"Dependency metadata mismatch: {field}")
    expected = set(product(POLICIES, DTYPES, VARIANTS, SEEDS, REPEATS))
    require(len(records) == len(expected), "Incomplete worker grid")
    indexed, cases = {}, {}
    for position, record in enumerate(records):
        key = tuple(record[field] for field in ("training_policy", "dtype", "variant", "seed", "repeat"))
        require(key in expected and key not in indexed, f"Unexpected/duplicate record: {key}")
        policy, dtype, variant, seed, repeat = key
        require(record["status"] == "ok" and record["checks_passed"] is True, f"Failed worker: {key}")
        method = "KNNImputer" if variant == "knn" else f"FaissImputer[{policy}]"
        require(record["method"] == method and record["expected_version"] == versions[variant], "Method mismatch")
        required = {"size": CONFIG["train_size"], "queries": query_count, "pattern": "random",
                    "threads": 1, "prefix_sizes": [CONFIG["train_size"]], "repeated_transforms": 2,
                    "input_dtype": dtype, "output_dtype": dtype, "scored_cells": query_count * 4,
                    "query_missing_rate": 0.2, "faiss_omp_threads": 1}
        require(all(record[field] == value for field, value in required.items()), f"Record config mismatch: {key}")
        env = record["environment"]
        require(all(env[field] == metadata[field] for field in ENVIRONMENT), f"Worker environment mismatch: {key}")
        require(env["faiss_imputer"] == versions[variant], "Installed package mismatch")
        require(record["threadpools"] and all(pool["num_threads"] == 1 for pool in record["threadpools"]),
                "Unexpected native thread count")
        repeated = record["repeated_transform_seconds"]
        require(len(repeated) == 2, "Expected two additional transforms")
        for value in repeated:
            finite(value, "additional transform", positive=True)
        sample = {field: record[field] for field in MEASURES if field in record}
        sample.update(second_transform_seconds=repeated[0], third_transform_seconds=repeated[1])
        for field in MEASURES:
            finite(sample[field], field, positive=field in TIMINGS or field == "worker_peak_rss_mib")
        require(sample["total_seconds"] == sample["fit_seconds"] + sample["transform_seconds"], "Total mismatch")
        require(sample["repeated_transform_median_seconds"] == median(repeated), "Additional median mismatch")
        for field in ("max_abs_difference_from_knn", "max_abs_difference_from_previous_release",
                      "max_abs_difference_from_first_repeat"):
            finite(record[field], field)
        require(record["max_abs_difference_from_first_repeat"] == 0,
                "Output differs between transform calls")
        if variant in ("previous", "current"):
            require(record["max_abs_difference_from_previous_release"] == 0,
                    "Faiss baseline/candidate output difference detected")
        fingerprint = record["fingerprints"]
        for field in ("train", "query", "truth"):
            digest(fingerprint[field])
        require(fingerprint["prefixes"] == {str(CONFIG["train_size"]): fingerprint["train"]}, "Prefix mismatch")
        digest(record["output_sha256"])
        case = {field: record[field] for field in ("fingerprints", "complete_donors", "train_missing_rate",
                                                   "query_missing_rate", "query_patterns", "scored_cells")}
        case_key = (policy, dtype, seed)
        require(case_key not in cases or cases[case_key] == case, "Input triplets/repeats differ")
        cases[case_key] = case
        if policy == "complete":
            require(record["complete_donors"] == CONFIG["train_size"] and record["train_missing_rate"] == 0,
                    "Complete input mismatch")
        else:
            require(0 < record["complete_donors"] < CONFIG["train_size"] and 0 < record["train_missing_rate"] < 1,
                    "Available input mismatch")
        indexed[key] = {"record_index": position, "seed": seed, "repeat": repeat, **sample,
                        "output_sha256": record["output_sha256"], **case}
    require(set(indexed) == expected, "Missing worker records")
    return indexed, {field: metadata[field] for field in ENVIRONMENT}, params


def analyze(directory, name, spec):
    members, versions, dependencies, provenance = read_archive(directory, spec)
    cells, common_environment, configurations = [], None, []
    for query_count in QUERIES:
        member = f"version_comparison_q{query_count}.json"
        indexed, environment, params = validate_member(decode(members[member]), query_count, spec, versions, dependencies)
        require(common_environment is None or common_environment == environment, "Environment differs within run")
        common_environment = environment
        configurations.append(params)
        for policy, dtype in product(POLICIES, DTYPES):
            groups = {variant: [indexed[(policy, dtype, variant, seed, repeat)]
                               for seed, repeat in product(SEEDS, REPEATS)] for variant in VARIANTS}
            pairs = []
            for before, after in zip(groups["previous"], groups["current"]):
                require(before["output_sha256"] == after["output_sha256"], "Matched output hashes differ")
                require(before["rmse"] == after["rmse"] and before["mae"] == after["mae"], "Matched quality differs")
                pairs.append({"seed": before["seed"], "repeat": before["repeat"],
                              "previous_record_index": before["record_index"],
                              "current_record_index": after["record_index"],
                              "timing_ratios": {field: before[field] / after[field] for field in TIMINGS},
                              "peak_rss_delta_mib": after["worker_peak_rss_mib"] - before["worker_peak_rss_mib"]})
            cells.append({"run": name, "member": member, "api": "fit_then_transform",
                          "training_policy": policy, "dtype": dtype, "queries": query_count,
                          "record_count_per_variant": len(groups["previous"]), "matched_pair_count": len(pairs),
                          "methods": {variant: {field: stats(row[field] for row in rows) for field in MEASURES}
                                      for variant, rows in groups.items()},
                          "timing_ratios": {field: stats(pair["timing_ratios"][field] for pair in pairs) for field in TIMINGS},
                          "peak_rss_delta_mib": stats(pair["peak_rss_delta_mib"] for pair in pairs),
                          "samples": groups, "pairs": pairs})
    return {"specification": spec, "provenance": provenance, "environment": common_environment,
            "configurations": configurations, "versions": versions, "cells": cells,
            "successful_records": sum(cell["record_count_per_variant"] * len(VARIANTS) for cell in cells),
            "matched_pairs": sum(cell["matched_pair_count"] for cell in cells)}


def formatted(summary, digits=6):
    return f"{summary['median']:.{digits}f} [{summary['min']:.{digits}f}–{summary['max']:.{digits}f}]"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def report(result):
    runs = result["runs"]
    lines = ["# Available float64 selected-distance batching", "",
             "Selected candidates are refined in bounded pair blocks using the existing guarded distance kernel. "
             "These archives compare that change and a subsequent same-code control.", "", "## Sources and configuration", ""]
    for name, run in runs.items():
        spec, path = run["specification"], run["provenance"]["path"]
        lines += [f"- **{spec['title']}**: [{spec['previous'][:7]}]({REPO}/commit/{spec['previous']}) → "
                  f"[{spec['current'][:7]}]({REPO}/commit/{spec['current']}); "
                  f"[workflow run]({REPO}/actions/runs/{spec['run']}); [raw archive]({REPO}/blob/main/{path}).",
                  f"  {run['successful_records']} successful records; {run['matched_pairs']} matched baseline/candidate pairs. "
                  f"CPU: {run['environment']['cpu_model']}; Python {run['environment']['python']}; "
                  f"NumPy {run['environment']['numpy']}; scikit-learn {run['environment']['scikit_learn']}; "
                  f"Faiss {run['environment']['faiss']}."]
    config = runs["before_after"]["configurations"][0]
    lines += ["", f"Each case uses {config['train_size']:,} training rows, {config['features']} features, "
              f"k={config['n_neighbors']}, uniform weights, random query missingness, and one native thread. "
              "Complete-policy training is fully observed; available-policy training has a target missing rate of 10%. "
              "Query missingness is 20%. Therefore policy rows are distinct workloads.", "", "## Aggregation definitions", "",
              "All tables are recalculated from `records[]`; stored JSON summaries are not used. Only the "
              "`fit_then_transform` API is measured here. There is no same-data `fit_transform` comparison. "
              "Runs, dtypes, training policies and query sizes are never pooled.", "",
              "Times are median [min–max] in seconds across 9 records = 3 seeds × 3 fresh-worker repeats. "
              "`total_seconds` is fit plus the first transform. `transform_seconds` is the first transform alone. "
              "The second and third calls are the two ordered entries in `repeated_transform_seconds`. "
              "The additional-call summary first takes their median within each worker, then summarizes those nine worker medians. "
              "It is not a median over 18 independent timing trials.", "",
              "Each ratio is baseline time / candidate time for a pair matched within the same run, "
              "training configuration, policy, dtype, query size, seed and repeat, with identical input fingerprints. "
              "Reported ratios are median [min–max] of the nine paired ratios; values above 1 favor the candidate. "
              "They are not ratios of displayed median times. Each timing phase has its own ratios.", "",
              "Peak RSS is the full worker-lifetime peak in MiB. Memory deltas are candidate minus baseline for each "
              "matched pair, summarized as median [min–max]; no memory speedup is defined. RMSE and MAE are errors "
              "against synthetic ground truth at the scored missing cells. Quality and donor counts use the same nine "
              "records; repeated seeds are not nine independently generated datasets.", ""]
    for run in runs.values():
        lines += [f"## {run['specification']['title']}: timings", ""]
        for field, label in (("transform_seconds", "First transform"),
                             ("total_seconds", "Fit + first transform"),
                             ("repeated_transform_median_seconds", "Additional-call worker median")):
            lines += [f"### {label}", ""]
            lines += table(["Policy", "dtype", "Queries", "Baseline seconds", "Candidate seconds", "Paired ratio"],
                           [(cell["training_policy"], cell["dtype"], cell["queries"],
                             formatted(cell["methods"]["previous"][field]), formatted(cell["methods"]["current"][field]),
                             formatted(cell["timing_ratios"][field])) for cell in run["cells"]])
    lines += ["## Complete float64: ordered transform calls at 3,000 queries", ""]
    phase_rows = []
    for run in runs.values():
        cell = next(c for c in run["cells"] if (c["training_policy"], c["dtype"], c["queries"]) == ("complete", "float64", 3000))
        for field, label in (("transform_seconds", "First"), ("second_transform_seconds", "Second"),
                             ("third_transform_seconds", "Third"), ("repeated_transform_median_seconds", "Additional-call worker median")):
            phase_rows.append((run["specification"]["title"], label, formatted(cell["methods"]["previous"][field]),
                               formatted(cell["methods"]["current"][field]), formatted(cell["timing_ratios"][field])))
    lines += table(["Run", "Call", "Baseline seconds", "Candidate seconds", "Paired ratio"], phase_rows)
    lines += ["## Worker peak RSS", ""]
    lines += table(["Run", "Policy", "dtype", "Queries", "Baseline MiB", "Candidate MiB", "Paired delta MiB"],
                   [(run["specification"]["title"], c["training_policy"], c["dtype"], c["queries"],
                     formatted(c["methods"]["previous"]["worker_peak_rss_mib"], 3),
                     formatted(c["methods"]["current"]["worker_peak_rss_mib"], 3), formatted(c["peak_rss_delta_mib"], 3))
                    for run in runs.values() for c in run["cells"]])
    lines += ["## Reconstruction error and complete donor counts", "",
              "All matched Faiss baseline/candidate output SHA256 digests agree, and their recorded maximum "
              "output differences are zero in both runs. Their RMSE/MAE values are identical in these records. The table shows the "
              "candidate (also the baseline) and KNN errors separately. This is observed output and aggregate "
              "metric agreement on these inputs, not a general prediction or algorithmic equivalence claim.", ""]
    lines += table(["Run", "Policy", "dtype", "Queries", "Method", "RMSE", "MAE", "Scored cells", "Complete donors"],
                   [(run["specification"]["title"], c["training_policy"], c["dtype"], c["queries"],
                     "Faiss baseline/candidate" if variant == "current" else "KNNImputer",
                     *(formatted(c["methods"][variant][field], 8 if field in ("rmse", "mae") else 0)
                       for field in ("rmse", "mae", "scored_cells", "complete_donors")))
                    for run in runs.values() for c in run["cells"] for variant in ("current", "knn")])
    lines += ["## Interpretation and limits", ""]
    for name, run in runs.items():
        selected = [c for c in run["cells"] if (c["training_policy"], c["dtype"]) == ("available", "float64")]
        for field, label in (("transform_seconds", "first transform"), ("total_seconds", "fit plus first transform")):
            ratios = [c["timing_ratios"][field]["median"] for c in selected]
            lines += [f"- {run['specification']['title']}, available float64 {label}: "
                      f"the three configuration-level median paired ratios range from {min(ratios):.6f}× to {max(ratios):.6f}×."]
    lines += ["", "The ordered-call tables show the baseline third-call acceleration for complete float64 in both "
              "the change comparison and the same-code control. Thus the observed discrepancy can arise without "
              "the selected-distance code change. The experiment does not establish its underlying cause, universal "
              "absence of regressions, or an allocator explanation. Same-code memory differences likewise prevent "
              "attributing the before/after RSS delta directly to the optimization.", "",
              "The control is called same-code because the intervening commit only archived benchmark evidence. "
              "The archives identify commits, distribution versions and wheel hashes, but do not contain installed "
              "package source files; this analyzer does not cryptographically verify equality of package source. "
              "No cross-run timings are pooled or subtracted to correct the measured speedups.", "", "## Reproduction", "",
              f"Use the [analysis workflow]({REPO}/blob/main/.github/workflows/analyze-available-selected-distances.yml) "
              "from GitHub Actions → Run workflow. It reads preserved archives and uploads regenerated Markdown "
              "and JSON; it does not execute imputers or new timing measurements.", "",
              f"The [stdlib analysis script]({REPO}/blob/main/benchmarks/analyze_available_selected_distances.py) "
              "also supports `python benchmarks/analyze_available_selected_distances.py`. "
              f"The [unrounded summary]({REPO}/blob/main/benchmarks/results/{STEM}-summary.json) "
              "retains original float values, per-record source indexes, input fingerprints, paired ratios, "
              "ordered timings, build provenance, and archive/member SHA256 digests. JSON floats use Python's "
              "round-trip representation; rounding is applied only in this Markdown report.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks/results",
                        help="Directory inside the repository containing the preserved archives")
    parser.add_argument("--report", type=Path, default=ROOT / f"docs/benchmarks/{STEM}.md")
    parser.add_argument("--summary", type=Path, default=ROOT / f"benchmarks/results/{STEM}-summary.json")
    args = parser.parse_args()
    result = {"schema_version": 1, "aggregation": {
        "api": "fit_then_transform", "record_count_per_cell": 9,
        "timing": "median/min/max across three seeds and three repeats, separately per phase",
        "speedup": "median/min/max of matched baseline/candidate timing ratios",
        "additional_calls": "two ordered calls; worker median summarized across nine workers",
        "memory": "worker peak RSS; candidate-minus-baseline paired delta; no memory ratios",
        "matching": "run, full archived configuration, policy, dtype, query count, seed, repeat, fingerprints",
        "record_indexes": "zero-based positions in the source member's records array",
    }, "runs": {name: analyze(args.results_dir, name, spec) for name, spec in SPECS.items()}}
    markdown = report(result)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for path, text in ((args.report, markdown), (args.summary, encoded)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
