"""Generate reproducible OFAT reports from the twelve preserved JSON files."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "9683d032089dc8c0f6882dc37cf639cdda5fe866"
RAW_DIR = ROOT / "benchmarks/results/ofat-2026-09-22"
REPORT_PATH = ROOT / "docs/benchmarks/fit-transform-ofat-9683d03.md"
SUMMARY_PATH = ROOT / "benchmarks/results/ofat-2026-09-22-summary.json"

APIS = ("fit_transform", "fit_then_transform")
DTYPES = ("float32", "float64")
SEEDS = (101, 202, 303)
KNN = "KNNImputer"
COMPLETE = "FaissImputer[complete]"
AVAILABLE = "FaissImputer[available]"
METHODS = (KNN, COMPLETE, AVAILABLE)

# filename, sizes, features, neighbors, missing rate, pattern, CPU family, repeats
RUNS = (
    ("01_baseline_10k.json", (10000,), 20, 5, 0.10, "mcar", "amd", 3),
    ("02_rows_1k-3k-5k-20k.json", (1000, 3000, 5000, 20000),
     20, 5, 0.10, "mcar", "amd", 3),
    ("03_features_10.json", (10000,), 10, 5, 0.10, "mcar", "amd", 3),
    ("04_features_50.json", (10000,), 50, 5, 0.10, "mcar", "amd", 3),
    ("05_features_100.json", (10000,), 100, 5, 0.10, "mcar", "amd", 3),
    ("06_missing_05.json", (10000,), 20, 5, 0.05, "mcar", "amd", 3),
    ("07_missing_20.json", (10000,), 20, 5, 0.20, "mcar", "amd", 3),
    ("08_pattern_mar.json", (10000,), 20, 5, 0.10, "mar", "amd", 3),
    ("09_neighbors_1.json", (10000,), 20, 1, 0.10, "mcar", "amd", 3),
    ("10_neighbors_15.json", (10000,), 20, 15, 0.10, "mcar", "amd", 3),
    ("11_neighbors_30_intel.json", (10000,), 20, 30, 0.10, "mcar", "intel", 3),
    ("12_stress_50k_intel.json", (50000,), 20, 5, 0.10, "mcar", "intel", 1),
)

MATCH_FIELDS = (
    "size",
    "features",
    "n_neighbors",
    "target_missing_rate",
    "missing_pattern",
    "dtype",
    "api",
    "seed",
    "repeat",
)

DATA_FIELDS = (
    "actual_missing_rate",
    "guaranteed_complete_rows",
    "complete_donors",
    "missing_patterns",
    "scored_cells",
    "quality_scope",
)

EXECUTION_FIELDS = (
    "threads",
    "faiss_omp_threads",
    "sklearn_working_memory_mib",
    "expected_version",
    "measure_phase_memory",
    "threadpools",
)

ENVIRONMENT_FIELDS = (
    "git_commit",
    "github_run_id",
    "github_run_attempt",
    "cpu_model",
    "python",
    "platform",
    "logical_cpus",
    "affinity_cpus",
    "faiss_imputer",
    "numpy",
    "scikit_learn",
    "faiss",
)

PHASE_MEMORY_FIELDS = (
    "rss_before_fit_mib",
    "rss_after_fit_mib",
    "fit_retained_rss_mib",
    "fit_phase_peak_rss_mib",
    "transform_phase_peak_rss_mib",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = list(values)
    require(bool(values), "Cannot summarize an empty group")
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ),
        "Non-numeric or non-finite measurement",
    )
    return {
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def format_stats(values, precision=".6f", scale=1):
    parts = [
        format(values[key] * scale, precision)
        for key in ("median", "min", "max")
    ]
    return f"{parts[0]} [{parts[1]}–{parts[2]}]"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def matching_key(filename, record):
    return (filename,) + tuple(record[field] for field in MATCH_FIELDS)


def validate_pair(left, right, context):
    for field in DATA_FIELDS + EXECUTION_FIELDS:
        require(left[field] == right[field], f"{context}: different {field}")
    for field in ("input", "truth", "missing"):
        require(
            left["fingerprints"][field] == right["fingerprints"][field],
            f"{context}: different {field} fingerprint",
        )
    for field in ENVIRONMENT_FIELDS:
        require(
            left["environment"][field] == right["environment"][field],
            f"{context}: different environment.{field}",
        )


def load_and_analyze():
    inputs = []
    cells = []
    datasets = []
    total_records = 0

    for filename, sizes, features, neighbors, rate, pattern, cpu, repeats in RUNS:
        path = RAW_DIR / filename
        raw = path.read_bytes()
        document = json.loads(raw)
        parameters = document["parameters"]
        metadata = document["metadata"]
        provenance = document["provenance"]

        require(document["schema_version"] == 1, f"{filename}: schema version")
        require(
            provenance["source_commit"] == SOURCE_COMMIT
            and metadata["git_commit"] == SOURCE_COMMIT,
            f"{filename}: unexpected source commit",
        )
        require(
            provenance["github_run_id"] == metadata["github_run_id"]
            and provenance["github_run_attempt"] == metadata["github_run_attempt"],
            f"{filename}: inconsistent run provenance",
        )

        expected = {
            "sizes": list(sizes),
            "features": features,
            "n_neighbors": neighbors,
            "target_missing_rate": rate,
            "missing_pattern": pattern,
            "guaranteed_complete_rows": neighbors,
            "repeats": repeats,
            "seeds": list(SEEDS),
            "apis": list(APIS),
            "dtypes": list(DTYPES),
            "methods": list(METHODS),
            "threads": 1,
            "sklearn_working_memory_mib": 256,
            "phase_memory_measured": False,
        }
        for field, value in expected.items():
            require(parameters[field] == value, f"{filename}: {field}")

        cpu_name = metadata["cpu_model"].lower()
        expected_cpu = "epyc 7763" if cpu == "amd" else "8573c"
        require(expected_cpu in cpu_name, f"{filename}: unexpected CPU")

        expected_count = len(sizes) * len(DTYPES) * len(APIS) * len(METHODS)
        expected_count *= len(SEEDS) * repeats
        require(
            len(document["records"]) == expected_count
            and parameters["expected_workers"] == expected_count,
            f"{filename}: wrong record count",
        )

        inputs.append({
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "record_count": expected_count,
            "metadata": metadata,
            "provenance": provenance,
            "parameters": parameters,
        })

        groups = defaultdict(list)
        data_references = {}
        expected_seed_repeats = set(
            itertools.product(SEEDS, range(1, repeats + 1))
        )

        for index, original in enumerate(document["records"], start=1):
            record = dict(original, record_index=index)
            context = f"{filename}: record {index}"

            require(record["status"] == "ok", f"{context}: failed worker")
            require(record["checks_passed"] is True, f"{context}: failed checks")
            require(record["measure_phase_memory"] is False, context)
            require(
                all(record[field] is None for field in PHASE_MEMORY_FIELDS),
                f"{context}: unexpected phase-memory measurement",
            )
            require(record["method"] in METHODS, f"{context}: method")
            require(record["api"] in APIS, f"{context}: API")
            require(record["dtype"] in DTYPES, f"{context}: dtype")
            require(record["size"] in sizes, f"{context}: size")
            require(record["input_dtype"] == record["dtype"], context)
            require(record["output_dtype"] == record["dtype"], context)
            require(record["quality_scope"] == "masked training entries", context)

            for field in (
                "features", "n_neighbors", "target_missing_rate",
                "missing_pattern", "guaranteed_complete_rows",
                "threads", "sklearn_working_memory_mib",
            ):
                require(record[field] == parameters[field], f"{context}: {field}")

            require(record["faiss_omp_threads"] == 1, context)
            require(record["expected_version"] == provenance["version"], context)
            for field in ENVIRONMENT_FIELDS:
                require(
                    record["environment"][field] == metadata[field],
                    f"{context}: environment.{field}",
                )

            for field in ("total_seconds", "rmse", "mae", "worker_peak_rss_mib"):
                value = record[field]
                stats([value])
                require(value >= 0, f"{context}: negative {field}")
            require(record["total_seconds"] > 0, f"{context}: zero timing")

            if record["api"] == "fit_transform":
                component_total = record["fit_transform_seconds"]
            else:
                component_total = record["fit_seconds"] + record["transform_seconds"]
            require(
                record["total_seconds"] == component_total,
                f"{context}: inconsistent timing components",
            )

            group_key = (
                record["size"], record["dtype"], record["api"], record["method"]
            )
            groups[group_key].append(record)

            data_key = (record["size"], record["dtype"], record["seed"])
            if data_key in data_references:
                validate_pair(data_references[data_key], record, context)
            else:
                data_references[data_key] = record

        expected_groups = set(itertools.product(sizes, DTYPES, APIS, METHODS))
        require(set(groups) == expected_groups, f"{filename}: incomplete groups")
        for key, records in groups.items():
            observed = {(r["seed"], r["repeat"]) for r in records}
            require(
                observed == expected_seed_repeats
                and len(records) == len(expected_seed_repeats),
                f"{filename}: missing or duplicate seed/repeat in {key}",
            )

        for size, dtype in itertools.product(sizes, DTYPES):
            seed_records = [
                data_references[(size, dtype, seed)] for seed in SEEDS
            ]
            datasets.append({
                "file": filename,
                "size": size,
                "dtype": dtype,
                "unique_seeds": len(SEEDS),
                "guaranteed_complete_rows": neighbors,
                "complete_donors": stats(r["complete_donors"] for r in seed_records),
                "actual_missing_rate": stats(
                    r["actual_missing_rate"] for r in seed_records
                ),
                "missing_patterns": stats(r["missing_patterns"] for r in seed_records),
                "scored_cells": stats(r["scored_cells"] for r in seed_records),
                "seeds": [
                    {
                        "seed": r["seed"],
                        "record_index": r["record_index"],
                        "fingerprints": r["fingerprints"],
                        **{field: r[field] for field in DATA_FIELDS},
                    }
                    for r in seed_records
                ],
            })

        for size, dtype, api in itertools.product(sizes, DTYPES, APIS):
            cell = {
                "file": filename,
                "size": size,
                "features": features,
                "n_neighbors": neighbors,
                "target_missing_rate": rate,
                "missing_pattern": pattern,
                "cpu_model": metadata["cpu_model"],
                "dtype": dtype,
                "api": api,
                "n": len(expected_seed_repeats),
                "methods": {},
                "speedups": {},
            }

            for method in METHODS:
                records = sorted(
                    groups[(size, dtype, api, method)],
                    key=lambda r: (r["seed"], r["repeat"]),
                )
                cell["methods"][method] = {
                    "n_records": len(records),
                    **{
                        field: stats(r[field] for r in records)
                        for field in (
                            "total_seconds", "rmse", "mae", "worker_peak_rss_mib"
                        )
                    },
                    "record_indices": [r["record_index"] for r in records],
                }

            reference = {
                matching_key(filename, r): r
                for r in groups[(size, dtype, api, KNN)]
            }
            for method in (COMPLETE, AVAILABLE):
                candidate = {
                    matching_key(filename, r): r
                    for r in groups[(size, dtype, api, method)]
                }
                require(reference.keys() == candidate.keys(), "Unmatched records")
                pairs = []
                for key in sorted(reference):
                    left, right = reference[key], candidate[key]
                    validate_pair(left, right, f"{filename}: {key}")
                    pairs.append({
                        "seed": left["seed"],
                        "repeat": left["repeat"],
                        "knn_record_index": left["record_index"],
                        "faiss_record_index": right["record_index"],
                        "knn_total_seconds": left["total_seconds"],
                        "faiss_total_seconds": right["total_seconds"],
                        "ratio": left["total_seconds"] / right["total_seconds"],
                        "absolute_rmse_difference": abs(left["rmse"] - right["rmse"]),
                        "absolute_mae_difference": abs(left["mae"] - right["mae"]),
                    })
                cell["speedups"][method] = {
                    "n_pairs": len(pairs),
                    **stats(pair["ratio"] for pair in pairs),
                    "pairs": pairs,
                }
            cells.append(cell)

        total_records += expected_count

    require(total_records == 1548, "Unexpected total record count")
    require(len(cells) == 60, "Unexpected API/dtype/configuration cell count")

    agreement = []
    for api, dtype in itertools.product(APIS, DTYPES):
        pairs = [
            {"file": cell["file"], "size": cell["size"], **pair}
            for cell in cells
            if cell["api"] == api and cell["dtype"] == dtype
            for pair in cell["speedups"][AVAILABLE]["pairs"]
        ]
        item = {"api": api, "dtype": dtype, "n_pairs": len(pairs)}
        for metric in ("rmse", "mae"):
            field = f"absolute_{metric}_difference"
            metric_stats = stats(pair[field] for pair in pairs)
            item[field] = {
                **metric_stats,
                "maximum_pairs": [
                    pair for pair in pairs if pair[field] == metric_stats["max"]
                ],
            }
        agreement.append(item)

    return {
        "schema_version": 1,
        "benchmark_source_commit": SOURCE_COMMIT,
        "methodology": {
            "timing_field": "records[].total_seconds",
            "timing_summary": "median [min–max] across seeds and repeats",
            "speedup": "median of matched KNN total_seconds / Faiss total_seconds",
            "matching_fields": ["input file/run"] + list(MATCH_FIELDS),
            "record_indices": "1-based positions in the input file's records array",
            "api_pooling": False,
            "dtype_pooling": False,
            "non_stress_records_per_method_cell": 9,
            "stress_records_per_method_cell": 3,
            "unique_seeds_per_configuration_and_dtype": 3,
            "numeric_precision": (
                "JSON numbers retain Python float round-trip precision; "
                "presentation rounding is applied only to Markdown."
            ),
        },
        "validation": {
            "input_files": len(inputs),
            "records": total_records,
            "timing_cells": len(cells) * len(METHODS),
            "speedup_cells": len(cells) * 2,
            "matched_pairs": sum(
                result["n_pairs"]
                for cell in cells
                for result in cell["speedups"].values()
            ),
        },
        "inputs": inputs,
        "datasets": datasets,
        "cells": cells,
        "available_metric_agreement": agreement,
    }


def make_report(result):
    index = {
        (cell["file"], cell["size"], cell["api"], cell["dtype"]): cell
        for cell in result["cells"]
    }
    baseline = (RUNS[0][0], 10000, "Baseline")

    sections = [
        ("Baseline", [baseline]),
        ("Rows sweep — AMD", [
            (RUNS[1][0], n, f"{n:,} rows") if n != 10000
            else (RUNS[0][0], n, "10,000 rows · baseline")
            for n in (1000, 3000, 5000, 10000, 20000)
        ]),
        ("Features sweep — AMD", [
            (RUNS[2][0], 10000, "10 features"),
            (RUNS[0][0], 10000, "20 features · baseline"),
            (RUNS[3][0], 10000, "50 features"),
            (RUNS[4][0], 10000, "100 features"),
        ]),
        ("Missing-rate sweep — AMD", [
            (RUNS[5][0], 10000, "5% MCAR"),
            (RUNS[0][0], 10000, "10% MCAR · baseline"),
            (RUNS[6][0], 10000, "20% MCAR"),
        ]),
        ("Missing-pattern sweep — AMD", [
            (RUNS[0][0], 10000, "MCAR · baseline"),
            (RUNS[7][0], 10000, "MAR"),
        ]),
        ("Neighbors sweep — AMD", [
            (RUNS[8][0], 10000, "k=1"),
            (RUNS[0][0], 10000, "k=5 · baseline"),
            (RUNS[9][0], 10000, "k=15"),
        ]),
        ("Intel k=30 observation", [(RUNS[10][0], 10000, "k=30")]),
        ("Intel 50,000-row stress result", [(RUNS[11][0], 50000, "50,000 rows")]),
    ]

    lines = [
        "# Same-data OFAT benchmark report",
        "",
        f"Benchmark source commit: `{SOURCE_COMMIT}`.",
        "",
        "Generated by `benchmarks/analyze_ofat.py` from the twelve preserved "
        "files in `benchmarks/results/ofat-2026-09-22/`.",
        "",
        "Full-precision statistics, input SHA-256 hashes, contributing record "
        "indices, and every matched timing ratio are preserved in "
        "[the machine-readable summary]"
        "(../../benchmarks/results/ofat-2026-09-22-summary.json).",
        "",
        "## Aggregation methodology",
        "",
        "- APIs and dtypes are always analyzed separately.",
        "- Time is `records[].total_seconds`, summarized as median [min–max].",
        "- For `fit_transform`, total time equals `fit_transform_seconds`. "
        "For `fit_then_transform`, it equals `fit_seconds + transform_seconds`.",
        "- Speedup is `median(KNN total_seconds / Faiss total_seconds)` over "
        "matched records, never the ratio of method-level median times.",
        "- Matching stays within one input file/run and includes size, features, "
        "neighbors, target missing rate, pattern, dtype, API, seed, and repeat. "
        "Data fingerprints and shared execution settings must also agree.",
        "- Ordinary cells contain three seeds × three repeats: nine records "
        "per method and nine pairs per speedup. Stress cells contain three "
        "seeds × one repeat: three records and three pairs.",
        "- Min–max describes observed variation across seeds and repeats, "
        "not a confidence interval. Nine records do not mean nine independent "
        "datasets.",
        "- The single baseline is reused by reference in each relevant sweep.",
        "- Phase-memory profiling was disabled for every run.",
        "- All calculations use raw `records`; existing `summaries` are not "
        "used as calculation inputs.",
        "",
        "## Scope and comparability",
        "",
        "Runs 1–10 used AMD EPYC 7763 runners. Runs 11–12 used Intel Xeon "
        "Platinum 8573C runners. Intel observations are presented separately. "
        "CPU-specific library dispatch and cloud-host variation can affect "
        "relative performance; matching a CPU model does not establish "
        "identical physical execution conditions.",
        "",
        "The neighbors sweep also changes `guaranteed_complete_rows` with k "
        "(1, 5, 15, 30). Incomplete inputs and missing masks therefore differ "
        "between k settings. Interpret it as a k sweep under this "
        "data-generation rule, not an identical-input comparison. The raw "
        "notes saying 'the first five rows' are stale for k=1/15/30; structured "
        "parameter and record fields determine the analysis.",
        "",
        "The Intel 50k result is a standalone stress observation. Its "
        "per-API medians summarize three seeds, with no repeated timing "
        "measurements for an individual seed.",
        "",
        "RMSE and MAE summarize reconstruction error against ground truth "
        "on masked training entries. Similar metrics between available and "
        "KNN indicate similar aggregate reconstruction error under these "
        "tested conditions, not identical predictions or algorithmic equivalence.",
        "",
        "Complete and available donor policies have different eligibility "
        "rules. Interpret complete-policy timing alongside donor availability "
        "and reconstruction error.",
        "",
        "## Input provenance",
        "",
    ]
    provenance_rows = []
    for item in result["inputs"]:
        name = Path(item["file"]).name
        metadata = item["metadata"]
        parameters = item["parameters"]
        run_id = metadata["github_run_id"]
        provenance_rows.append([
            name,
            ", ".join(map(str, parameters["sizes"])),
            parameters["features"],
            parameters["n_neighbors"],
            parameters["target_missing_rate"],
            parameters["missing_pattern"],
            parameters["repeats"],
            metadata["cpu_model"],
            f"[{run_id}](https://github.com/ScionKim/FaissImputer/actions/runs/{run_id})",
        ])
    lines.append(table(
        ["File", "Rows", "Features", "k", "Target rate", "Pattern",
         "Repeats", "CPU", "GitHub run"],
        provenance_rows,
    ))
    lines.extend([
        "",
        f"Validation: {result['validation']['records']:,} successful records; "
        f"{result['validation']['timing_cells']} timing cells; "
        f"{result['validation']['speedup_cells']} speedup cells; "
        f"{result['validation']['matched_pairs']:,} matched ratios.",
    ])

    for title, cases in sections:
        lines.extend(["", f"## {title}", ""])
        lines.append(
            "Times are seconds: median [min–max]. Speedups are medians of "
            "matched record-level ratios. N applies to each method and "
            "each speedup pairing."
        )
        lines.append(
            "Donor counts and reconstruction metrics are listed in the "
            "corresponding sections below."
        )
        for api, dtype in itertools.product(APIS, DTYPES):
            lines.extend(["", f"### `{api}` · `{dtype}`", ""])
            rows = []
            for filename, size, label in cases:
                cell = index[(filename, size, api, dtype)]
                methods = cell["methods"]
                rows.append([
                    label,
                    format_stats(methods[KNN]["total_seconds"]),
                    format_stats(methods[COMPLETE]["total_seconds"]),
                    f"{cell['speedups'][COMPLETE]['median']:.4f}×",
                    format_stats(methods[AVAILABLE]["total_seconds"]),
                    f"{cell['speedups'][AVAILABLE]['median']:.4f}×",
                    cell["n"],
                ])
            lines.append(table(
                ["Condition", "KNN time (s)", "Complete time (s)",
                 "Complete speedup", "Available time (s)",
                 "Available speedup", "N"],
                rows,
            ))

    lines.extend([
        "",
        "## Donor availability and realized missingness",
        "",
        "Each row summarizes three unique seed datasets, without counting "
        "methods, APIs, or timing repetitions again. Values are median [min–max]. "
        "`complete_donors` counts fully observed rows; it is not the "
        "feature/query-specific eligible donor count for available mode.",
        "",
    ])
    for dtype in DTYPES:
        lines.extend([f"### `{dtype}`", ""])
        rows = []
        for data in result["datasets"]:
            if data["dtype"] != dtype:
                continue
            rows.append([
                data["file"],
                data["size"],
                data["guaranteed_complete_rows"],
                format_stats(data["complete_donors"], ".0f"),
                format_stats(data["actual_missing_rate"], ".6f", 100),
                format_stats(data["missing_patterns"], ".0f"),
                format_stats(data["scored_cells"], ".0f"),
            ])
        lines.append(table(
            ["File", "Rows", "Guaranteed complete rows",
             "Complete donors", "Actual missing (%)",
             "Missing patterns", "Scored cells"],
            rows,
        ))
        lines.append("")

    lines.extend([
        "## Reconstruction error",
        "",
        "RMSE and MAE are medians [min–max] over the same records used for "
        "timing. Repeated measurements of a seed are not additional independent "
        "quality datasets. Metrics score masked training entries.",
    ])
    for api, dtype in itertools.product(APIS, DTYPES):
        lines.extend(["", f"### `{api}` · `{dtype}`", ""])
        rows = []
        for cell in result["cells"]:
            if (cell["api"], cell["dtype"]) != (api, dtype):
                continue
            for method in METHODS:
                values = cell["methods"][method]
                rows.append([
                    cell["file"],
                    cell["size"],
                    method,
                    format_stats(values["rmse"], ".10g"),
                    format_stats(values["mae"], ".10g"),
                    values["n_records"],
                ])
        lines.append(table(
            ["File", "Rows", "Method", "RMSE", "MAE", "N"], rows
        ))

    lines.extend([
        "",
        "## Available versus KNN metric agreement",
        "",
        "These are absolute differences between matched individual records, "
        "not differences between method-level median errors. APIs and dtypes "
        "remain separate. Each row covers all twelve runs, including the "
        "separately identified Intel observations.",
        "",
    ])
    rows = []
    for item in result["available_metric_agreement"]:
        rmse = item["absolute_rmse_difference"]
        mae = item["absolute_mae_difference"]
        rows.append([
            item["api"],
            item["dtype"],
            item["n_pairs"],
            f"{rmse['median']:.10g}",
            f"{rmse['max']:.10g}",
            f"{mae['median']:.10g}",
            f"{mae['max']:.10g}",
        ])
    lines.append(table(
        ["API", "dtype", "Pairs", "Median |ΔRMSE|", "Max |ΔRMSE|",
         "Median |ΔMAE|", "Max |ΔMAE|"],
        rows,
    ))
    lines.extend([
        "",
        "The summary JSON identifies every pair attaining each maximum, "
        "including input filename, seed, repeat, and 1-based record positions.",
        "",
        "`checks_passed` concerns the benchmark's within-method checks. "
        "It does not establish equality between KNN and available predictions.",
        "",
        "## Memory interpretation",
        "",
        "Full-precision worker peak RSS summaries are included in the "
        "machine-readable output. Peak RSS includes preparation and validation "
        "and is not fitted-model memory. Phase-specific and retained-fit "
        "memory were not measured in these runs.",
        "",
        "## Regeneration",
        "",
        "The analysis script reads the twelve explicitly named input files, "
        "validates provenance and pairing, then generates this Markdown report "
        "and its companion summary JSON. It does not run either imputer, "
        "modify raw input files, or use previously quoted benchmark numbers. "
        "Output ordering is fixed and no generation timestamp is inserted.",
        "",
    ])
    return "\n".join(lines)


def main():
    result = load_and_analyze()
    report = make_report(result)
    summary = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"

    # Complete validation and rendering before writing either output.
    for path, content in (
        (REPORT_PATH, report),
        (SUMMARY_PATH, summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)


if __name__ == "__main__":
    main()