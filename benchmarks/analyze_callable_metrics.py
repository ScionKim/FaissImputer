"""Analyze the preserved same-data callable-metric benchmark."""

import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
STEM = "fit-transform-callable-76e0230"
SOURCE = "76e02301aed7202656680ff119b8d0f726e67ef4"
RAW_SHA256 = "7672a71977691b5f879a9bf22c92a438a79117e6dd3e9f83ed686cca5042eb47"
RAW = ROOT / "benchmarks/results" / f"{STEM}.json"
REPORT = ROOT / "docs/benchmarks" / f"{STEM}.md"
SUMMARY = ROOT / "benchmarks/results" / f"{STEM}-summary.json"

SIZES = (300, 1000)
DTYPES = ("float32", "float64")
APIS = ("fit_transform", "fit_then_transform")
METRICS = ("builtin", "callable_nan_euclidean")
METHODS = (
    "KNNImputer",
    "FaissImputer[complete]",
    "FaissImputer[available]",
)
SEEDS = (101, 202, 303)
CELL_FIELDS = ("size", "dtype", "api", "metric", "method")
MATCH_FIELDS = (
    "size", "features", "n_neighbors", "target_missing_rate",
    "missing_pattern", "dtype", "api", "seed", "repeat",
)
PHASE_FIELDS = (
    "rss_before_fit_mib", "rss_after_fit_mib", "fit_retained_rss_mib",
    "fit_phase_peak_rss_mib", "transform_phase_peak_rss_mib",
)
MEASURES = (
    "total_seconds", "fit_seconds", "transform_seconds",
    "fit_transform_seconds", "worker_peak_rss_mib", "rmse", "mae",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = list(values)
    return {
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def key(record, fields):
    return tuple(record[field] for field in fields)


def analyze(raw):
    require(hashlib.sha256(raw).hexdigest() == RAW_SHA256,
            "Raw JSON SHA-256 differs from the preserved benchmark")
    data = json.loads(raw)
    records = data["records"]
    params = data["parameters"]
    metadata = data["metadata"]
    provenance = data["provenance"]
    require(data["schema_version"] == 2, "Unexpected input schema")
    require(metadata["git_commit"] == provenance["source_commit"] == SOURCE,
            "Benchmark source commit differs")
    require(metadata["faiss_imputer"] == provenance["version"],
            "Package provenance differs")
    for field in ("github_run_id", "github_run_attempt"):
        require(metadata[field] == provenance[field], f"Provenance: {field}")

    expected = {
        "sizes": list(SIZES), "dtypes": list(DTYPES),
        "apis": list(APIS), "metrics": list(METRICS),
        "methods": list(METHODS), "seeds": list(SEEDS),
        "repeats": 3, "features": 20, "n_neighbors": 5,
        "target_missing_rate": 0.10, "missing_pattern": "mcar",
        "guaranteed_complete_rows": 5, "threads": 1,
        "phase_memory_measured": False, "expected_workers": 432,
    }
    for field, value in expected.items():
        require(params[field] == value, f"Unexpected parameter: {field}")

    grid_fields = CELL_FIELDS + ("seed", "repeat")
    grid = set(itertools.product(
        SIZES, DTYPES, APIS, METRICS, METHODS, SEEDS, (1, 2, 3)
    ))
    actual = [key(record, grid_fields) for record in records]
    require(len(actual) == len(set(actual)) == 432 and set(actual) == grid,
            "Missing, duplicate, or unexpected records")

    groups = defaultdict(list)
    inputs, outputs, donors = {}, {}, {}
    for index, record in enumerate(records):
        require(record["status"] == "ok" and record["checks_passed"] is True,
                f"Failed record: {index}")
        require(record["input_dtype"] == record["output_dtype"] == record["dtype"],
                f"Dtype differs: {index}")
        for field in (
            "features", "n_neighbors", "target_missing_rate", "missing_pattern",
            "guaranteed_complete_rows", "threads", "sklearn_working_memory_mib",
        ):
            require(record[field] == params[field], f"Configuration: {field}")
        for field, value in metadata.items():
            if field != "created_at_utc":
                require(record["environment"][field] == value,
                        f"Environment differs: {field}")
        require(record["expected_version"] == provenance["version"],
                "Worker package version differs")
        require(record["faiss_omp_threads"] == 1 and all(
            pool["num_threads"] == 1 for pool in record["threadpools"]
        ), "Thread limits differ")
        require(record["measure_phase_memory"] is False and all(
            record[field] is None for field in PHASE_FIELDS
        ), "Phase-memory measurements are not canonical timing results")
        for field in MEASURES:
            value = record[field]
            require(value is None or (math.isfinite(value) and value >= 0),
                    f"Invalid measurement: {index}/{field}")
        require(record["total_seconds"] > 0, "Nonpositive total time")
        if record["api"] == "fit_transform":
            require(record["fit_seconds"] is None and
                    record["transform_seconds"] is None and
                    record["fit_transform_seconds"] == record["total_seconds"],
                    "Invalid fit_transform timing")
        else:
            require(record["fit_transform_seconds"] is None and math.isclose(
                record["total_seconds"],
                record["fit_seconds"] + record["transform_seconds"],
                rel_tol=1e-12, abs_tol=1e-12,
            ), "Invalid fit_then_transform timing")
        require(record["quality_scope"] == "masked training entries",
                "Unexpected reconstruction scope")

        case = key(record, ("size", "dtype", "seed"))
        require(inputs.setdefault(case, record["fingerprints"]) ==
                record["fingerprints"], "Inputs differ across comparisons")
        output_key = case + (record["method"], record["metric"])
        output = key(record, ("output_sha256", "rmse", "mae"))
        require(outputs.setdefault(output_key, output) == output and
                record["max_abs_difference_from_first_success"] == 0,
                "Outputs differ across APIs or repeats")

        donor_key = (record["size"], record["seed"])
        donor = {
            field: record[field] for field in (
                "complete_donors", "actual_missing_rate",
                "scored_cells", "missing_patterns",
            )
        }
        require(donors.setdefault(donor_key, donor) == donor,
                "Dataset counts differ across methods, metrics, APIs or dtypes")
        groups[key(record, CELL_FIELDS)].append(index)

    stored = {key(row, CELL_FIELDS): row for row in data["summaries"]}
    require(len(stored) == len(data["summaries"]) == len(groups) == 48,
            "Unexpected summary groups")
    require(set(stored) == set(groups), "Summary keys differ")
    for group, indices in groups.items():
        summary = stored[group]
        require(len(indices) == 9 and
                summary["planned_workers"] == 9 and
                summary["recorded_workers"] == 9 and
                summary["successful_workers"] == 9 and
                summary["pending_workers"] == 0 and
                summary["status_counts"] == {"ok": 9},
                "Incomplete summary")
        for field in MEASURES + PHASE_FIELDS:
            values = [records[i][field] for i in indices
                      if records[i][field] is not None]
            require(summary.get(field) == (stats(values) if values else None),
                    f"Stored summary differs: {group}/{field}")

    def ratios(numerators, denominators, extra_field):
        fields = MATCH_FIELDS + (extra_field,)
        left = {key(records[i], fields): i for i in numerators}
        right = {key(records[i], fields): i for i in denominators}
        require(len(left) == len(right) == 9 and set(left) == set(right),
                "Records cannot be matched one-to-one")
        pairs = []
        for pair_key in sorted(left):
            a, b = left[pair_key], right[pair_key]
            require(records[a]["fingerprints"] == records[b]["fingerprints"],
                    "Paired inputs differ")
            pairs.append({
                "numerator_record_index": a,
                "denominator_record_index": b,
                "seed": records[a]["seed"],
                "repeat": records[a]["repeat"],
                "numerator_seconds": records[a]["total_seconds"],
                "denominator_seconds": records[b]["total_seconds"],
                "ratio": records[a]["total_seconds"] / records[b]["total_seconds"],
            })
        return {"n_pairs": 9, **stats(p["ratio"] for p in pairs), "pairs": pairs}

    cells = []
    for group, indices in sorted(groups.items()):
        size, dtype, api, metric, method = group
        observations = []
        for seed in SEEDS:
            seed_indices = [i for i in indices if records[i]["seed"] == seed]
            first = records[seed_indices[0]]
            observations.append({
                "seed": seed, "record_indices": seed_indices,
                "rmse": first["rmse"], "mae": first["mae"],
            })
        cell = {
            **dict(zip(CELL_FIELDS, group)),
            "n_records": 9, "record_indices": indices,
            "total_seconds": stats(records[i]["total_seconds"] for i in indices),
            "worker_peak_rss_mib": stats(
                records[i]["worker_peak_rss_mib"] for i in indices
            ),
            "quality": {
                "n_seeds": 3, "observations": observations,
                "rmse": stats(o["rmse"] for o in observations),
                "mae": stats(o["mae"] for o in observations),
            },
        }
        if method != "KNNImputer":
            reference = groups[(size, dtype, api, metric, "KNNImputer")]
            cell["knn_over_faiss"] = ratios(reference, indices, "metric")
        if metric == "callable_nan_euclidean":
            reference = groups[(size, dtype, api, "builtin", method)]
            cell["callable_over_builtin"] = ratios(indices, reference, "method")
        cells.append(cell)

    return {
        "schema_version": 1,
        "raw_file": RAW.relative_to(ROOT).as_posix(),
        "raw_sha256": RAW_SHA256,
        "benchmark_source_commit": SOURCE,
        "metadata": metadata, "provenance": provenance, "parameters": params,
        "definitions": {
            "record_indices": "Zero-based indices into the raw records array.",
            "timing": "Median/min/max of total_seconds across 3 seeds x 3 repeats.",
            "knn_over_faiss": "Median of nine matched KNN/Faiss total_seconds ratios.",
            "callable_over_builtin": "Median of nine same-method matched callable/builtin ratios.",
            "matching": list(MATCH_FIELDS),
            "matching_extra": "metric for KNN/Faiss; method for callable/builtin; one run only.",
            "quality": "Median/min/max over 3 seeds after verifying API/repeat copies agree.",
            "donors": "One observation per size/seed; counts verified equal across dtypes.",
        },
        "validation": {"records": 432, "timing_cells": 48, "records_per_cell": 9},
        "cells": cells,
        "datasets": [
            {"size": size, "seed": seed, **donors[(size, seed)]}
            for size in SIZES for seed in SEEDS
        ],
    }


def render(summary):
    def span(values, digits=6):
        return (
            f"{values['median']:.{digits}f} "
            f"[{values['min']:.{digits}f}–{values['max']:.{digits}f}]"
        )

    cells = {key(cell, CELL_FIELDS): cell for cell in summary["cells"]}
    lines = [
        "# Same-data callable metric benchmark — 76e0230",
        "",
        f"Benchmark source: `{SOURCE}`.",
        f"Runner CPU: **{summary['metadata']['cpu_model']}**.",
        f"GitHub Actions run: [{summary['metadata']['github_run_id']}]"
        f"(https://github.com/ScionKim/FaissImputer/actions/runs/"
        f"{summary['metadata']['github_run_id']}).",
        "",
        f"[Raw JSON](../../benchmarks/results/{STEM}.json) · "
        f"[Full-precision summary](../../benchmarks/results/{STEM}-summary.json) · "
        "[Analysis script](../../benchmarks/analyze_callable_metrics.py)",
        "",
        f"Raw-file SHA-256: `{RAW_SHA256}`.",
        "",
        "## Scope and aggregation",
        "",
        "This source-build study measures 300 and 1,000 training rows, "
        "20 features, 5 neighbors, 10% target MCAR missingness, uniform "
        "weights, and one native thread. Phase-memory sampling is disabled.",
        "",
        "The 432 records cover three methods, two metrics, two APIs, two "
        "dtypes, three seeds, and three repeats at each size. All records "
        "passed validation. This report uses only this preserved run.",
        "",
        "- APIs and dtypes are reported separately. Both APIs include fitting.",
        "- Time and process peak RSS are median [min–max] over nine records "
        "(three seeds × three repeats) per size/method/metric/API/dtype.",
        "- KNN/Faiss is the median of nine matched total_seconds ratios. "
        "Values above one favor FaissImputer. Times are not divided after aggregation.",
        "- Pairing uses the same run, size, features, neighbors, target missing "
        "rate, missingness pattern, dtype, API, metric, seed, and repeat.",
        "- Callable/builtin is a separate median of nine paired time ratios "
        "within the same method; values above one mean the callback takes longer.",
        "- Reconstruction scores and donor counts summarize three seed "
        "datasets. Repeats and APIs are not independent accuracy observations.",
        "- Min–max gives the observed range, not a confidence interval. "
        "The summary retains unrounded floats and zero-based raw-record indices.",
        "",
        "## Metric paths",
        "",
        "Builtin mode retains KNNImputer's nan_euclidean metric and "
        "FaissImputer's l2 metric. Callable mode supplies the same Python "
        "function to all methods. It returns the square root of the shared "
        "squared differences multiplied by original feature count / shared "
        "feature count, using float64 arithmetic for either input dtype. "
        "It returns NaN when no feature is shared.",
        "",
        "Callable FaissImputer evaluates distances directly without a Faiss "
        "index. Complete mode restricts candidates to fully observed donors. "
        "Its timing advantage therefore accompanies a different donor pool; "
        "it is not evidence of Faiss index acceleration for callbacks.",
        "",
    ]
    for api in APIS:
        for dtype in DTYPES:
            lines += [
                f"## Timing: {api} / {dtype}", "",
                "Each time and RSS cell uses nine records. Each KNN/Faiss "
                "cell uses nine matched pairs.", "",
                "| Metric | Rows | Method | Seconds, median [min–max] | "
                "KNN/Faiss | Peak RSS MiB, median [min–max] |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
            for metric, size, method in itertools.product(METRICS, SIZES, METHODS):
                cell = cells[(size, dtype, api, metric, method)]
                ratio = cell.get("knn_over_faiss")
                speed = f"{ratio['median']:.4f}×" if ratio else "—"
                label = "builtin" if metric == "builtin" else "callable"
                lines.append(
                    f"| {label} | {size} | {method} | {span(cell['total_seconds'])} "
                    f"| {speed} | {span(cell['worker_peak_rss_mib'], 2)} |"
                )
            lines += [""]

    lines += [
        "## Callable cost relative to builtin", "",
        "Every value is the median of nine same-method matched "
        "callable/builtin total_seconds ratios. Larger values mean more time.", "",
        "| Rows | Method | fit_transform f32 | fit_transform f64 | "
        "fit_then_transform f32 | fit_then_transform f64 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for size, method in itertools.product(SIZES, METHODS):
        values = [
            f"{cells[(size, dtype, api, 'callable_nan_euclidean', method)]['callable_over_builtin']['median']:.2f}×"
            for api in APIS for dtype in DTYPES
        ]
        lines.append(f"| {size} | {method} | " + " | ".join(values) + " |")

    lines += [
        "", "## Reconstruction quality", "",
        "RMSE and MAE measure reconstruction error against hidden ground "
        "truth at masked training entries. Each cell below is median "
        "[min–max] over three seeds. All API/repeat copies were checked "
        "for agreement before retaining one observation per seed.",
        "",
        "Similar aggregate errors do not establish equality of individual "
        "imputed values or algorithmic equivalence. These measurements "
        "concern this one callback and these generated datasets.",
        "",
    ]
    for dtype in DTYPES:
        lines += [
            f"### {dtype}", "",
            "| Metric | Rows | Method | RMSE, median [min–max] | "
            "MAE, median [min–max] |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        for metric, size, method in itertools.product(METRICS, SIZES, METHODS):
            quality = cells[(size, dtype, "fit_transform", metric, method)]["quality"]
            label = "builtin" if metric == "builtin" else "callable"
            lines.append(
                f"| {label} | {size} | {method} | "
                f"{span(quality['rmse'], 10)} | {span(quality['mae'], 10)} |"
            )
        lines += [""]

    lines += [
        "## Donor counts", "",
        "These counts agree across dtypes, metrics, methods, APIs, and "
        "repeats. Complete donors are fully observed training rows; this "
        "is not the available policy's query/feature-specific donor count. "
        "The first five training rows were kept complete.", "",
        "| Rows | Seed | Complete donors | Masked cells | Actual missing rate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in summary["datasets"]:
        lines.append(
            f"| {dataset['size']} | {dataset['seed']} | "
            f"{dataset['complete_donors']} | {dataset['scored_cells']} | "
            f"{dataset['actual_missing_rate']:.6f} |"
        )
    lines += [
        "", "## Memory and reproduction", "",
        "Worker peak RSS includes preparation and validation before JSON "
        "serialization. It is not fitted-model memory; phase-memory fields "
        "are null.", "",
        "Run the **Analyze callable metric benchmark results** workflow to "
        "regenerate this report and the full-precision JSON summary. The "
        "script verifies the raw-file hash, provenance, full record grid, "
        "shared inputs, API/repeat consistency, and stored summaries. It "
        "does not execute imputers or modify the raw JSON.", "",
    ]
    return "\n".join(lines)


def main():
    summary = analyze(RAW.read_bytes())
    report = render(summary)
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    SUMMARY.write_text(encoded, encoding="utf-8")
    print(f"Validated {summary['validation']['records']} records")
    print(f"Report: {REPORT.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()