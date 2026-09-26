"""Recompute held-out Wine Quality and Abalone benchmark reports."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
import math
from pathlib import Path
from statistics import median
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "ef04b1b0274d3abdd977440dd61e7d91e328c64d"
STEM = "real-data-datasets-ef04b1b"
METHODS = (
    "SimpleImputer[mean]", "SimpleImputer[median]", "KNNImputer",
    "FaissImputer[complete]", "FaissImputer[available]",
)
MANIFEST = {
    "wine_quality_white": {
        "file": "wine-quality-white.json",
        "sha256": "2c4d748590efa26607cf8f6258de0b20cd58fe0bd90247a07363cc6598aec5f1",
        "run_id": "36066134388", "rows": 4898, "features": 11,
        "driver": "alcohol", "excluded": ["quality"],
    },
    "abalone": {
        "file": "abalone.json",
        "sha256": "4ce80d0aad87ddc9831a2131d752feff3278d563343ef643096c07e7f87fd64c",
        "run_id": "36073113007", "rows": 4177, "features": 7,
        "driver": "Length", "excluded": ["Sex", "Rings"],
    },
}
TIME_FIELDS = ("fit_seconds", "transform_seconds", "total_seconds")
ERROR_FIELDS = (
    "rmse_standardized", "mae_standardized",
    "rmse_original_units", "mae_original_units",
)
ENV_FIELDS = (
    "git_commit", "python", "platform", "cpu_model", "logical_cpus",
    "affinity_cpus", "faiss_imputer", "numpy", "scikit_learn", "faiss",
)
PAIR_FIELDS = (
    "dataset_id", "train_size", "query_size", "mechanism", "dtype",
    "missing_rate", "mar_driver", "mar_reference_rows", "seed", "repeat",
)
CELL_FIELDS = ("dataset_id", "train_size", "mechanism", "dtype", "method")
REPEAT_FIELDS = (
    "case", "model_parameters", "output_sha256", "scored_cells",
    "rmse", "mae", "feature_quality", "agreement_with_knn",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(record, fields):
    return tuple(record[field] for field in fields)


def stats(values):
    values = list(values)
    require(
        values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 for value in values
        ),
        "Expected finite nonnegative measurements",
    )
    return {
        "count": len(values), "median": median(values),
        "min": min(values), "max": max(values), "values": values,
    }


def recorded_distribution(values):
    summary = stats(values)
    return {name: summary[name] for name in ("median", "min", "max")}


def model_parameters(method):
    if method.startswith("SimpleImputer"):
        strategy = "median" if method.endswith("[median]") else "mean"
        return {"copy": True, "strategy": strategy}
    common = {"copy": True, "n_neighbors": 5, "weights": "uniform"}
    if method == "KNNImputer":
        return {**common, "metric": "nan_euclidean"}
    return {
        **common, "metric": "l2", "strategy": "mean",
        "index_factory": "Flat",
        "donor_policy": "complete" if method.endswith("[complete]") else "available",
    }


def validate_quality(record):
    case = record["case"]
    features = record["feature_quality"]
    counts = case["query_mask"]["missing_per_feature"]
    require(len(features) == len(counts) == len(case["feature_names"]),
            "Feature quality length mismatch")
    require(sum(counts) == record["scored_cells"] > 0, "Scoring count mismatch")
    squared, absolute = 0.0, 0.0
    for feature, count, name, scale in zip(
        features, counts, case["feature_names"], case["scaler_scale"]
    ):
        require(feature["feature"] == name and feature["scored_cells"] == count,
                "Feature quality identity mismatch")
        if count == 0:
            require(all(feature[field] is None for field in ERROR_FIELDS),
                    "Unscored feature must have null errors")
            continue
        require(scale > 0 and math.isfinite(scale), "Invalid scaler scale")
        for field in ERROR_FIELDS:
            stats([feature[field]])
        for metric in ("rmse", "mae"):
            require(math.isclose(
                feature[metric + "_original_units"],
                feature[metric + "_standardized"] * scale,
                rel_tol=1e-12, abs_tol=1e-15,
            ), "Source-unit quality mismatch")
        squared += count * feature["rmse_standardized"] ** 2
        absolute += count * feature["mae_standardized"]
    require(math.isclose(record["rmse"], math.sqrt(squared / sum(counts)),
                         rel_tol=1e-12, abs_tol=1e-15), "Aggregate RMSE mismatch")
    require(math.isclose(record["mae"], absolute / sum(counts),
                         rel_tol=1e-12, abs_tol=1e-15), "Aggregate MAE mismatch")


def load_run(directory, dataset_id, spec):
    path = directory / spec["file"]
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == spec["sha256"], f"Changed raw file: {path}")
    run = json.loads(raw)
    require(run["schema_version"] == 2, "Unexpected benchmark schema")
    provenance, dataset = run["provenance"], run["dataset"]
    require(provenance["source_commit"] == run["metadata"]["git_commit"] == SOURCE,
            "Benchmark source commit mismatch")
    require(provenance["github_run_id"] == spec["run_id"]
            and provenance["github_run_attempt"] == "1", "Run identity mismatch")
    require(dataset["rows"] == spec["rows"]
            and dataset["features"] == len(dataset["feature_names"]) == spec["features"]
            and dataset["excluded_columns"] == spec["excluded"]
            and dataset["target_used"] is False, "Dataset selection mismatch")
    parameters = run["parameters"]
    expected = {
        "dataset_id": dataset_id, "mar_driver": spec["driver"],
        "train_sizes": [1000, 3000], "query_size": 1000,
        "mechanisms": ["MCAR", "MAR"], "dtypes": ["float32", "float64"],
        "methods": list(METHODS), "seeds": [101, 202, 303], "repeats": 3,
        "nominal_overall_missing_rate": 0.1, "mar_reference_rows": 1000,
        "threads": 1, "sklearn_working_memory_mib": 256, "expected_workers": 360,
    }
    require(all(parameters[name] == value for name, value in expected.items()),
            "Run configuration mismatch")
    archive_bytes = (directory / "source-data" / f"{dataset_id}.zip").read_bytes()
    require(sha256(archive_bytes).hexdigest() == dataset["source_archive_sha256"],
            "Source archive checksum mismatch")
    with ZipFile(BytesIO(archive_bytes)) as archive:
        source_bytes = archive.read(dataset["source_file"])
    require(sha256(source_bytes).hexdigest() == dataset["source_file_sha256"],
            "Source data file checksum mismatch")

    records = run["records"]
    grid_fields = ("train_size", "mechanism", "dtype", "seed", "repeat", "method")
    expected_grid = Counter(product(
        (1000, 3000), ("MCAR", "MAR"), ("float32", "float64"),
        (101, 202, 303), (1, 2, 3), METHODS,
    ))
    require(Counter(key(row, grid_fields) for row in records) == expected_grid,
            "Missing or duplicated benchmark records")
    case_refs, repeat_refs, query_refs = {}, {}, {}
    for row in records:
        require(row["status"] == "ok" and row["checks_passed"] is True,
                "Unsuccessful worker record")
        require(row["dataset"] == dataset and row["dataset_id"] == dataset_id,
                "Worker dataset mismatch")
        require(row["query_size"] == 1000 and row["missing_rate"] == 0.1
                and row["mar_reference_rows"] == 1000
                and row["mar_driver"] == spec["driver"], "Worker configuration mismatch")
        require(row["model_parameters"] == model_parameters(row["method"]),
                "Unexpected model parameters")
        require(row["input_dtype"] == row["output_dtype"] == row["dtype"],
                "Worker dtype mismatch")
        require(all(row["environment"][field] == run["metadata"][field]
                    for field in ENV_FIELDS), "Worker environment mismatch")
        require(row["threads"] == row["faiss_omp_threads"] == 1
                and row["threadpools"]
                and all(pool["num_threads"] == 1 for pool in row["threadpools"]),
                "Worker thread count mismatch")
        for field in TIME_FIELDS + ("worker_peak_rss_mib", "rmse", "mae"):
            stats([row[field]])
        require(row["total_seconds"] > 0, "Nonpositive total time")
        require(math.isclose(
            row["fit_seconds"] + row["transform_seconds"], row["total_seconds"],
            rel_tol=1e-9, abs_tol=1e-12,
        ), "Inconsistent fit/transform/total timing")
        case = row["case"]
        require(
            case["n_train"] == row["train_size"] and case["n_query"] == 1000
            and case["seed"] == row["seed"] and case["mechanism"] == row["mechanism"]
            and case["input_dtype"] == row["dtype"] and case["truth_dtype"] == "float64"
            and case["always_observed"] == [spec["driver"]]
            and case["feature_names"] == dataset["feature_names"]
            and case["fingerprints"]["dataset"] == dataset["dataset_sha256"]
            and case["mar_reference_rows"] == (1000 if row["mechanism"] == "MAR" else None),
            "Prepared case configuration mismatch",
        )
        case_key = key(row, ("train_size", "mechanism", "dtype", "seed"))
        require(case_refs.setdefault(case_key, case) == case, "Unmatched input cases")
        signature = {field: row[field] for field in REPEAT_FIELDS}
        repeat_key = case_key + (row["method"],)
        require(repeat_refs.setdefault(repeat_key, signature) == signature,
                "Repeated output/quality changed")
        query_key = key(row, ("mechanism", "seed"))
        query_signature = [
            case["fingerprints"][field]
            for field in ("query_row_ids", "raw_query", "query_mask")
        ] + [case["mar_cutoff"], case["mar_reference_rows"]]
        require(query_refs.setdefault(query_key, query_signature) == query_signature,
                "Raw query rows or masks changed across sizes/dtypes")
        agreement = row["agreement_with_knn"]
        require(agreement["scored_cells"] == row["scored_cells"],
                "Agreement scoring count mismatch")
        stats([agreement["max_abs_difference"]])
        if row["method"] == "KNNImputer":
            require(agreement["max_abs_difference"] == 0, "KNN self-comparison mismatch")
        validate_quality(row)

    grouped = defaultdict(list)
    for row in records:
        grouped[key(row, CELL_FIELDS)].append(row)
    stored = {key(row, CELL_FIELDS): row for row in run["summaries"]}
    require(len(run["summaries"]) == len(stored) == len(grouped) == 40,
            "Stored summary count mismatch")
    for cell_key, rows in grouped.items():
        summary = stored[cell_key]
        require(summary["planned_workers"] == summary["recorded_workers"]
                == summary["successful_workers"] == 9
                and summary["pending_workers"] == 0
                and summary["status_counts"] == {"ok": 9}, "Stored summary counts")
        for field in TIME_FIELDS + ("worker_peak_rss_mib", "rmse", "mae"):
            require(summary[field] == recorded_distribution(row[field] for row in rows),
                    f"Stored summary mismatch: {field}")
        require(summary["agreement_with_knn"] == {
            "compared_workers": 9,
            "max_abs_difference": recorded_distribution(
                row["agreement_with_knn"]["max_abs_difference"] for row in rows
            ),
        }, "Stored agreement summary mismatch")
    return run


def analyze(directory):
    datasets, cells, donors = [], [], []
    for dataset_id, spec in MANIFEST.items():
        run = load_run(directory, dataset_id, spec)
        datasets.append({
            "dataset_id": dataset_id, "file": spec["file"], "sha256": spec["sha256"],
            **{field: run[field] for field in ("metadata", "provenance", "dataset", "parameters")},
        })
        indexed = list(enumerate(run["records"]))
        grouped = defaultdict(list)
        references = {}
        for index, row in indexed:
            grouped[key(row, CELL_FIELDS)].append((index, row))
            if row["method"] == "KNNImputer":
                references[key(row, PAIR_FIELDS)] = (index, row)
        for cell_key in sorted(grouped, key=lambda k: (k[1], k[2], k[3], METHODS.index(k[4]))):
            rows = sorted(grouped[cell_key], key=lambda item: (item[1]["seed"], item[1]["repeat"]))
            seeds = [(index, row) for index, row in rows if row["repeat"] == 1]
            require(len(rows) == 9 and len(seeds) == 3, "Unexpected aggregation counts")
            cell = {
                **dict(zip(CELL_FIELDS, cell_key)),
                "record_indices": [index for index, _ in rows],
                "quality_record_indices": [index for index, _ in seeds],
                "timing": {
                    field: stats(row[field] for _, row in rows) for field in TIME_FIELDS
                },
                "worker_peak_rss_mib": stats(row["worker_peak_rss_mib"] for _, row in rows),
                "quality": {
                    field: stats(row[field] for _, row in seeds) for field in ("rmse", "mae")
                },
                "agreement_with_knn": stats(
                    row["agreement_with_knn"]["max_abs_difference"] for _, row in seeds
                ),
                "feature_quality": [],
                "speedup": None,
            }
            for column, name in enumerate(run["dataset"]["feature_names"]):
                features = [row["feature_quality"][column] for _, row in seeds]
                feature = {
                    "feature": name,
                    "scored_cells": [value["scored_cells"] for value in features],
                }
                for field in ERROR_FIELDS:
                    values = [value[field] for value in features]
                    require(all(value is None for value in values)
                            or all(value is not None for value in values),
                            "Inconsistent feature scoring across seeds")
                    feature[field] = None if values[0] is None else stats(values)
                cell["feature_quality"].append(feature)
            if cell_key[-1].startswith("FaissImputer"):
                pairs = []
                for index, row in rows:
                    knn_index, knn = references[key(row, PAIR_FIELDS)]
                    require(knn["case"] == row["case"], "Speedup input mismatch")
                    pairs.append({
                        "seed": row["seed"], "repeat": row["repeat"],
                        "knn_record_index": knn_index, "faiss_record_index": index,
                        "knn_total_seconds": knn["total_seconds"],
                        "faiss_total_seconds": row["total_seconds"],
                        "ratio": knn["total_seconds"] / row["total_seconds"],
                    })
                cell["speedup"] = {**stats(pair["ratio"] for pair in pairs), "pairs": pairs}
            cells.append(cell)
        for train_size, mechanism, seed in product((1000, 3000), ("MCAR", "MAR"), (101, 202, 303)):
            matches = [
                (index, row) for index, row in indexed
                if (row["train_size"], row["mechanism"], row["seed"], row["repeat"], row["method"])
                == (train_size, mechanism, seed, 1, "KNNImputer")
            ]
            require(len(matches) == 2, "Expected two dtype references for donor counts")
            fields = ("train_mask", "query_mask", "complete_donors", "mar_cutoff",
                      "scaler_mean", "scaler_scale")
            first = matches[0][1]["case"]
            require(all(row["case"][field] == first[field]
                        for _, row in matches for field in fields),
                    "Donor/mask metadata differs across dtypes")
            donors.append({
                "dataset_id": dataset_id, "train_size": train_size,
                "mechanism": mechanism, "seed": seed,
                "record_indices": [index for index, _ in matches],
                **{field: first[field] for field in fields},
            })
    require(len(cells) == 80 and len(donors) == 24, "Unexpected total cell counts")
    return {
        "schema_version": 1, "benchmark_source_commit": SOURCE,
        "operation": "held-out fit followed by first transform",
        "aggregation": {
            "timing": "Median [min,max] of 9 records: 3 seeds x 3 repeats.",
            "speedup": "Median of 9 matched KNN/Faiss total_seconds ratios.",
            "pair_fields": list(PAIR_FIELDS),
            "pair_constants": {
                "source_commit": SOURCE, "query_size": 1000, "n_neighbors": 5,
                "weights": "uniform", "index_factory": "Flat",
                "features": {name: spec["features"] for name, spec in MANIFEST.items()},
            },
            "quality": "Median [min,max] of 3 recorded seed metrics; repeats verified equal.",
            "agreement": "Three seed maxima over masked query cells; table displays their maximum.",
            "donors": "One observation per dataset/training-size/mechanism/seed, shared across dtypes.",
            "values": "Unrounded Python float values; JSON preserves round-trip precision.",
            "indices": "Zero-based positions in each dataset's preserved raw JSON records array.",
        },
        "datasets": datasets, "cells": cells, "donors": donors,
    }


def interval(summary, digits):
    return (
        f"{summary['median']:.{digits}f} "
        f"[{summary['min']:.{digits}f}-{summary['max']:.{digits}f}]"
    )


def markdown(result):
    lines = [
        f"# Held-out real-data benchmarks - {SOURCE[:7]}", "",
        f"Benchmark source: `{SOURCE}`.", "",
        f"[Full-precision summary](../../benchmarks/results/{STEM}-summary.json) | "
        "[Analysis script](../../benchmarks/analyze_real_data_datasets.py)", "",
        "## Scope and aggregation", "",
        "These runs measure fitting on 1,000 or 3,000 training rows followed by the "
        "first transform of 1,000 disjoint held-out query rows. They do not measure "
        "same-data fit_transform. Datasets, dtypes, and missingness mechanisms are separate.", "",
        "- Time uses total_seconds, including fit and first transform. Time and process peak RSS "
        "are median [min-max] across nine records (three seeds x three repeats).",
        "- KNN/Faiss is the median of nine matched KNNImputer total_seconds / FaissImputer "
        "total_seconds ratios. Above one favors Faiss; it is not a ratio of median times.",
        "- Matching uses dataset/run, feature count, training/query sizes, mechanism, dtype, "
        "missing rate, MAR driver/reference prefix, neighbors, weights, seed and repeat. "
        "The validated model settings fix five neighbors, uniform weights and Flat Faiss indexes.",
        "- RMSE and MAE summarize reconstruction error against held-out float64 ground truth. "
        "Each quality cell is median [min-max] across three seeds after verifying equal repeat "
        "metrics. Predictions are not archived: this script reaggregates recorded errors.",
        "- Output difference vs KNN is a separate comparison on masked query entries. "
        "Its table value is the maximum across three seed-level maxima, in standardized units. "
        "Similar reconstruction metrics do not establish prediction or algorithmic equivalence.",
        "- Scalers use observed training values only and differ by case. Source-unit feature "
        "errors refer to values as supplied in the data file, not inferred physical units.",
        "- The selected MAR driver remains observed under both MCAR and MAR. MAR uses its "
        "median over a common 1,000-row training prefix. Actual missing rates appear below.",
        "- Peak RSS includes loading, preparation, warmup and validation; timing excludes them. "
        "It is whole-worker peak memory. Min-max is an observed range, not a confidence interval.",
        "- Full-precision JSON includes input hashes, record indices, nine paired ratios, "
        "three-seed quality values, and per-feature standardized/source-unit errors.", "",
    ]
    for item in result["datasets"]:
        dataset_id, dataset = item["dataset_id"], item["dataset"]
        run_id = item["provenance"]["github_run_id"]
        lines += [
            f"## {dataset['dataset']}", "",
            f"Source: [{dataset['citation']}]({dataset['source']}). "
            f"License: {dataset['license']}.",
            f"Features: {dataset['features']}; excluded columns: "
            + ", ".join(dataset["excluded_columns"]) + ". "
            f"Always observed: {item['parameters']['mar_driver']}.",
            f"CPU model: **{item['metadata']['cpu_model']}**. "
            f"[Actions run {run_id}](https://github.com/ScionKim/FaissImputer/actions/runs/{run_id}).",
            f"[Raw JSON](../../benchmarks/results/{STEM}/{item['file']}); "
            f"SHA-256: `{item['sha256']}`.", "",
        ]
        for dtype in ("float32", "float64"):
            selected = [cell for cell in result["cells"]
                        if cell["dataset_id"] == dataset_id and cell["dtype"] == dtype]
            lines += [
                f"### Timing and memory: {dtype}", "",
                "Each time/RSS cell uses nine records; each KNN/Faiss cell uses nine matched pairs.", "",
                "| Train | Pattern | Method | Total seconds, median [min-max] | KNN/Faiss | Peak RSS MiB, median [min-max] |",
                "| ---: | --- | --- | ---: | ---: | ---: |",
            ]
            for cell in selected:
                speed = "-" if cell["speedup"] is None else f"{cell['speedup']['median']:.4f}x"
                lines.append(
                    f"| {cell['train_size']} | {cell['mechanism']} | {cell['method']} | "
                    f"{interval(cell['timing']['total_seconds'], 6)} | {speed} | "
                    f"{interval(cell['worker_peak_rss_mib'], 2)} |"
                )
            lines += [
                "", f"### Reconstruction quality and KNN output comparison: {dtype}", "",
                "RMSE/MAE use three seed observations. The final column is the maximum "
                "masked-cell output difference from KNN over those three seeds.", "",
                "| Train | Pattern | Method | RMSE, median [min-max] | MAE, median [min-max] | Max output difference vs KNN |",
                "| ---: | --- | --- | ---: | ---: | ---: |",
            ]
            for cell in selected:
                lines.append(
                    f"| {cell['train_size']} | {cell['mechanism']} | {cell['method']} | "
                    f"{interval(cell['quality']['rmse'], 6)} | "
                    f"{interval(cell['quality']['mae'], 6)} | "
                    f"{cell['agreement_with_knn']['max']:.6g} |"
                )
            lines.append("")
        lines += [
            "### Donors and actual missingness", "",
            "Each row is one seed dataset, shared by both dtypes and all timing repeats. "
            "Complete mode restricts candidates to complete training rows. Available mode "
            "can use observed entries in incomplete rows; per-feature availability is "
            "training size minus missing_per_feature in the full-precision JSON.", "",
            "| Train | Pattern | Seed | Complete donors | Train missing % | Query missing % |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in result["donors"]:
            if row["dataset_id"] == dataset_id:
                lines.append(
                    f"| {row['train_size']} | {row['mechanism']} | {row['seed']} | "
                    f"{row['complete_donors']} | "
                    f"{100 * row['train_mask']['overall_missing_rate']:.4f} | "
                    f"{100 * row['query_mask']['overall_missing_rate']:.4f} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks/results" / STEM)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/benchmarks" / f"{STEM}.md")
    parser.add_argument("--summary", type=Path, default=ROOT / "benchmarks/results" / f"{STEM}-summary.json")
    args = parser.parse_args()
    result = analyze(args.results_dir)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    report = markdown(result)
    for path, content in ((args.summary, encoded), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Validated 720 records; wrote {args.report} and {args.summary}")


if __name__ == "__main__":
    main()