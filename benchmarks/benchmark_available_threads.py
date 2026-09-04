"""Compare native thread counts for the 128 MiB available-donor candidate."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks import benchmark_available_batches as batches
from benchmarks import benchmark_scaling_threads as base


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    if args.output is None:
        args.output = base.ROOT / "benchmark_outputs/available_threads.json"
        sys.argv.extend(["--output", str(args.output)])

    old_methods, old_worker = base.METHODS, base.run_worker
    base.METHODS = (
        "KNNImputer",
        "FaissImputer[available,128MiB]",
    )
    base.run_worker = batches.run_worker
    try:
        exit_code = base.main()
    finally:
        base.METHODS, base.run_worker = old_methods, old_worker

    results = json.loads(args.output.read_text(encoding="utf-8"))
    results["parameters"]["experiment_batch_budgets_mib"] = [128]
    results["notes"] = [
        note for note in results["notes"]
        if not note.startswith(("Complete policy", "One seed and one repeat"))
    ]
    results["notes"].extend([
        "Only KNNImputer and the 128 MiB available candidate are benchmarked.",
        "The original 16 MiB and experimental 64 MiB variants are not rerun.",
        "The experimental subclass changes only the batch budget in memory.",
        "Product files and the released implementation remain unchanged.",
        "128 MiB is a batch-sizing budget, NOT a total process memory cap.",
        "FAISS/OpenMP and BLAS thread limits change together; this is not "
        "a FAISS-only threading experiment.",
        "Output differences from KNN, one thread, and the first repeat "
        "are recorded as diagnostics.",
        "Repeated timings on one synthetic seed do not establish general performance.",
    ])
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Thread comparison saved: {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
