"""Apply the frozen correctness checks to an explicitly identified source candidate."""

from __future__ import annotations

import argparse
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
CANDIDATE_ID = "available-uniform-compact-v1"
SPEC_SHA256 = "d134b49fc2374b5a99ceb6eeb1203b3fad60ba1c558d18547b79dc9b3e74a0b8"
VERIFIER_SHA256 = "4d8906c273470d9324d47456edca4583e440d3ca923ea811fd910752a74d37f2"
SOURCE_SHA256 = "6e68733947b88bccc7a99e089deaf7207a15b9382f0c8b164cec7b8d3cb36360"
PACKAGE_VERSIONS = {
    "faiss-imputer": "0.3.22", "numpy": "2.5.3", "scikit-learn": "1.9.1",
    "faiss-cpu": "1.15.1", "scipy": "1.18.1", "threadpoolctl": "3.7.0",
    "joblib": "1.6.0",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(manifest):
    expected = {
        "schema_version": 1, "candidate_id": CANDIDATE_ID,
        "implementation_kind": "source_candidate", "baseline_version": "0.3.22",
        "baseline_correctness_status": "failed",
        "baseline_correctness_run_id": "38638261596", "diagnostic_run_id": "38681953970",
        "baseline_source_sha256": "bf82b170a6af3161b49cf623f714b5b80ee849ff91a599f5fc05268f9acbd17a",
        "candidate_source_sha256": SOURCE_SHA256,
        "frozen_spec_sha256": SPEC_SHA256, "verifier_sha256": VERIFIER_SHA256,
        "package_versions": PACKAGE_VERSIONS,
        "criterion": {"rtol": 3e-6, "atol": 0.0, "reference": "KNNImputer"},
        "performance_measurements": False, "automatic_retry": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen candidate field changed: {key}")


def load_context(spec, candidate_path, manifest):
    # Import the installed candidate before adding repository benchmark helpers.
    from faiss_imputer import FaissImputer

    module = importlib.import_module(FaissImputer.__module__)
    source = Path(module.__file__).resolve()
    if source.is_relative_to(ROOT) or file_hash(source) != SOURCE_SHA256:
        raise ValueError("The loaded class must come from the isolated installed candidate")
    if file_hash(ROOT / "faiss_imputer/faiss_imputer.py") != SOURCE_SHA256:
        raise ValueError("The checkout candidate differs from the frozen source")
    for distribution, expected in PACKAGE_VERSIONS.items():
        if version(distribution) != expected:
            raise ValueError(f"Unexpected dependency version: {distribution}")
    verifier_path = ROOT / "benchmarks/verify_benchmark_correctness.py"
    if file_hash(verifier_path) != VERIFIER_SHA256:
        raise ValueError("The original correctness verifier changed")
    sys.path.insert(0, str(ROOT))
    try:
        verifier = importlib.import_module("benchmarks.verify_benchmark_correctness")
    finally:
        sys.path.pop(0)
    if Path(verifier.__file__).resolve() != verifier_path.resolve():
        raise ValueError("Unexpected verifier import origin")
    verifier.validate_spec(spec)
    for relative, expected in verifier.HELPER_HASHES.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Original data helper changed: {relative}")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Python 3.12 is required")
    implementation = {
        "kind": "source_candidate", "candidate_id": CANDIDATE_ID,
        "candidate_manifest_sha256": file_hash(candidate_path),
        "source_sha256": file_hash(source), "loaded_definition_path": str(source),
        "driver_sha256": file_hash(__file__), "verifier_sha256": VERIFIER_SHA256,
        "baseline_version": "0.3.22", "published_baseline_correctness_status": "failed",
        "spec_source_override": "The v3 workload and criterion are unchanged; this run evaluates the labelled source candidate instead of the published wheel implementation.",
        "distribution_metadata_note": manifest["distribution_metadata_note"],
    }
    return verifier, implementation


def worker(verifier, implementation, spec, spec_path, identity, output):
    # This sidecar labels even partial raw verifier records after a timeout.
    verifier.save_json(output.with_name(output.stem + "-implementation.json"), {
        "identity": identity, "implementation": implementation,
    })
    result = verifier.worker(spec, spec_path, identity, output)
    result["implementation"] = implementation
    result["performance_gate_status"] = "not_evaluated"
    verifier.save_json(output, result)
    return result


def orchestrate(verifier, implementation, spec, spec_path, candidate_path, manifest, output):
    directory = output.with_name(output.stem + "-workers")
    directory.mkdir(parents=True, exist_ok=False)
    expected = verifier.worker_schedule(spec)
    result = {
        "status": "running", "implementation": implementation,
        "candidate_manifest": manifest, "reference_specification": spec,
        "spec_sha256": file_hash(spec_path), "verification_plan": verifier.VERIFICATION_PLAN,
        "expected_workers": expected, "worker_directory": directory.name,
        "criterion_id": verifier.CRITERION_ID, "rtol": verifier.RTOL, "atol": verifier.ATOL,
        "published_baseline_correctness_status": "failed",
        "performance_gate_status": "not_evaluated", "records": [],
    }
    verifier.save_json(output, result)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({name: "1" for name in verifier.THREAD_ENV})
    fingerprints = {}
    started = time.monotonic()
    for identity in expected:
        remaining = 1500 - (time.monotonic() - started)
        stem = f"{identity['donor_policy']}-seed{identity['seed']}-repeat{identity['repeat']}"
        worker_output = directory / (stem + ".json")
        if remaining <= 0:
            record = {**identity, "status": "resource_bounded", "evidence_validated": False,
                      "completion_status": "not_executed", "reason": "total_worker_budget_exhausted"}
        else:
            command = [sys.executable, "-I", str(Path(__file__).resolve()),
                       "--spec", str(spec_path), "--candidate", str(candidate_path),
                       "--output", str(worker_output), "--worker",
                       "--policy", identity["donor_policy"], "--seed", str(identity["seed"]),
                       "--repeat", str(identity["repeat"])]
            before = time.monotonic()
            timed_out, returncode = False, None
            with (directory / (stem + ".log")).open("x", encoding="utf-8") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, timeout=min(300, remaining), check=False)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            try:
                record = json.loads(worker_output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {**identity, "status": "process_error", "reason": "missing_worker_output"}
            record["evidence_validated"] = False
            record["worker_wall_seconds"] = time.monotonic() - before
            record["returncode"] = returncode
            if timed_out:
                record.update(status_before_timeout=record["status"], status="resource_bounded",
                              completion_status="not_completed", reason="worker_wall_budget_exceeded")
            elif record.get("status") in ("passed", "failed") and returncode == int(record["status"] != "passed"):
                try:
                    if record.get("implementation") != implementation:
                        raise ValueError("Worker implementation identity differs")
                    sidecar = json.loads(worker_output.with_name(stem + "-implementation.json").read_text(encoding="utf-8"))
                    if sidecar != {"identity": identity, "implementation": implementation}:
                        raise ValueError("Worker implementation sidecar differs")
                    verifier.audit_worker(record, identity, result["spec_sha256"], directory)
                    expected_sources = {**verifier.HELPER_HASHES,
                                        "benchmarks/verify_benchmark_correctness.py": VERIFIER_SHA256}
                    if record.get("source_sha256") != expected_sources:
                        raise ValueError("Worker source provenance differs")
                    for block in record["blocks"]:
                        key = (identity["donor_policy"], identity["seed"], block["dtype"])
                        if fingerprints.setdefault(key, block["fingerprints"]) != block["fingerprints"]:
                            raise ValueError("Prepared inputs differ across methods or repeats")
                    record["evidence_validated"] = True
                except Exception:
                    record.update(status="invalid_evidence", audit_error=traceback.format_exc())
            elif record.get("status") not in ("error", "process_error"):
                record.update(status="process_error", reason="worker_status_exit_mismatch")
        # A partial/error record is still candidate evidence, never release evidence.
        record.setdefault("implementation", implementation)
        result["records"].append(record)
        result["summary"] = verifier.summarize_records(expected, result["records"])
        verifier.save_json(output, result)
        print(f"{stem}: {record['status']}", flush=True)
    result["status"] = result["summary"]["status"]
    result["preflight_wall_seconds"] = time.monotonic() - started
    verifier.save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("complete", "available"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec_path, candidate_path, output = args.spec.resolve(), args.candidate.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output exists; preserve earlier candidate evidence")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if file_hash(spec_path) != SPEC_SHA256:
        parser.error("The frozen v3 specification changed")
    verifier, implementation = load_context(spec, candidate_path, manifest)
    if args.worker:
        identity = {"donor_policy": args.policy, "seed": args.seed, "repeat": args.repeat}
        if identity not in verifier.worker_schedule(spec):
            parser.error("Worker identity outside the frozen schedule")
        result = worker(verifier, implementation, spec, spec_path, identity, output)
    else:
        result = orchestrate(verifier, implementation, spec, spec_path, candidate_path, manifest, output)
    print(f"Candidate correctness: {result['status']}; published baseline: failed; performance gate: not_evaluated", flush=True)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
