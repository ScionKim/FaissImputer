"""Offline regressions; run in CI with the benchmark dependencies installed."""

from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from benchmarks import diagnose_abalone_output as diagnostic


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["float32", "float64"])
def released_archive(request, tmp_path):
    archive_path = ROOT / "benchmarks/results/released_abalone_0.3.22.zip"
    assert sha256(archive_path.read_bytes()).hexdigest() == diagnostic.RELEASE_ARCHIVE_SHA256
    with ZipFile(archive_path) as archive:
        raw = archive.read(f"version_comparison_abalone_{request.param}.json")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_bytes(raw)
    profile = diagnostic.diagnostic_profile(f"released_{request.param}")
    return profile, diagnostic.read_baseline(baseline_path, profile)


def test_archived_selection_uses_current_release_and_exact_target(released_archive):
    profile, baseline = released_archive
    target = profile["target"]
    assert target == {
        "dataset_id": "abalone", "train_size": 3000, "query_size": 1000,
        "mechanism": "MCAR", "dtype": baseline["parameters"]["dtype"], "seed": 101,
    }
    chosen = diagnostic.select_records(baseline, profile)
    # Both released variants share the method name; method-only selection is ambiguous.
    faiss_rows = [
        row for row in baseline["records"]
        if row["method"] == "FaissImputer[available]" and row["repeat"] == 1
        and all(row.get(key) == value for key, value in target.items())
    ]
    assert {row["variant"] for row in faiss_rows} == {"previous", "current"}
    assert chosen["faiss"][1]["variant"] == "current"
    assert chosen["knn"][1]["variant"] == "knn"
    for index, row in chosen.values():
        assert row is baseline["records"][index]
        assert row["expected_version"] == "0.3.22"
        assert row["input_dtype"] == row["output_dtype"] == target["dtype"]
    assert chosen["knn"][1]["case"] == chosen["faiss"][1]["case"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "missing or duplicated"),
        ("duplicate", "missing or duplicated"),
        ("failed", "Archived worker failed"),
        ("unchecked", "Archived worker failed"),
        ("neighbors", "configuration differs"),
        ("dtype", "configuration differs"),
        ("version", "different package release"),
        ("case", "different prepared cases"),
        ("provenance", "benchmark provenance"),
        ("parameters", "benchmark configuration"),
        ("incomplete", "complete released real-data benchmark"),
    ],
)
def test_changed_archived_evidence_is_rejected(released_archive, mutation, message):
    profile, original = released_archive
    baseline = deepcopy(original)
    index, row = diagnostic.select_records(baseline, profile)["faiss"]
    if mutation == "missing":
        baseline["records"].pop(index)
    elif mutation == "duplicate":
        baseline["records"].append(deepcopy(row))
    elif mutation == "failed":
        row["status"] = "error"
    elif mutation == "unchecked":
        row["checks_passed"] = False
    elif mutation == "neighbors":
        row["n_neighbors"] = 4
    elif mutation == "dtype":
        row["output_dtype"] = "float64" if profile["target"]["dtype"] == "float32" else "float32"
    elif mutation == "version":
        row["environment"]["faiss_imputer"] = "0.3.21"
    elif mutation == "case":
        row["case"] = {"fingerprints": "changed"}
    elif mutation == "provenance":
        baseline["metadata"]["git_commit"] = "unrelated-source"
    elif mutation == "parameters":
        baseline["parameters"]["current_version"] = "0.3.21"
    elif mutation == "incomplete":
        baseline["complete"] = False
    with pytest.raises(ValueError, match=message):
        diagnostic.select_records(baseline, profile)


@pytest.mark.parametrize("case", ["historical", "released_float32", "released_float64"])
def test_checksum_rejection_precedes_json_parsing(case, tmp_path, monkeypatch):
    path = tmp_path / "changed.json"
    path.write_bytes(b'{"records": []}')

    def unexpected_parse(*args, **kwargs):
        pytest.fail("Unverified archive bytes reached the JSON parser")

    monkeypatch.setattr(diagnostic.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="Archived JSON checksum mismatch"):
        diagnostic.read_baseline(path, diagnostic.diagnostic_profile(case))


def test_historical_profile_and_records_remain_supported():
    profile = diagnostic.diagnostic_profile("historical")
    assert profile["published"] is False
    assert profile["target"] == {
        "dataset_id": "abalone", "train_size": 3000, "query_size": 1000,
        "mechanism": "MAR", "dtype": "float64", "seed": 303,
    }
    path = ROOT / "benchmarks/results/real-data-datasets-ef04b1b/abalone.json"
    chosen = diagnostic.select_records(diagnostic.read_baseline(path, profile), profile)
    assert set(chosen) == {"knn", "faiss"}
    assert all(row["repeat"] == 1 for _, row in chosen.values())


def test_cli_without_case_preserves_historical_defaults(tmp_path, monkeypatch):
    captured = {}

    def capture(args, report):
        captured.update(vars(args))
        report["status"] = "ok"

    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "diagnose", capture)
    monkeypatch.setattr(diagnostic, "threadpool_limits", lambda **kwargs: nullcontext())
    monkeypatch.setattr(diagnostic, "config_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(diagnostic.faiss, "omp_set_num_threads", lambda count: None)
    monkeypatch.setattr(diagnostic.sys, "argv", [
        "diagnose_abalone_output", "--expected-version", "historical-version",
        "--provenance", str(tmp_path / "provenance.json"), "--data-home", str(tmp_path),
    ])
    assert diagnostic.main() == 0
    assert captured["case"] == "historical"
    assert captured["baseline"] == tmp_path / "benchmarks/results/real-data-datasets-ef04b1b/abalone.json"
    assert captured["output"] == tmp_path / "benchmark_outputs/abalone_output_diagnostic.json"


@pytest.mark.parametrize("dtype, exponent", [(np.float32, 23), (np.float64, 52)])
def test_exact_distances_preserve_represented_precision_and_shared_count(dtype, exponent):
    adjacent = np.nextafter(dtype(1), dtype(2))
    train = [[Fraction.from_float(float(adjacent)), None, Fraction(2)]]
    query = np.asarray([1, np.nan, 0], dtype=dtype)
    expected = Fraction(6) + Fraction(3, 2) * Fraction(1, 2**exponent) ** 2
    assert diagnostic.exact_distances(train, query) == [expected]


def test_exact_boundary_allows_alternative_ties_but_rejects_farther_donors():
    train = np.asarray([0, 10, 20, 30, 40, 50, 60, np.nan]).reshape(-1, 1)
    distances = list(map(Fraction, [0, 1, 2, 3, 4, 4, 5, 0]))
    reference = diagnostic.reference_cell(train, 0, distances)
    assert reference["exact_boundary_tie_ids"] == [4, 5]
    assert reference["boundary_slots"] == 1
    assert reference["minimum_exact_mean"]["float64"] == 20
    assert reference["maximum_exact_mean"]["float64"] == 22
    alternative = diagnostic.selection_details(train, 0, [0, 1, 2, 3, 5], distances, reference, 22)
    assert alternative["admissible_exact_top_k"] is True
    farther = diagnostic.selection_details(train, 0, [0, 1, 2, 3, 6], distances, reference, 24)
    assert farther["admissible_exact_top_k"] is False
    assert farther["selected_strictly_farther_ids"] == [6]
    omitted = diagnostic.selection_details(train, 0, [1, 2, 3, 4, 5], distances, reference, 30)
    assert omitted["admissible_exact_top_k"] is False
    assert omitted["omitted_strictly_closer_ids"] == [0]


def test_float32_rounding_residual_is_not_rounded_back_to_zero():
    train = np.ones((5, 1), dtype=np.float32)
    train[-1, 0] = np.nextafter(np.float32(1), np.float32(2))
    distances = [Fraction(0)] * 5
    reference = diagnostic.reference_cell(train, 0, distances)
    detail = diagnostic.selection_details(train, 0, range(5), distances, reference, np.float32(1))
    exact_mean = detail["exact_mean"]
    assert Fraction(int(exact_mean["numerator"]), int(exact_mean["denominator"])) == (
        Fraction(1) + Fraction(1, 5 * 2**23)
    )
    assert -2.5e-8 < detail["this_query_output_minus_rounded_exact_mean"] < -2.3e-8


@pytest.fixture
def published_wheel(tmp_path):
    package = tmp_path / "installed" / "faiss_imputer"
    package.mkdir(parents=True)
    contents = {name: f"retained wheel bytes: {name}\n".encode() for name in diagnostic.CORE_SHA256}
    for name, raw in contents.items():
        (package / name).write_bytes(raw)
    wheel = tmp_path / "wheels" / "faiss_imputer-0.3.22-py3-none-any.whl"
    wheel.parent.mkdir()
    provenance = {
        "kind": "published-wheel", "version": "0.3.22",
        "archive_sha256": diagnostic.RELEASE_ARCHIVE_SHA256, "wheel_filename": wheel.name,
    }

    def rebuild(version):
        with ZipFile(wheel, "w") as archive:
            archive.writestr(
                "faiss_imputer-0.3.22.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: faiss-imputer\nVersion: {version}\n\n",
            )
            for name, raw in contents.items():
                archive.writestr(f"faiss_imputer/{name}", raw)
        provenance["wheel_sha256"] = sha256(wheel.read_bytes()).hexdigest()

    rebuild("0.3.22")
    return tmp_path / "provenance.json", provenance, package, wheel, contents, rebuild


def test_published_core_hashes_come_from_retained_wheel(published_wheel):
    path, provenance, package, _, contents, _ = published_wheel
    hashes = diagnostic.verify_published_wheel(path, provenance, package)
    assert hashes == {name: sha256(raw).hexdigest() for name, raw in contents.items()}


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("version", "Downloaded wheel is not faiss-imputer 0.3.22"),
        ("installed_bytes", "Installed core file differs"),
        ("wheel_hash", "Published wheel checksum mismatch"),
    ],
)
def test_changed_published_wheel_evidence_is_rejected(published_wheel, mutation, message):
    path, provenance, package, wheel, _, rebuild = published_wheel
    if mutation == "version":
        rebuild("0.3.21")
    elif mutation == "installed_bytes":
        (package / "_matrix.py").write_bytes(b"different installed bytes")
    elif mutation == "wheel_hash":
        wheel.write_bytes(wheel.read_bytes() + b"changed wheel bytes")
    with pytest.raises(ValueError, match=message):
        diagnostic.verify_published_wheel(path, provenance, package)
