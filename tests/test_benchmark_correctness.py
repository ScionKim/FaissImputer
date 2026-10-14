"""Behavioral tests for benchmark correctness evidence and aggregation."""

import numpy as np
import pytest

from benchmarks import verify_benchmark_correctness as verifier


def test_comparison_scores_only_missing_cells_and_accepts_exact_boundary():
    reference = np.array([[0.0, 1.0], [2.0, 0.0]])
    actual = np.array([[999.0, 1.125], [2.25, -999.0]])
    missing = np.array([[False, True], [True, False]])

    result = verifier.compare_outputs(actual, reference, missing, rtol=0.125)

    assert result["status"] == "passed"
    assert result["compared_cells"] == 2
    assert result["violation_count"] == 0
    assert result["max_abs_difference"] == 0.25
    assert result["zero_reference_violation_count"] == 0
    assert result["worst_violation"] is None


def test_comparison_rejects_first_float_beyond_boundary_with_cell_evidence():
    reference = np.array([[8.0]])
    actual = np.array([[np.nextafter(9.0, np.inf)]])

    result = verifier.compare_outputs(
        actual, reference, np.array([[True]]), rtol=0.125
    )

    assert result["status"] == "failed"
    assert result["violation_count"] == 1
    assert result["max_abs_difference"] == actual[0, 0] - 8.0
    assert result["worst_violation"] == {
        "index": [0, 0],
        "actual": actual[0, 0],
        "reference": 8.0,
        "allowed_error": 1.0,
        "absolute_error": actual[0, 0] - 8.0,
    }


def test_zero_reference_requires_exact_zero_but_accepts_signed_zero():
    reference = np.array([[0.0, -0.0]])
    missing = np.ones((1, 2), dtype=bool)

    exact = verifier.compare_outputs(np.array([[-0.0, 0.0]]), reference, missing)
    near = verifier.compare_outputs(np.array([[0.0, 1e-20]]), reference, missing)

    assert exact["status"] == "passed"
    assert exact["zero_reference_violation_count"] == 0
    assert near["status"] == "failed"
    assert near["violation_count"] == 1
    assert near["zero_reference_violation_count"] == 1
    assert near["worst_violation"]["index"] == [0, 1]
    assert near["worst_violation"]["allowed_error"] == 0.0


def test_relative_tolerance_uses_reference_operand():
    missing = np.array([[True]])
    larger = np.array([[2.0]])
    smaller = np.array([[1.0]])

    forward = verifier.compare_outputs(larger, smaller, missing, rtol=0.5)
    reverse = verifier.compare_outputs(smaller, larger, missing, rtol=0.5)

    assert forward["status"] == "failed"
    assert forward["worst_violation"]["allowed_error"] == 0.5
    assert reverse["status"] == "passed"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_repeat_value_hash_accepts_signed_zero_but_detects_changed_values(dtype):
    first = np.array([[0.0, -0.0, 1.0]], dtype=dtype)
    repeated = np.array([[-0.0, 0.0, 1.0]], dtype=dtype)
    before = repeated.copy()

    assert verifier.array_hash(first) != verifier.array_hash(repeated)
    assert verifier.output_value_hash(first) == verifier.output_value_hash(repeated)
    assert verifier.array_hash(repeated) == verifier.array_hash(before)

    repeated[0, 2] = 2.0
    assert verifier.output_value_hash(first) != verifier.output_value_hash(repeated)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_comparison_fails_without_numeric_violation_evidence(invalid_value):
    result = verifier.compare_outputs(
        np.array([[invalid_value]]), np.array([[1.0]]), np.array([[True]])
    )

    assert result["status"] == "failed"
    assert result["reason"]
    assert result["violation_count"] is None
    assert result["max_abs_difference"] is None


def test_shape_mismatch_fails_instead_of_broadcasting():
    result = verifier.compare_outputs(
        np.array([[1.0]]), np.array([[1.0, 1.0]]), np.array([[True, True]])
    )

    assert result["status"] == "failed"
    assert result["reason"]
    assert result["violation_count"] is None


def _structural_inputs():
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    query = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    before = query.copy()
    missing = np.isnan(query)
    output = np.array([[1.0, 2.5], [2.5, 4.0]], dtype=np.float32)
    return output, query, before, missing, train, verifier.array_hash(train)


def test_structural_checks_accept_unchanged_inputs_containing_missing_values():
    checks = verifier.structural_checks(*_structural_inputs())

    for key in (
        "shape_preserved",
        "dtype_preserved",
        "finite",
        "observed_values_preserved",
        "query_unchanged",
        "training_unchanged",
        "separate_output",
    ):
        assert checks[key], key


@pytest.mark.parametrize(
    ("changed_array", "failed_check"),
    [
        (0, "observed_values_preserved"),
        (1, "query_unchanged"),
        (4, "training_unchanged"),
    ],
)
def test_structural_checks_detect_mutations(changed_array, failed_check):
    inputs = _structural_inputs()
    inputs[changed_array][0, 0] = 99.0

    checks = verifier.structural_checks(*inputs)

    assert not checks[failed_check]


def test_repeated_call_detects_mutated_retained_first_output():
    first, query, before, missing, train, train_hash = _structural_inputs()
    first_copy = first.copy()
    repeated = first_copy.copy()
    first[0, 1] = 999.0

    checks = verifier.structural_checks(
        repeated,
        query,
        before,
        missing,
        train,
        train_hash,
        first_copy=first_copy,
        retained_first=first,
    )

    assert checks["identical_to_call_1"]
    assert not checks["retained_first_output_unchanged"]


def _summary_inputs():
    expected = [
        {"donor_policy": "complete", "seed": 101, "repeat": 1},
        {"donor_policy": "available", "seed": 101, "repeat": 1},
    ]
    records = [
        {**identity, "status": "passed", "evidence_validated": True}
        for identity in expected
    ]
    return expected, records


def test_summary_passes_only_complete_validated_worker_set():
    expected, records = _summary_inputs()

    summary = verifier.summarize_records(expected, records)

    assert summary["status"] == "passed"
    assert summary["expected_workers"] == 2
    assert summary["received_workers"] == 2
    assert summary["passed_workers"] == 2
    assert summary["failed_workers"] == 0
    assert summary["unevaluated_workers"] == 0


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("missing", "not_evaluated"),
        ("running", "not_evaluated"),
        ("duplicate", "failed"),
        ("failed", "failed"),
    ],
)
def test_incomplete_or_invalid_worker_sets_cannot_pass(case, expected_status):
    expected, records = _summary_inputs()
    if case == "missing":
        records.pop()
    elif case == "duplicate":
        records.append(records[0].copy())
    else:
        records[1]["status"] = case

    summary = verifier.summarize_records(expected, records)

    assert summary["status"] == expected_status


def test_passed_label_without_validated_evidence_cannot_pass_summary():
    expected, records = _summary_inputs()
    records[1]["evidence_validated"] = False

    summary = verifier.summarize_records(expected, records)

    assert summary["status"] != "passed"
