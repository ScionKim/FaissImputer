"""Focused checks for the float32 correctness diagnostic's evidence rules."""

import itertools
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks import diagnose_available_float32_correctness as diagnostic


class ExactMeanTests(unittest.TestCase):
    def test_severe_cancellation_is_exact_and_order_independent(self):
        for values in itertools.permutations([1e20, 1.0, -1e20]):
            with self.subTest(values=values):
                self.assertEqual(diagnostic.exact_mean(values), Fraction(1, 3))

    def test_mean_uses_binary_float_values(self):
        self.assertEqual(
            diagnostic.exact_mean([0.1, 0.2]),
            (Fraction.from_float(0.1) + Fraction.from_float(0.2)) / 2,
        )

    def test_empty_and_nonfinite_inputs_are_rejected(self):
        for values in ([], [float("nan")], [float("inf")], [float("-inf")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    diagnostic.exact_mean(values)


class DonorEvidenceTests(unittest.TestCase):
    def test_donor_order_is_ignored_but_multiplicity_is_preserved(self):
        self.assertTrue(diagnostic.same_donor_multiset([7, 7, 9], [9, 7, 7]))
        self.assertFalse(diagnostic.same_donor_multiset([7, 7, 9], [7, 9, 9]))

    def test_failed_archive_reproduction_cannot_support_historical_attribution(self):
        for trace_matches, capture_complete, same_donors, faithful_reducer_replay in (
            itertools.product((False, True), repeat=4)
        ):
            evidence = {
                "archive_reproduced": False,
                "trace_matches": trace_matches,
                "capture_complete": capture_complete,
                "same_donors": same_donors,
                "faithful_reducer_replay": faithful_reducer_replay,
            }
            with self.subTest(**evidence):
                self.assertEqual(
                    diagnostic.classify_cell(**evidence), "current_execution_only"
                )

    def test_weaker_evidence_blocks_donor_or_aggregation_attribution(self):
        cases = (
            (False, False, False, False, "trace_output_mismatch"),
            (True, False, False, False, "trace_incomplete"),
            (True, True, False, False, "reducer_replay_mismatch"),
            (True, True, False, True, "different_selected_donors"),
            (True, True, True, True, "same_donors_aggregation_difference"),
        )
        for trace_matches, capture_complete, same_donors, replay, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    diagnostic.classify_cell(
                        archive_reproduced=True,
                        trace_matches=trace_matches,
                        capture_complete=capture_complete,
                        same_donors=same_donors,
                        faithful_reducer_replay=replay,
                    ),
                    expected,
                )


class EvidencePreservationTests(unittest.TestCase):
    def test_existing_worker_evidence_is_neither_reused_nor_overwritten(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostic.json"
            directory = diagnostic.reserve_worker_directory(output)
            log = directory / "seed101.log"
            record = directory / "seed101.json"
            log.write_bytes(b"original log\n")
            record.write_bytes(b'{"status": "error"}\n')
            with self.assertRaises(FileExistsError):
                diagnostic.reserve_worker_directory(output)
            self.assertEqual(log.read_bytes(), b"original log\n")
            self.assertEqual(record.read_bytes(), b'{"status": "error"}\n')
            self.assertFalse(output.exists())

            other = diagnostic.reserve_worker_directory(
                Path(temporary) / "different-diagnostic.json"
            )
            self.assertNotEqual(other, directory)
            self.assertEqual(list(other.iterdir()), [])
            self.assertEqual(log.read_bytes(), b"original log\n")


if __name__ == "__main__":
    unittest.main()
