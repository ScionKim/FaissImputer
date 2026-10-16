"""Small regression checks for the diagnostic; execute on GitHub only.

These tests use callback fakes and tiny arrays. They never fit an imputer or
launch a diagnostic worker.
"""

from copy import deepcopy
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from benchmarks import diagnose_ordered_transform_calls as diagnostic


SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/specs/validation_0.3.22_v1.json"
)


def load_spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


class TestFrozenProtocol(unittest.TestCase):
    def test_changed_measured_experiment_is_rejected(self):
        spec = load_spec()
        diagnostic.validate_spec(spec)
        changes = (
            ("train_rows", 500000),
            ("ordered_dtypes", ["float64", "float32"]),
            ("calls_per_dtype", 5),
            ("seeds", [101, 101, 303]),
            ("repeats_per_seed", 2),
            ("threads", 2),
            ("weights", "distance"),
            ("outer_parallelism", 2),
        )
        for name, value in changes:
            with self.subTest(field=name):
                changed = deepcopy(spec)
                changed["diagnostic"][name] = value
                with self.assertRaises(ValueError):
                    diagnostic.validate_spec(changed)

    def test_warmup_cannot_silently_change_process_history(self):
        for name, value in (
            ("train_rows", 20000),
            ("seed", 101),
            ("per_dtype", False),
            ("separate_model", False),
            ("enabled", False),
        ):
            with self.subTest(field=name):
                spec = load_spec()
                spec["diagnostic"]["warmup"][name] = value
                with self.assertRaises(ValueError):
                    diagnostic.validate_spec(spec)

    def test_altered_isolation_or_reference_threshold_is_rejected(self):
        changes = (
            (("diagnostic", "process_isolation", "worker_count"), 18),
            (("diagnostic", "process_isolation", "dtype_blocks_same_worker"), False),
            (("diagnostic", "process_isolation", "alternate_order"), True),
            (("diagnostic", "process_isolation", "automatic_retry"), True),
            (("diagnostic", "missingness", "train_fraction"), 0.1),
            (("diagnostic", "missingness", "query_missing_features_per_row"), 3),
            (("diagnostic", "missingness", "query_fraction"), 0.5),
            (("diagnostic", "missingness", "query_selection"), "shared_columns"),
            (("diagnostic", "reference", "tolerance"), {"rtol": 0.01, "atol": 0.01}),
            (("package", "version"), "0.3.23"),
            (("revision",), "unreviewed_revision"),
        )
        for path, value in changes:
            with self.subTest(field=".".join(path)):
                spec = load_spec()
                parent = spec
                for name in path[:-1]:
                    parent = parent[name]
                parent[path[-1]] = value
                with self.assertRaises(ValueError):
                    diagnostic.validate_spec(spec)

    def test_worker_schedule_contains_each_seed_repeat_once(self):
        spec = load_spec()
        schedule = diagnostic.worker_schedule(spec)
        identities = [(item["seed"], item["repeat"]) for item in schedule]
        self.assertEqual(
            identities,
            [(seed, repeat) for seed in (101, 202, 303) for repeat in (1, 2, 3)],
        )
        self.assertEqual(len(set(identities)), len(identities))
        # Both dtypes belong to one worker; splitting them into different
        # workers would remove the process state being investigated.
        self.assertTrue(all(set(item) == {"seed", "repeat"} for item in schedule))

    def test_both_timed_blocks_finish_before_reference_or_trace(self):
        events = []
        snapshots = []

        def callback(phase):
            def run(dtype):
                events.append((phase, dtype))
                return {"dtype": dtype, "phase": phase}
            return run

        result = diagnostic.run_protocol(
            load_spec()["diagnostic"],
            callback("timed"), callback("reference"), callback("trace"),
            snapshots.append,
        )
        self.assertEqual(events, [
            ("timed", "float32"), ("timed", "float64"),
            ("reference", "float32"), ("reference", "float64"),
            ("trace", "float32"), ("trace", "float64"),
        ])
        self.assertEqual(snapshots[0], {"blocks": [], "references": [], "traces": []})
        self.assertEqual(len(snapshots), 7)
        self.assertEqual([len(snapshot["blocks"]) for snapshot in snapshots], [0, 1, 2, 2, 2, 2, 2])
        self.assertEqual(snapshots[-1], result)
        result["blocks"][0]["dtype"] = "mutated_after_checkpoint"
        self.assertEqual(snapshots[1]["blocks"][0]["dtype"], "float32")
        self.assertEqual(snapshots[-1]["blocks"][0]["dtype"], "float32")

    def test_reference_failure_preserves_both_completed_timed_blocks(self):
        events = []
        snapshots = []

        def timed(dtype):
            events.append(("timed", dtype))
            return {"dtype": dtype, "status": "ok"}

        def reference(dtype):
            events.append(("reference", dtype))
            raise RuntimeError("reference failed after primary measurements")

        def trace(dtype):
            self.fail("Tracing must not run after reference raises")

        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            diagnostic.run_protocol(
                load_spec()["diagnostic"], timed, reference, trace, snapshots.append,
            )
        self.assertEqual(events, [
            ("timed", "float32"), ("timed", "float64"), ("reference", "float32"),
        ])
        self.assertEqual(
            [block["dtype"] for block in snapshots[-1]["blocks"]],
            ["float32", "float64"],
        )
        self.assertEqual(snapshots[-1]["references"], [])
        self.assertEqual(snapshots[-1]["traces"], [])


class TestOutputChecks(unittest.TestCase):
    def setUp(self):
        self.before = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        self.query = self.before.copy()
        self.missing = np.isnan(self.before)
        self.train = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
        self.train_hash = diagnostic.array_hash(self.train)
        self.output = np.array([[1.0, 3.0], [5.0, 4.0]], dtype=np.float32)

    def checks(self, output=None, first=None):
        return diagnostic.output_checks(
            self.output if output is None else output,
            self.query, self.before, self.missing, self.train, self.train_hash,
            first=first,
        )

    def test_repeat_identity_does_not_substitute_for_observed_value_preservation(self):
        self.assertTrue(all(self.checks(first=self.output.copy()).values()))
        corrupted = self.output.copy()
        corrupted[0, 0] = 9.0
        checks = self.checks(corrupted, first=corrupted.copy())
        self.assertTrue(checks["identical_to_call_1"])
        self.assertFalse(checks["observed_values_preserved"])

    def test_changed_imputed_value_breaks_repeat_identity(self):
        repeated = self.output.copy()
        repeated[0, 1] = 8.0
        checks = self.checks(repeated, first=self.output)
        self.assertTrue(checks["observed_values_preserved"])
        self.assertFalse(checks["identical_to_call_1"])

    def test_shape_dtype_and_nonfinite_output_fail_independently(self):
        checks = self.checks(self.output[:1])
        self.assertFalse(checks["shape_preserved"])
        self.assertFalse(checks["observed_values_preserved"])
        checks = self.checks(self.output.astype(np.float64))
        self.assertFalse(checks["dtype_preserved"])
        self.assertTrue(checks["shape_preserved"])
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                output = self.output.copy()
                output[0, 1] = value
                self.assertFalse(self.checks(output)["finite"])

    def test_input_mutation_and_aliasing_are_detected(self):
        self.query[0, 0] = 9.0
        self.train[0, 0] = 10.0
        checks = self.checks()
        self.assertFalse(checks["query_unchanged"])
        self.assertFalse(checks["training_unchanged"])
        self.assertFalse(self.checks(self.query)["separate_output"])

    def test_reference_differences_are_descriptive_without_an_acceptance_threshold(self):
        reference = self.output.copy()
        reference[0, 1] += 0.25
        result = diagnostic.difference(self.output, reference)
        self.assertTrue(result["comparable"])
        self.assertFalse(result["array_equal"])
        self.assertEqual(result["different_elements"], 1)
        self.assertEqual(result["max_abs_difference"], 0.25)
        self.assertNotIn("passed", result)
        self.assertNotIn("equivalent", result)

    def test_six_call_block_detects_mutation_of_retained_first_output(self):
        expected = self.output.copy()
        events = []

        class FakeModel:
            def __init__(self):
                self.call_count = 0
                self.retained = None

            def fit(self, train):
                events.append(("fit", train))
                return self

            def transform(self, query):
                self.call_count += 1
                events.append((self.call_count, query))
                if self.call_count == 1:
                    self.retained = expected.copy()
                    return self.retained
                if self.call_count == 2:
                    # The new result is correct, but a previously returned
                    # array changes behind the diagnostic's back.
                    self.retained[0, 1] += 10.0
                return expected.copy()

        model = FakeModel()
        helper = SimpleNamespace(current_rss_mib=lambda: 1.0, peak_rss_mib=lambda: 1.0)
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "first.npy"
            with (
                patch.object(diagnostic, "make_case", return_value=(
                    self.train, self.query, expected, self.missing,
                )),
                patch.object(diagnostic, "warm_up", return_value=(
                    self.train[:1], self.query[:1], 0.0,
                )),
                patch.object(diagnostic, "new_model", return_value=model),
            ):
                result = diagnostic.measure_block(
                    helper, load_spec()["diagnostic"], 101, "float32", output_path,
                )
            saved_first = np.load(output_path, allow_pickle=False)
        self.assertEqual([event[0] for event in events], ["fit", 1, 2, 3, 4, 5, 6])
        self.assertIs(events[0][1], self.train)
        self.assertTrue(all(event[1] is self.query for event in events[1:]))
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual([call["call"] for call in result["calls"]], [1, 2, 3, 4, 5, 6])
        self.assertTrue(result["calls"][0]["checks_passed"])
        for call in result["calls"][1:]:
            self.assertTrue(call["checks"]["identical_to_call_1"])
            self.assertFalse(call["checks"]["retained_first_output_unchanged"])
            self.assertFalse(call["checks_passed"])
        self.assertTrue(np.array_equal(saved_first, expected))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestFaissStageTimer(unittest.TestCase):
    def test_instrumentation_preserves_factory_type_arguments_and_return_values(self):
        clock = FakeClock()
        events = []
        training = object()
        queries = object()
        distances, neighbors = object(), object()

        class Index:
            def train(self, values):
                events.append(("train", values))
                clock.advance(3)
                return training

            def add(self, values):
                events.append(("add", values))
                clock.advance(5)
                return values

            def search(self, values, k, *, parameters=None):
                events.append(("search", values, k, parameters))
                clock.advance(7)
                return distances, neighbors

        def factory(dimensions, description, *, metric):
            events.append(("factory", dimensions, description, metric))
            clock.advance(2)
            return Index()

        module = SimpleNamespace(index_factory=factory)
        originals = {name: Index.__dict__[name] for name in ("train", "add", "search")}
        with diagnostic.FaissStageTimer(module, clock=clock) as timer:
            index = module.index_factory(2, "Flat", metric="L2")
            self.assertIs(type(index), Index)
            self.assertIs(index.train(training), training)
            self.assertIs(index.add(training), training)
            actual_distances, actual_neighbors = index.search(queries, 5, parameters="sentinel")
            self.assertIs(actual_distances, distances)
            self.assertIs(actual_neighbors, neighbors)
            self.assertEqual(events, [
                ("factory", 2, "Flat", "L2"), ("train", training),
                ("add", training), ("search", queries, 5, "sentinel"),
            ])
            self.assertEqual(timer.totals, {
                "factory": {"calls": 1, "seconds": 2.0},
                "train": {"calls": 1, "seconds": 3.0},
                "add": {"calls": 1, "seconds": 5.0},
                "search": {"calls": 1, "seconds": 7.0},
            })
            timer.reset()
            index.search(queries, 5)
            self.assertEqual(timer.totals["search"], {"calls": 1, "seconds": 7.0})
            self.assertEqual(timer.totals["factory"]["calls"], 0)
            self.assertNotIn("search", index.__dict__)
            index_reference = weakref.ref(index)
            del index
            gc.collect()
            self.assertIsNone(index_reference(), "Active hooks must not retain native index instances")
        self.assertIs(module.index_factory, factory)
        for name, original in originals.items():
            self.assertIs(Index.__dict__[name], original)

    def test_exception_restores_inherited_methods_and_records_failed_call(self):
        clock = FakeClock()

        class BaseIndex:
            def train(self, values):
                return values

            def add(self, values):
                return values

            def search(self, values, k):
                clock.advance(4)
                raise ValueError("native search failure")

        class ConcreteIndex(BaseIndex):
            pass

        factory = lambda *args: ConcreteIndex()
        module = SimpleNamespace(index_factory=factory)
        original_search = BaseIndex.search
        with self.assertRaisesRegex(ValueError, "native search failure"):
            with diagnostic.FaissStageTimer(module, clock=clock) as timer:
                index = module.index_factory(2, "Flat")
                index.search(object(), 5)
        self.assertIs(module.index_factory, factory)
        self.assertIs(BaseIndex.search, original_search)
        self.assertTrue(all(name not in ConcreteIndex.__dict__ for name in ("train", "add", "search")))
        self.assertEqual(timer.totals["search"], {"calls": 1, "seconds": 4.0})


if __name__ == "__main__":
    unittest.main()
