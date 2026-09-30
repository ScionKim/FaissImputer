"""Offline checks for diagnostic tracing; no real model or dataset is run."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import profile_available_transform as profile


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Rows:
    """A shape/length token, not a numerical array."""

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


class Flags:
    def sum(self):
        return 1


class FakeIndex:
    def __init__(self, clock, faiss):
        self.clock = clock
        self.faiss = faiss
        self.donors64 = Rows(4)
        self.present = object()
        self.query_ref = None
        self.matrix = None
        self.precise_rows = {}
        self.inputs_seen = []
        self.raise_in_kernel = False

    def clear_cache(self):
        self.query_ref = None
        self.matrix = None
        self.precise_rows = {}

    def _prepared_distances(self, queries):
        self.clock.advance(2)
        return Rows(len(queries))

    def _prepare_search_matrix(self, queries):
        matrix = self._prepared_distances(queries)
        self.clock.advance(1)
        return matrix, Flags()

    def _distances_to(self, query, donors, present):
        if self.raise_in_kernel:
            raise ValueError("fake distance failure")
        self.clock.advance(3)
        return [0.0] * len(donors)

    def _direct_distances(self, query):
        return self._distances_to(query, self.donors64, self.present)

    @staticmethod
    def _precise_topk(distances, k):
        return distances[:k], list(range(min(k, len(distances))))

    def search(self, queries, k):
        self.inputs_seen.append(queries)
        if queries is not self.query_ref:
            self.clear_cache()
            self.matrix, _ = self._prepare_search_matrix(queries)
            self.precise_rows[0] = self._direct_distances(object())
            self.query_ref = queries
        self.faiss.kmin(self.matrix, k)
        if 1 not in self.precise_rows:
            self.precise_rows[1] = self._direct_distances(object())
        for distances in self.precise_rows.values():
            self._precise_topk(distances, k)
        # Deliberately use every donor as selected candidates. Classification
        # must depend on nesting, not len(donors) == total donor count.
        self._distances_to(object(), self.donors64, self.present)
        self.clock.advance(1)
        return object(), object()

    def retain_queries(self, rows):
        self.query_ref = Rows(len(rows))
        self.matrix = Rows(len(rows))
        self.precise_rows = {}
        self.clock.advance(1)
        return self.query_ref


class FakeModel:
    def __init__(self, clock, faiss):
        self.clock = clock
        self.available_index_ = FakeIndex(clock, faiss)
        self.expansion = False
        self.initial_queries = None
        self.retained_queries = None

    def _aggregate(self, values, *, axis, ignore_nan):
        self.clock.advance(2)
        return values

    def _transform_available_batched(self, queries):
        self.initial_queries = queries
        self.available_index_.search(queries, 16)
        if self.expansion:
            queries = self.available_index_.retain_queries([1])
            self.retained_queries = queries
            self.available_index_.search(queries, 32)
        self._aggregate(Rows(len(queries)), axis=1, ignore_nan=True)
        self.clock.advance(1)
        return queries

    def transform(self, queries):
        try:
            return self._transform_available_batched(queries)
        finally:
            self.available_index_.clear_cache()


def make_fake_case():
    clock = FakeClock()

    def kmin(matrix, k):
        clock.advance(1)
        return object(), object()

    faiss = SimpleNamespace(kmin=kmin)
    model = FakeModel(clock, faiss)
    collector = profile.TimerCollector(clock)
    return clock, faiss, model, collector


class TestTimerCollector(unittest.TestCase):
    def test_nested_self_time_partitions_root_without_double_counting(self):
        clock = FakeClock()
        collector = profile.TimerCollector(clock)
        with collector.measure("instrumented_transform"):
            clock.advance(2)
            with collector.measure("search"):
                clock.advance(3)
                with collector.measure("kernel"):
                    clock.advance(5)
                clock.advance(7)
            clock.advance(11)
        result = collector.snapshot()
        self.assertEqual(result["timings"]["instrumented_transform"]["inclusive_seconds"], 28)
        self.assertEqual(result["timings"]["instrumented_transform"]["self_seconds"], 13)
        self.assertEqual(result["timings"]["search"]["inclusive_seconds"], 15)
        self.assertEqual(result["timings"]["search"]["self_seconds"], 10)
        self.assertEqual(result["timings"]["kernel"]["self_seconds"], 5)
        self.assertEqual(result["accounting"]["sum_of_all_traced_self_seconds"], 28)
        self.assertEqual(result["accounting"]["root_minus_self_sum_seconds"], 0)

    def test_exception_unwinds_frames_and_preserves_original_exception(self):
        clock = FakeClock()
        collector = profile.TimerCollector(clock)
        with self.assertRaisesRegex(ValueError, "original failure"):
            with collector.measure("instrumented_transform"):
                clock.advance(2)
                with collector.measure("kernel"):
                    clock.advance(3)
                    raise ValueError("original failure")
        self.assertEqual(collector.stack, [])
        result = collector.snapshot()
        self.assertEqual(result["timings"]["kernel"]["inclusive_seconds"], 3)
        self.assertEqual(result["accounting"]["sum_of_all_traced_self_seconds"], 5)


class TestAvailableTrace(unittest.TestCase):
    def test_full_refinement_and_selected_calls_are_classified_by_nesting(self):
        _, faiss, model, collector = make_fake_case()
        with profile.AvailableTrace(model, faiss, collector):
            with collector.measure("instrumented_transform"):
                model.transform(Rows(2))
        result = collector.snapshot()
        counters = result["counters"]
        self.assertEqual(counters["full_suspect_row_events"], 1)
        self.assertEqual(counters["full_tie_row_events"], 1)
        self.assertEqual(counters["selected_row_events"], 1)
        self.assertEqual(counters["full_suspect_donor_pair_events"], 4)
        self.assertEqual(counters["full_tie_donor_pair_events"], 4)
        self.assertEqual(counters["selected_donor_pair_events"], 4)
        self.assertEqual(counters["matrix_build_calls"], 1)
        self.assertEqual(counters["precise_topk_calls"], 2)
        self.assertEqual(counters["aggregation_row_events"], 2)
        self.assertEqual(result["accounting"]["root_minus_self_sum_seconds"], 0)
        self.assertTrue(result["search_events"][0]["cache_identity_after"])
        self.assertTrue(profile._cache_is_empty(model.available_index_))

    def test_expansion_keeps_actual_returned_query_and_cache_identity(self):
        _, faiss, model, collector = make_fake_case()
        model.expansion = True
        queries = Rows(2)
        with profile.AvailableTrace(model, faiss, collector):
            with collector.measure("instrumented_transform"):
                result = model.transform(queries)
        self.assertIs(model.initial_queries, queries)
        self.assertIs(model.available_index_.inputs_seen[0], queries)
        self.assertIs(model.available_index_.inputs_seen[1], model.retained_queries)
        self.assertIs(result, model.retained_queries)
        events = collector.search_events
        self.assertEqual([event["requested_k"] for event in events], [16, 32])
        self.assertEqual([event["cache_hit_before"] for event in events], [False, True])
        self.assertEqual([event["query_rows"] for event in events], [2, 1])
        self.assertEqual(collector.counters["matrix_build_calls"], 1)
        self.assertEqual(collector.counters["expansion_search_calls"], 1)
        self.assertTrue(collector.retain_events[0]["returned_cache_identity"])
        # Events must contain scalars, never references to query arrays.
        for event in events + collector.retain_events:
            self.assertTrue(all(isinstance(value, (str, int, float, bool)) for value in event.values()))

    def test_all_patches_and_static_binding_are_restored_after_failure(self):
        _, faiss, model, collector = make_fake_case()
        index = model.available_index_
        original_static = FakeIndex._precise_topk
        model_dict = set(model.__dict__)
        index_dict = set(index.__dict__)
        faiss_original = faiss.kmin
        index.raise_in_kernel = True
        bindings = {
            (id(owner), name): getattr(owner, name)
            for owner, names in (
                (model, ("_transform_available_batched", "_aggregate")),
                (index, ("search", "_prepare_search_matrix", "_prepared_distances",
                         "_direct_distances", "_distances_to", "_precise_topk", "retain_queries")),
                (faiss, ("kmin",)),
            ) for name in names
        }
        with self.assertRaisesRegex(ValueError, "fake distance failure"):
            with profile.AvailableTrace(model, faiss, collector):
                self.assertEqual(index._precise_topk([3, 2, 1], 2), ([3, 2], [0, 1]))
                self.assertIs(FakeIndex._precise_topk, original_static)
                with collector.measure("instrumented_transform"):
                    model.transform(Rows(2))
        self.assertIs(FakeIndex._precise_topk, original_static)
        self.assertIs(faiss.kmin, faiss_original)
        self.assertEqual(set(model.__dict__), model_dict)
        self.assertEqual(set(index.__dict__), index_dict)
        for owner, names in (
            (model, ("_transform_available_batched", "_aggregate")),
            (index, ("search", "_prepare_search_matrix", "_prepared_distances",
                     "_direct_distances", "_distances_to", "_precise_topk", "retain_queries")),
            (faiss, ("kmin",)),
        ):
            for name in names:
                self.assertEqual(getattr(owner, name), bindings[(id(owner), name)])
        self.assertEqual(collector.stack, [])
        self.assertTrue(profile._cache_is_empty(index))
        self.assertEqual(collector.search_events[0]["status"], "error")

    def test_source_contract_rejects_unknown_search_structure(self):
        _, _, model, _ = make_fake_case()
        # The fake algorithm is intentionally not the guarded production
        # structure. Fail instead of guessing refinement causes from it.
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            profile.guard_search_contract(model.available_index_)


class TestCandidateProvenance(unittest.TestCase):
    def test_version_and_source_commit_suffix_must_agree(self):
        commit = "a" * 40
        version = f"0.3.21+bench.{commit[:12]}"
        provenance = {"version": version, "source_commit": commit}
        profile._validate_provenance(provenance, version)
        with self.assertRaisesRegex(ValueError, "version differs"):
            profile._validate_provenance(provenance, "0.3.21")
        with self.assertRaisesRegex(ValueError, "suffix"):
            profile._validate_provenance(
                {"version": version, "source_commit": "b" * 40}, version,
            )
        with self.assertRaisesRegex(ValueError, "full source commit"):
            profile._validate_provenance(
                {"version": version, "source_commit": commit[:12]}, version,
            )


class TestOrchestration(unittest.TestCase):
    def test_timeout_preserves_completed_records_and_continues(self):
        commit = "123456789abc" + "0" * 28
        version = "0.3.21+bench.123456789abc"
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            provenance = folder / "candidate.json"
            provenance.write_text(json.dumps({
                "version": version, "source_commit": commit,
            }), encoding="utf-8")
            output = folder / "profile.json"
            args = profile._parser().parse_args([
                "--data-home", str(folder / "cache"),
                "--expected-version", version,
                "--provenance", str(provenance),
                "--dtypes", "float32", "--seeds", "101", "202", "303",
                "--output", str(output),
            ])
            checkpoints = []

            def fake_worker(command, **kwargs):
                self.assertEqual(kwargs["timeout"], profile.WORKER_TIMEOUT_SECONDS)
                checkpoints.append(json.loads(output.read_text(encoding="utf-8")))
                seed = int(command[command.index("--seed") + 1])
                if seed == 202:
                    raise profile.subprocess.TimeoutExpired(command, kwargs["timeout"])
                destination = Path(command[command.index("--output") + 1])
                ok = seed == 101
                profile._write_json(destination, {
                    "configuration": profile._configuration(args, "float32", seed),
                    "status": "ok" if ok else "error", "checks_passed": ok,
                })
                return profile.subprocess.CompletedProcess(
                    command, 0 if ok else 1, stdout="", stderr="",
                )

            with patch.object(profile.subprocess, "run", side_effect=fake_worker):
                with redirect_stdout(StringIO()):
                    code = profile._orchestrate(args)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([len(item["records"]) for item in checkpoints], [0, 1, 2])
            self.assertEqual(checkpoints[2]["records"][0]["status"], "ok")
            self.assertEqual(
                checkpoints[2]["records"][1]["process_error"]["type"], "TimeoutExpired",
            )
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["counts"], {"records": 3, "ok": 1})
            self.assertEqual(
                [row["configuration"]["seed"] for row in result["records"]],
                [101, 202, 303],
            )


if __name__ == "__main__":
    unittest.main()
