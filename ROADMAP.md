# Roadmap

Updated after [0.3.6](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.6) to record the available-donor precision fix in 0.3.5 and NumPy integer neighbor-count support in 0.3.6. The original roadmap was based on a source and regression review of the 0.3.4-era commit [`bc592934`](https://github.com/ScionKim/FaissImputer/tree/bc592934e83d5435672a1be31801613ef7b6c06d).

Upcoming work focuses on targeted memory and performance improvements, alongside remaining numerical and interoperability validation. Measurements must identify the version actually tested. These priorities are not release-date commitments or promises of universal speedups or numerical identity with `KNNImputer`.

## Completed through 0.3.4

- Correct neighbor-to-donor mapping, observed-feature search, and fitted fallback statistics.
- Opt-in `donor_policy="available"` support for partially observed training rows, with per-target donor eligibility and fallbacks.
- Reduced complete-donor column-projection overhead and batched available-donor distance calculations.
- Prepared donor-side arrays reused across available-donor query batches, with the retained-memory tradeoff documented.
- Input-preservation checks, failed-refit state cleanup, and available-donor numerical safeguards. Remaining numerical issues are listed below.
- Pipeline and ColumnTransformer integration, feature names, and optional pandas output.
- Minimum/latest dependency CI, package metadata checks, installed-wheel smoke tests, and automated PyPI publishing.
- Reproducible synthetic benchmarks, a real-data MCAR/MAR pilot, and historical batching/thread experiments. Coverage and current-release measurements still need expansion.

## Completed in 0.3.5

### Isolate available-donor precision refinement per query

**Previous problem:** In 0.3.4, a numerical safeguard could switch the entire query batch from float32 to float64 neighbor selection. Adding an unrelated query could therefore change the imputation of an existing query.

Historical reproduction on 0.3.4 with NumPy 2.5.2, scikit-learn 1.9.0, and Faiss 1.15.0:

```python
import numpy as np
from faiss_imputer import FaissImputer

imputer = FaissImputer(n_neighbors=1, donor_policy="available").fit(
    [[1, 0.0001, 10], [1, 0, 20]]
)
imputer.transform([[0, 0, np.nan]])                  # 0.3.4: missing value 10
imputer.transform([[0, 0, np.nan], [1, 0, np.nan]])  # 0.3.4: first missing value 20
```

The distances in this example are distinct before float32 rounding; the closer donor supplies 20. The fix shipped in 0.3.5 returns 20 for the target in both calls. Historical comments above describe the reproduced 0.3.4 behavior, not the fixed result.

Delivered changes:

- Cache direct float64 distances per affected query, retaining FAISS selection for ordinary rows.
- Detect relevant float32 ties, including ties across the candidate boundary, and inspect the full tied group for donors competing to fill the same missing feature.
- Resolve equal direct distances in training-row order for refined rows, reusing their distances as the candidate search expands.
- Add regressions for single versus grouped calls, query reordering, candidate expansion, near and true ties, mean/median aggregation, cache reuse and cleanup, and the ordinary FAISS path. The full suite passed 150 tests.

A [same-runner paired benchmark](https://github.com/ScionKim/FaissImputer/actions/runs/34009545328) compared baseline commit [`386234bd`](https://github.com/ScionKim/FaissImputer/tree/386234bd138d68e46f2b79d0f8c4f7b2dfcbc9d8) with fix commit [`a279e10e`](https://github.com/ScionKim/FaissImputer/tree/a279e10e9a10226cd89732a2030b7ad7fb7b167d) in main/fix/fix/main order on one AMD EPYC 7763 runner. Across 24 Available configurations, mean transform-time changes ranged from -3.92% to +0.72%. No material slowdown was observed in that workload. These measurements apply to the recorded source revisions; they are not a benchmark of the published 0.3.5 wheel. One seed and two observations per version do not establish a speedup or precise overhead, and tie-heavy workloads were not separately benchmarked.

The existing heuristic numerical-risk guard remains unchanged. This release addresses the reproduced batch-wide precision-switch failure and relevant float32 ties; it does not establish batch-independent or exact neighbor ordering for every floating-point input, or numerical identity with `KNNImputer`.

## Completed in 0.3.6

### Accept NumPy integer neighbor counts

- Accept positive Python and NumPy integer scalars for `n_neighbors` under both donor policies.
- Support Pipeline/GridSearchCV parameter grids generated with `np.arange()`.
- Preserve the original estimator parameter for scikit-learn cloning.
- Convert local neighbor counts to Python integers before FAISS search and candidate-expansion arithmetic.
- Explicitly reject Python and NumPy booleans. Python `True` was previously accepted as `1`.
- Preserve policy-specific donor-count limits, failed-fit cleanup, and input preservation.

Validation: 189 tests passed, including 39 new regression cases covering integer types, model selection, cloning, invalid parameters, donor limits, and candidate-expansion overflow.

## Follow-up: targeted performance improvements

### Avoid unused complete-donor index storage

The complete policy currently builds a full-dimensional index during `fit()`, but `transform()` builds projected indexes for its observed-feature patterns instead of searching that stored index.

- Remove or avoid the unused allocation while retaining appropriate fit-time factory validation and fitted-state checks.
- Verify supported metrics/factories, failed refits, feature-name handling, and output behavior.
- Measure fit time, retained fitted memory, and transform time against the corrected baseline.

### Avoid unnecessary candidate expansion for sparse targets

The available policy widens candidate selection for the whole batch until every missing target has enough donors or all donors have been considered. A target with fewer than `n_neighbors` observed donors can force exhaustive selection, including repeated work for already-resolved queries.

- Investigate per-target donor counts and candidate pools, and expansion restricted to unresolved queries.
- Preserve exact eligible-neighbor selection and the documented behavior when fewer than `n_neighbors` donors exist or none share observed features.
- Add a benchmark with highly missing target columns and mixed easy/difficult queries; uniform low-rate MCAR alone does not cover this case.
- Compare outputs with an independent reference and the corrected baseline, including ties and insufficient-donor cases.

## Follow-up: current-release benchmarks and documentation

### Measure the workloads users actually run

The [historical million-row pilot](docs/benchmarks/available-batching-90c8cfb8.md) used one million training rows but only 300 query rows and one timing run. It predates 0.3.4 donor preparation. It does not establish the cost of imputing one million query rows or running `fit_transform()` on one million rows.

- Benchmark the released package being documented, against KNNImputer and an appropriate prior FaissImputer baseline on matching hardware and inputs.
- Vary training rows, query rows, and feature count independently; extend missingness, neighbor-count, and pattern coverage where informative.
- Separate fit, first transform, repeated transforms, and same-data `fit_transform()` at feasible sizes. Exact available-donor pairwise work grows with both donor and query counts.
- Report quality against hidden ground truth separately from output agreement with another imputer.
- Measure retained fitted memory and phase-specific peak memory, distinguishing these from whole-worker peak RSS and internal batch-sizing budgets.
- Use multiple seeds and repetitions. Record versions, hardware, thread limits, timing dispersion, scope limitations, reproduction commands, and linked raw results.
- Extend the current single-dataset real-data pilot with datasets representing useful application workloads. Retain SimpleImputer as a low-cost baseline where appropriate.
- Keep long performance runs manual or scheduled; use small correctness checks in CI rather than noisy per-commit speed gates.

Publish each new report with its measured revision and environment. Keep historical reports labeled as historical rather than relabeling their measurements as current-release results.

### Clarify compatibility and help users choose

- Add a concise KNNImputer comparison covering donor defaults, neighbor defaults, float32 conversion, all-missing columns, and unsupported options such as weights and missing indicators.
- Explain that scikit-learn integration does not imply identical constructor options or identical donor choices. The [real-data pilot](docs/benchmarks/real-data-a3bd1ce3.md) already documents substantive per-cell differences despite similar average errors.
- Keep completed items and remaining evidence gaps in this roadmap current as changes ship.

## Later work, driven by evidence and user needs

- **Extreme numerical scales:** protect mean/median aggregation from intermediate float32 overflow and review complete-donor distance underflow/overflow. Use explicit finite-input reproductions and an independent reference. Reproduced errors on ordinary-scale inputs belong in the next correctness patch.
- **Factory support:** define which index factories remain valid when queries have different observed-feature counts. For example, a factory can accept the fitted dimension and reject a projected dimension. Provide clear validation or a documented fallback.
- **Broader interoperability checks:** add standard scikit-learn estimator checks and address remaining error-message requirements; expand installation/basic-execution coverage to Windows and macOS.
- **Memory controls:** use current measurements to evaluate a public batch/working-memory setting and donor-block processing. An internal batch budget must not be presented as a total RAM limit.
- **Additional API features:** consider distance weighting, missing indicators, and empty-feature policies when concrete use cases justify their behavior and maintenance cost.
- **Approximate search and GPU work:** pursue a specific workload and an acceptable accuracy/performance tradeoff first. Neither changes the priority of consistent results in the existing exact modes.
