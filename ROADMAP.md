# Roadmap

Updated for [0.3.13](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.13) to record empty-feature handling and configurable missing-value markers, and to track the remaining compatibility options. The original roadmap was based on a source and regression review of the 0.3.4-era commit [`bc592934`](https://github.com/ScionKim/FaissImputer/tree/bc592934e83d5435672a1be31801613ef7b6c06d).

FaissImputer 0.3.13 adds `keep_empty_features` and configurable `missing_values`. The first three prioritized compatibility options (missing indicators, empty-feature handling, and missing-value markers) are now implemented. Further numerical and interoperability work remains guided by reproductions. These priorities are not release-date commitments or promises of universal speedups or numerical identity with `KNNImputer`.

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

## Completed in 0.3.7

### Avoid unused complete-donor Flat index storage

- Skip full-dimensional donor insertion during `fit()` when the complete policy uses `index_factory="Flat"`.
- Retain an empty fitted index for metadata; `transform()` continues to build projected indexes for observed-feature patterns.
- Preserve training and insertion validation for other index factories.
- Add regressions for L2/IP behavior, projected donor selection, and factory validation.

Validation: 196 tests passed, including seven new regression cases.

A dedicated released-version comparison of complete-policy fit time, retained fitted memory, and transform time remains part of the benchmark work below.

## Completed in 0.3.8

### Avoid unnecessary candidate expansion for sparse targets

- Compute per-feature observed donor counts during fit.
- Exclude completed queries from further candidate expansion.
- Stop when every missing target has enough donors, all observed donors for those targets have been found, or finite neighbors are exhausted.
- Reuse prepared distance matrices and per-query float64 refinements for unresolved queries, remapping cached row indices after completed queries are removed.
- Preserve partial fills, fitted-statistic fallbacks, input preservation, and cache cleanup.

Validation: 209 tests passed, including 13 new regression cases covering mixed and reordered queries, sparse targets, finite-neighbor exhaustion, cache remapping, later precision refinement, and numerical extremes.

The [v0.3.7 versus v0.3.8 benchmark](docs/benchmarks/available-expansion-0.3.8.md) covered 12 configurations and 48 successful workers on one runner. Each configuration used v0.3.7/v0.3.8/v0.3.8/v0.3.7 order. Output differences were zero within every comparison.

Targeted sparse and exhausted-neighbor workloads reduced mean transform time by 88.79-91.07% and process peak RSS by 27.47-46.66%. Ordinary workloads showed small differences, with baseline variability affecting the largest apparent improvement. Mean fit time increased by 4.60-8.85%, approximately 0.27-1.69 ms.

These are synthetic pilot results using one seed and two observations per release. Peak RSS includes process setup, data generation, warmup, and fit. The report and archived raw results document the full scope and limitations. The existing heuristic numerical-risk guard remains unchanged; general exact neighbor ordering is not established.

## Completed in 0.3.9

### Prevent intermediate overflow in mean and median aggregation

- Reproduce nonfinite fitted statistics and imputations from finite float32 inputs whose expected aggregates are representable.
- Recompute only nonfinite aggregation results using float64, processing one affected row or column at a time.
- Preserve ordinary float32 aggregation results, output dtype, and input preservation.
- Cover both donor policies, mean and median, selected-neighbor aggregation, all-missing query fallback, and available-donor no-overlap fallback.
- Validate against an independent higher-precision reference after float32 input conversion.

Validation: 229 tests passed in GitHub CI, including 20 new regression cases covering positive, negative, and mixed-sign values. A [paired v0.3.8/v0.3.9 run](https://github.com/ScionKim/FaissImputer/actions/runs/34134144596) subsequently found complete-policy transform regressions. The batching change below addresses that overhead while retaining the overflow repair.

## Completed in 0.3.10

### Batch complete-donor aggregation while retaining overflow repair

- Aggregate selected donor values across query chunks instead of calling the aggregation helper once per query.
- Preserve neighbor search, ordinary float32 reduction order, nonfinite-result repair, output dtype, and input preservation.
- Retain per-query handling for search results containing invalid neighbor IDs, including the existing all-invalid error.
- Bound query chunk size and target an approximately 8 MiB donor-value gather. This is an internal sizing target, not a total-memory limit; a single query and other temporary arrays can exceed it.
- Add 15 regression cases covering reduction order, chunk boundaries, reordered queries, extreme-value repair, invalid IDs, and all-invalid errors. The implementation PR passed all eight GitHub checks.

The [complete-aggregation recovery pilot](docs/benchmarks/complete-aggregation-e5ef482.md) compared v0.3.8 source with commit `e5ef4825213d86228e3e041ea4a69c3396f39416`, which still carried 0.3.9 package metadata. It used one runner, three seeds, five query/feature shapes, and baseline/candidate/candidate/baseline order. All 240 workers and 60 seed/configuration comparisons validated; outputs were byte-identical and fitted-statistic hashes matched within each comparison.

Complete-policy first-transform time improved in every group and seed: paired group reductions were 16.69-38.77% for mean and 29.69-60.96% for median. Available-policy changes ranged from -3.59% to +3.84%, with nine of ten groups slower. Some complete-policy measurements had substantial repeat spread; the report records both improvements and remaining overhead.

This ordinary-scale source-checkout pilot does not establish released-wheel performance, extreme-value repair cost, repeated-transform performance, imputation quality, or retained fitted memory. Those evidence gaps remain below.

## Completed after 0.3.10, through 0.3.12

- Add distance and callable weights for mean aggregation under both donor policies, with uniform weighting remaining the default.
- Add optional missingness indicators learned from all training rows before donor filtering, including feature names and pandas output.
- Document the released 0.3.10 KNNImputer comparisons and the differences in supported options and defaults. Historical performance results retain their measured versions.

## Completed in 0.3.13

### Handle empty training columns

- Add `keep_empty_features=False`: omit columns entirely missing during fit, or retain them with zero values when enabled.
- Use non-empty features for donor selection and imputation while preserving the original input schema and distance normalization.
- Keep indicators based on original query values and align output feature names and pandas columns with the selected policy.
- Support entirely empty training data, and clear learned feature selection after failed refits.

### Support configurable missing values

- Add `missing_values=np.nan`, with support for finite numeric markers under both donor policies.
- Detect markers before float32 conversion, preserving the distinction between missing entries and observed values that round to the same float32 value.
- Use the same missingness decisions for donor preparation, fallback statistics, indicators, and empty-column handling.
- Preserve input data and clear learned state after failed refits. Reject unexpected NaN entries when a numeric marker is configured.

## Next API option

The remaining compatibility candidates are the `nan_euclidean` metric name and callable metrics, preserving float64 inputs, and a `copy` option. Assess their implementation cost and performance effects before selecting the next feature.

## Remaining benchmark and documentation work

### Measure the workloads users actually run

The [historical million-row pilot](docs/benchmarks/available-batching-90c8cfb8.md) used one million training rows but only 300 query rows and one timing run. It predates 0.3.4 donor preparation. It does not establish the cost of imputing one million query rows or running `fit_transform()` on one million rows.

- Benchmark the released package being documented, against KNNImputer and an appropriate prior FaissImputer baseline on matching hardware and inputs.
- Extend the query/feature-size pilot with independent training-size sweeps and broader missingness, neighbor-count, and pattern coverage. Repeat the available-policy controls to determine whether their observed overhead persists.
- Separate fit, first transform, repeated transforms, and same-data `fit_transform()` at feasible sizes. Exact available-donor pairwise work grows with both donor and query counts.
- Report quality against hidden ground truth separately from output agreement with another imputer.
- Measure retained fitted memory and phase-specific peak memory, distinguishing these from whole-worker peak RSS and internal batch-sizing budgets.
- Use multiple seeds and repetitions. Record versions, hardware, thread limits, timing dispersion, scope limitations, reproduction commands, and linked raw results.
- Extend the current single-dataset real-data pilot with datasets representing useful application workloads. Retain SimpleImputer as a low-cost baseline where appropriate.
- Keep long performance runs manual or scheduled; use small correctness checks in CI rather than noisy per-commit speed gates.

Publish each new report with its measured revision and environment. Keep historical reports labeled as historical rather than relabeling their measurements as current-release results.

### Clarify compatibility and help users choose

- Keep the KNNImputer comparison current as options ship, covering donor and neighbor defaults, precision, empty columns, missing-value markers, weights, and indicators.
- Explain that scikit-learn integration does not imply identical constructor options or identical donor choices. The [real-data pilot](docs/benchmarks/real-data-a3bd1ce3.md) already documents substantive per-cell differences despite similar average errors.
- Keep completed items and remaining evidence gaps in this roadmap current as changes ship.

## Later work, driven by evidence and user needs

- **Distance numerical scales:** review complete-donor distance underflow/overflow using finite-input reproductions and an independent reference. This remains separate from the aggregation overflow fix completed in 0.3.9. Reproduced errors on ordinary-scale inputs take priority.
- **Factory support:** define which index factories remain valid when queries have different observed-feature counts. For example, a factory can accept the fitted dimension and reject a projected dimension. Provide clear validation or a documented fallback.
- **Broader interoperability checks:** add standard scikit-learn estimator checks and address remaining error-message requirements; expand installation/basic-execution coverage to Windows and macOS.
- **Memory controls:** use current measurements to evaluate a public batch/working-memory setting and donor-block processing. An internal batch budget must not be presented as a total RAM limit.
- **Additional API features:** continue the compatibility sequence above; distance weighting, missing indicators, empty-feature handling, and configurable missing-value markers are implemented. The remaining candidates are metric compatibility, float64 preservation, and a copy option.
- **Approximate search and GPU work:** pursue a specific workload and an acceptable accuracy/performance tradeoff first. Neither changes the priority of consistent results in the existing exact modes.
