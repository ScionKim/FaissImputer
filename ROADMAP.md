# Roadmap

This roadmap covers the 0.3.20 release and the priorities that follow.
Supported API options do not imply numerical identity with KNNImputer.

## 0.3.20: available-donor performance

Improve available-donor search with built-in L2 metrics,
`metric="l2"` and `metric="nan_euclidean"`:

- Reuse work buffers for distance corrections and shared-feature counts.
- Use floating-point matrix multiplication to count shared observed features.
- Compute search-preparation masks and tolerances in groups of query rows.
- Release temporary preparation arrays before precise distance refinement.
- Add regression coverage for precision, chunk boundaries, cache reuse,
  input preservation, and temporary-array lifetime.

A [direct comparison with PyPI 0.3.19](docs/benchmarks/available-optimizations-b8e5a0f3.md)
measured 50–54% less available-donor transform time and 13.0–14.5% lower
peak process RSS. All 108 workers passed their checks, with identical
imputation outputs versus 0.3.19 in every measured case.

These measurements used pre-release source candidate `b8e5a0f3`.
The report identifies the workload, hardware, versions, and raw results.
Callable-metric performance was not measured in this comparison.

## Next priority: broader benchmark coverage

Initial released-package and optimization comparisons are available.
Extend them to workloads that the existing measurements do not cover:

- Vary training size, query count, feature count, missingness patterns,
  and neighbor counts independently.
- Measure same-data `fit_transform()` at feasible sizes alongside fit,
  first transform, and repeated transforms.
- Measure callable metrics separately from built-in metrics.
- Distinguish retained fitted memory and phase-specific peaks from
  whole-process peak RSS.
- Extend real-data coverage, retaining simple baselines and reporting
  quality against hidden ground truth separately from output agreement.
- Use published wheels when reporting released-package performance;
  identify source-candidate measurements by their actual commit and version.

Continue recording matching inputs, hardware, thread limits, seeds,
repetitions, timing variation, reproduction instructions, and raw results.

Run performance measurements through manually triggered GitHub Actions.
Keep routine CI focused on correctness rather than timing thresholds.

Historical million-row training measurements do not establish performance
for one million queries.

## Later work, guided by evidence

- **Distance precision:** investigate reproduced complete-donor distance
  underflow, overflow, and ordering errors against an independent reference.
- **Index factories:** clarify support when query masks change the
  projected feature dimension; provide useful errors or documented fallbacks.
- **Interoperability:** expand scikit-learn estimator checks and
  installation/basic-execution coverage on Windows and macOS.
- **Memory controls:** evaluate working-memory settings and donor-block
  processing using measurements. Internal batch budgets are not total RAM limits.
- **Approximate search and GPU:** pursue a concrete workload with an explicit
  accuracy/performance tradeoff.

## Implemented capabilities

- Complete and available donor policies, mean/median aggregation, and
  uniform, distance, and callable weights.
- Callable distance metrics under both donor policies, including callback
  validation, undefined-distance handling, and deterministic tie handling.
- Missing indicators, empty-feature handling, configurable numeric missing
  markers, and the `nan_euclidean` metric alias.
- Copy control and float32/float64 value preservation.
- Pipelines, feature names, and optional pandas output.
- Search and aggregation batching, numerical safeguards, and failed-fit cleanup.
- Dependency compatibility CI, wheel checks, and automated PyPI publishing.
- Released-package and source-candidate benchmarks with archived results.

## History and supporting evidence

- [Release history](https://github.com/ScionKim/FaissImputer/releases)
- [API reference](docs/api.md) and [usage examples](docs/usage.md)
- [Benchmark index](docs/benchmarks/README.md)
- [Released 0.3.19 comparison](docs/benchmarks/released_versions_0.3.19.md)
- [Available-donor expansion: 0.3.7 versus 0.3.8](docs/benchmarks/available-expansion-0.3.8.md)
- [Historical complete-aggregation recovery pilot](docs/benchmarks/complete-aggregation-e5ef482.md)
- [Historical real-data comparison](docs/benchmarks/real-data-a3bd1ce3.md)

Historical results retain their measured versions and environments.
Detailed implementation history remains available in Git.