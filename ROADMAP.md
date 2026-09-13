# Roadmap

This roadmap tracks development toward 0.3.17 and the priorities that follow.
Supported API options do not imply numerical identity with KNNImputer.

## Current work: 0.3.17

Support callable distance metrics:

- Accept a callable under both donor policies with `index_factory="Flat"`.
- Pass NaN-normalized query and donor rows in the original feature order.
- Validate returned distances, exclude undefined distances, and resolve ties
  in training-row order.
- Support existing aggregation, weights, indicators, empty features,
  output dtypes, pandas output, and copy behavior.
- Add regression coverage and document callback rules and performance costs.

GitHub Tests and Compatibility must pass before merging and releasing.

## Next priority: released-package benchmarks

Start with a small reproducible comparison of the new released package,
KNNImputer, and an appropriate previous FaissImputer release.

- Compare both donor policies and float32/float64 inputs on matching
  hardware and data. Measure callable metrics separately from built-in metrics.
- Report fit, first transform, repeated transforms, and `fit_transform()`
  separately, using feasible training and query sizes.
- Measure retained fitted memory separately from process peak memory.
- Report imputation quality against hidden ground truth separately from
  agreement with another imputer.
- Record package versions, hardware, thread limits, seeds, repetitions,
  timing variation, reproduction commands, and raw results.
- Run performance measurements through manually triggered GitHub Actions.
  Keep routine CI focused on correctness rather than timing thresholds.

Expand workloads and datasets after the initial comparison identifies
useful questions. Historical million-row training measurements do not
establish performance for one million queries.

## Later work, guided by evidence

- **Distance precision:** investigate reproduced complete-donor distance
  underflow, overflow, and ordering errors against an independent reference.
- **Index factories:** clarify support when query masks change the
  projected feature dimension; provide useful errors or documented fallbacks.
- **Interoperability:** expand scikit-learn estimator checks and
  installation/basic-execution coverage on Windows and macOS.
- **Memory controls:** evaluate working-memory settings and donor-block
  processing using measurements. Internal batch budgets are not total RAM limits.
- **Real-data coverage:** extend the current pilot to additional datasets,
  retaining simple baselines and separate quality measurements.
- **Approximate search and GPU:** pursue a concrete workload with an explicit
  accuracy/performance tradeoff.

## Implemented through 0.3.16

- Complete and available donor policies, mean/median aggregation, and
  uniform, distance, and callable weights.
- Missing indicators, empty-feature handling, configurable numeric missing
  markers, and the `nan_euclidean` metric alias.
- Copy control and float32/float64 value preservation.
- Pipelines, feature names, and optional pandas output.
- Search and aggregation batching, numerical safeguards, and failed-fit cleanup.
- Dependency compatibility CI, wheel checks, and automated PyPI publishing.

## History and supporting evidence

- [Release history](https://github.com/ScionKim/FaissImputer/releases)
- [API reference](docs/api.md) and [usage examples](docs/usage.md)
- [Benchmark reports](docs/benchmarks)
- [Available-donor expansion: 0.3.7 versus 0.3.8](docs/benchmarks/available-expansion-0.3.8.md)
- [Historical complete-aggregation recovery pilot](docs/benchmarks/complete-aggregation-e5ef482.md)
- [Historical real-data comparison](docs/benchmarks/real-data-a3bd1ce3.md)

Historical results retain their measured versions and environments.
Detailed implementation history remains available in Git.