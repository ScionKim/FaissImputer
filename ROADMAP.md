# Roadmap

This roadmap tracks released capabilities, merged changes, and remaining
priorities. Supported API options do not imply numerical identity with
KNNImputer.

## Released: 0.3.20

Available-donor search with built-in L2 metrics reuses distance work
buffers, batches preparation work, and releases temporary arrays before
precision refinement.

The [published-package comparison](docs/benchmarks/released_versions_0.3.20.md)
measured a 2.71–2.73× first-transform speedup over KNNImputer on one
available-donor workload. Fit plus first transform was 2.44–2.49× as fast,
although fit itself was slower.

The report records both donor policies, float32/float64 inputs, timing
variation, process memory, output agreement, and synthetic-data quality.
These results do not establish performance on other workloads.

## Merged since 0.3.20

- **Complete-donor distance repair:** native Flat L2 searches selectively
  recompute neighbors in float64 for detected underflow, overflow, and
  invalid search results. Regression coverage includes aggregation,
  weights, query batches, and ties across donor chunks.
- **Estimator checks:** official scikit-learn checks cover both donor
  policies with `n_neighbors=1` through the existing compatibility matrix.
- **Package checks:** wheel and source-distribution installation checks,
  release-metadata regression tests, and Windows/macOS wheel-installation
  smoke checks using Python 3.12.
- **Projected-index errors:** non-Flat failures report the factory,
  observed-feature count, and donor count while preserving the original
  Faiss exception. See [index factory requirements](docs/api.md#complete-donors).

The complete-donor correction is not included in the published 0.3.20
package. It does not guarantee float64 neighbor ordering for every input.

## Next priority: broader benchmark coverage

The [query-count studies](docs/benchmarks/README.md#query-count-studies)
and [same-data API comparison](docs/benchmarks/fit-transform-110e37bf.md)
are complete.

The same-data study compares `fit_transform(X)` with `fit(X)` followed
by `transform(X)`. Its primary conclusions use 10,000 and 20,000 rows.
The 3,000-row case covers the crossover region; 1,000 rows are retained
only for small-data regression tracking.

The [held-out real-data comparison](docs/benchmarks/real-data-coverage-8f289647.md)
is also complete. It covers California Housing with 10,000 and 15,000
training rows, 3,000 held-out queries, MCAR/MAR missingness, and
float32/float64 inputs. Methods include mean and median baselines,
KNNImputer, and both FaissImputer donor policies.

Reports under `docs/benchmarks/` document validation checks and
reproduction instructions, with links to archived raw results.

The [float32 investigation](docs/benchmarks/real-data-coverage-8f289647.md#agreement-with-knnimputer)
is complete for the 15,000-row MAR case with seed 303.

FaissImputer's float32 results remained consistent with the independent
float64 direct-distance reference, while KNNImputer's float32 distance
calculations changed neighbor ordering in the affected rows. Both
implementations agreed when the same prepared inputs were promoted
to float64. This conclusion is limited to the reproduced case.

The phase-separated memory investigation
[is complete for the same-data fit_transform workload](docs/benchmarks/fit-transform-ac84bfa1abec-phase-memory.md).
Retained fitted memory was 0.3–3.8% of process peak RSS at 20,000 rows
across methods and dtypes; whole-process peak RSS is dominated by
transform for KNNImputer and available-donor. Timings from
`--phase-memory` runs are reference-only. This conclusion is limited
to the measured configurations (20 features, 5 neighbors, 10%
missingness, one pinned thread).

The [same-data OFAT sweep](docs/benchmarks/fit-transform-ofat-9683d03.md)
at source commit `9683d03` is complete for the measured configurations.
It covers row counts, feature counts, missing rates, MCAR/MAR
missingness, and neighbor counts.

A reproducible analysis script generates the report and full-precision
summary from twelve preserved raw JSON files. APIs and dtypes are
reported separately, with median [min–max] timings and median
matched-record speedups.

The AMD neighbors sweep varies guaranteed complete rows with k, so
incomplete inputs are not identical across neighbor settings.
The Intel k=30 observation and Intel 50,000-row stress result are
reported separately from the AMD sweeps. The stress result summarizes
three seeds with one repeat per seed.

Remaining work:
- Measure callable metrics separately.
- Extend held-out measurements to additional real datasets, retaining
  simple baselines. Report quality against hidden ground truth
  separately from agreement with another imputer.

Use published wheels for released-package claims and identify development
measurements by source commit. Record matching inputs, hardware, thread
limits, seeds, repetitions, timing variation, reproduction instructions,
and raw results.

Run performance measurements through manually triggered GitHub Actions.
Keep routine CI focused on correctness rather than timing thresholds.
Historical million-row training measurements do not establish performance
for one million queries.

## Later work, guided by evidence

- **Distance precision:** investigate remaining cancellation and ordering
  errors against an independent reference, beyond the cases covered by
  the merged Flat L2 correction.
- **Memory controls:** evaluate configurable working-memory budgets and
  donor-block processing using measurements. Internal batch budgets are
  not total RAM limits.
- **Approximate search and GPU:** pursue a concrete workload with an
  explicit accuracy/performance tradeoff.

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
- Dependency compatibility CI and automated PyPI publishing.
- Released-package and source-candidate benchmarks with archived results.

## History and supporting evidence

- [Release history](https://github.com/ScionKim/FaissImputer/releases)
- [API reference](docs/api.md) and [usage examples](docs/usage.md)
- [Benchmark index](docs/benchmarks/README.md)
- [Published 0.3.20 comparison](docs/benchmarks/released_versions_0.3.20.md)
- [Historical 0.3.19 comparison](docs/benchmarks/released_versions_0.3.19.md)

Historical reports retain their measured versions and environments.
Detailed implementation history remains available in Git.
