# Query-count sweep: candidate versus 0.3.20 and KNNImputer

This benchmark compares the unreleased candidate
`0.3.20+bench.0772effd5480`, published FaissImputer 0.3.20, and
scikit-learn KNNImputer as query count increases.

[Benchmark reports](README.md) · [Project README](../../README.md#performance)

## Findings

- All 324 benchmark workers completed and passed output and
  input-preservation checks.
- Candidate outputs matched published 0.3.20 exactly in all paired cases.
- Complete-donor first-transform time increased by **10.0–13.0%**
  relative to 0.3.20. The candidate includes additional numerical
  safeguards; their performance cost warrants further investigation.
- Available-donor first-transform time changed by less than **0.5%**
  in either direction relative to 0.3.20.
- Against KNNImputer, candidate first-transform speedups were
  **1.92–2.45×** for complete donors and **3.04–3.41×** for available donors.
- Candidate fitting was slower than KNNImputer fitting. Including fit,
  speedups were **1.87–2.44×** for complete donors and **2.72–3.37×**
  for available donors.

These are results for one synthetic workload family on one runner.
They do not establish a general speedup or describe a new published release.

## Conditions

| Setting | Value |
| --- | --- |
| Training rows | 20,000 |
| Query rows | 300, 1,000, 3,000 |
| Features | 20 |
| Neighbors | 5 |
| Aggregation | Mean, uniform weights |
| Candidate and 0.3.20 search | Built-in L2, `index_factory="Flat"` |
| Complete-policy training data | Fully observed |
| Available-policy training data | 10% MCAR missingness |
| Query missingness | Four randomly selected missing features per row |
| Input dtypes | float32 and float64 |
| Seeds | 101, 202, 303 |
| Repetitions | Three fresh workers per seed and method |
| Transforms per worker | First transform, then two additional transforms |
| Native threads | One |
| KNNImputer working memory | 256 MiB |
| CPU | AMD EPYC 9V74 80-Core Processor |
| Runner CPUs | Four logical CPUs available |
| Platform | Linux 6.17.0-1022-azure x86_64 |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Faiss | 1.15.0 |

All three query counts ran sequentially on the same runner, in ascending
order. Methods rotated execution order across repetitions.

Within each policy, dtype, and seed, input fingerprints matched across
methods and repetitions. Training fingerprints also matched across query
counts. The JSON records do not establish that smaller query arrays are
prefixes of larger ones.

## First-transform timing

Values are medians of nine successful workers per table row and method,
in milliseconds. Fit time is excluded.

`Change` is candidate time relative to 0.3.20; positive values mean slower.
`Speedup` is KNNImputer time divided by candidate time.

| Queries | Policy | Dtype | 0.3.20 | Candidate | KNNImputer | Change | Speedup |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 300 | complete | float32 | 139.64 | 157.76 | 302.55 | +13.0% | 1.92× |
| 300 | complete | float64 | 175.23 | 192.71 | 376.11 | +10.0% | 1.95× |
| 300 | available | float32 | 94.72 | 95.05 | 288.70 | +0.3% | 3.04× |
| 300 | available | float64 | 109.93 | 110.39 | 344.39 | +0.4% | 3.12× |
| 1,000 | complete | float32 | 441.50 | 497.20 | 1,029.64 | +12.6% | 2.07× |
| 1,000 | complete | float64 | 549.59 | 607.30 | 1,180.11 | +10.5% | 1.94× |
| 1,000 | available | float32 | 295.86 | 296.50 | 962.31 | +0.2% | 3.25× |
| 1,000 | available | float64 | 348.01 | 346.47 | 1,113.15 | −0.4% | 3.21× |
| 3,000 | complete | float32 | 1,135.77 | 1,271.91 | 3,112.46 | +12.0% | 2.45× |
| 3,000 | complete | float64 | 1,403.90 | 1,547.61 | 3,668.26 | +10.2% | 2.37× |
| 3,000 | available | float32 | 874.19 | 873.91 | 2,978.02 | 0.0% | 3.41× |
| 3,000 | available | float64 | 1,030.17 | 1,032.45 | 3,465.27 | +0.2% | 3.36× |

The displayed 0.0% change is rounded.

Fit, first transform, and repeated transforms were timed separately after
a small untimed warmup. Timings exclude process startup, input generation,
validation, RSS sampling, and garbage collection between timed phases.

Fit-plus-transform comparisons use the median of each worker's combined
fit and first-transform time, not the sum of separately computed medians.
They do not measure a separate `fit_transform()` call.

Raw results include individual timings, repeated-transform measurements,
and summary minima and maxima. Small differences should not be interpreted
as reliable improvements without further evidence.

## Memory observations

At 3,000 queries, median whole-worker peak RSS was:

| Policy | Dtype | 0.3.20 | Candidate | KNNImputer |
| --- | --- | ---: | ---: | ---: |
| complete | float32 | 149.86 MiB | 149.83 MiB | 572.24 MiB |
| complete | float64 | 154.62 MiB | 152.44 MiB | 698.95 MiB |
| available | float32 | 257.53 MiB | 257.21 MiB | 572.26 MiB |
| available | float64 | 260.56 MiB | 263.07 MiB | 712.59 MiB |

Whole-worker peak RSS includes imports, data generation, warmup,
validation, and output handling. It is not retained fitted-model memory
or the peak of an isolated transform call.

The raw files also record post-fit RSS changes. Those measurements include
allocator effects and should not be treated as exact model sizes.

## Output checks and quality

All workers passed output shape, dtype, finiteness, observed-value
preservation, input-preservation, and repeated-output checks.

Candidate and 0.3.20 output hashes matched in all 108 paired measurements.
Maximum output difference between those versions was zero.

Against KNNImputer, maximum absolute output differences were:

| Policy | Dtype | Maximum difference |
| --- | --- | ---: |
| complete | float32 | 0 |
| complete | float64 | 0 |
| available | float32 | 4.768 × 10⁻⁷ |
| available | float64 | 8.882 × 10⁻¹⁶ |

RMSE and MAE against hidden synthetic ground truth are recorded separately
in the raw files. Matching another implementation is not itself a measure
of imputation quality.

This ordinary-scale workload does not replace regression tests for
distance underflow, overflow, or other numerical edge cases.

## Provenance and reproduction

- [GitHub Actions run 35052350396, attempt 1](https://github.com/ScionKim/FaissImputer/actions/runs/35052350396)
- Source commit: `0772effd548011b08fd646eae80cfc3096673bab`
- Candidate wheel: `0.3.20+bench.0772effd5480`
- Baseline: published FaissImputer `0.3.20`
- [Workflow at the measured commit](https://github.com/ScionKim/FaissImputer/blob/0772effd548011b08fd646eae80cfc3096673bab/.github/workflows/benchmark-released.yml)

The workflow used `candidate` mode with an empty `baseline_commit`.
It installed the candidate wheel and published baseline in separate
environments with matching dependencies, then ran the benchmark outside
the source checkout.

The three query counts shared a 1,800-second measurement budget.
Each completed all 108 planned workers. The job timeout was 45 minutes.

Raw results:

- [300 queries](../../benchmarks/results/query-count-sweep-0772effd-q300.json)
- [1,000 queries](../../benchmarks/results/query-count-sweep-0772effd-q1000.json)
- [3,000 queries](../../benchmarks/results/query-count-sweep-0772effd-q3000.json)

The earlier interrupted run used an Intel runner and is excluded from
these tables. Its timings must not be combined with this AMD run.

Query counts ran in a fixed ascending order, so runner drift and execution
order remain possible influences. These measurements cover up to 3,000
queries with 20,000 training rows; they do not establish million-query
performance, callable-metric performance, or behavior on other datasets.