# Complete-donor distance-check optimization

This benchmark compares two unreleased FaissImputer builds:

- Before: `0.3.20+bench.0772effd5480`
- After: `0.3.20+bench.fde59b1dd160`

The change reduces routine checking overhead while retaining the existing
complete-donor numerical repair path.

[Benchmark reports](README.md) ·
[Previous query-count study](query-count-sweep-0772effd.md)

## Findings

- All 324 workers completed and passed output and input-preservation checks.
- Before and after outputs matched exactly in all 108 paired measurements.
- Complete-donor first-transform time decreased by **7.6–9.1%**.
- Complete-donor repeated-transform time decreased by **7.8–9.2%**.
- Including fit, complete-donor time decreased by **7.4–9.0%**.
- Available-donor first-transform time changed by less than **0.7%**
  in either direction.

The results support this optimization on the measured workload.
This run directly compares two unreleased commits; it does not establish
the optimized candidate's remaining overhead against published 0.3.20.

## Change under evaluation

The optimized implementation computes conservative coordinate bounds once
per transform, using original coordinates before any in-place imputation.

When coordinate bounds, returned distances, and neighbor indices are safe,
it skips repeated per-pattern coordinate copies and detailed checks.
Zero, subnormal, nonfinite, negative, or otherwise unsafe search results
continue through the existing repair path.

These ordinary-scale benchmarks measure routine processing cost.
Numerical-edge regression tests serve a separate purpose.

## Conditions

| Setting | Value |
| --- | --- |
| Training rows | 20,000 |
| Query rows | 300, 1,000, 3,000 |
| Features | 20 |
| Neighbors | 5 |
| Aggregation | Mean, uniform weights |
| FaissImputer search | Built-in L2, `index_factory="Flat"` |
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
| Available logical CPUs | Four |
| Platform | Linux 6.17.0-1022-azure x86_64 |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Faiss | 1.15.0 |

All query counts ran sequentially on the same runner in ascending order.
Method order rotated across repetitions.

Input fingerprints matched across implementations and repetitions within
each workload. Installed versions matched their expected versions, and
dependency versions and thread limits matched across environments.

## Complete-donor first-transform timing

Values are medians of nine workers per row and implementation, in
milliseconds. Fit time is excluded.

Time reduction is `1 - after / before`.

| Queries | Dtype | Before | After | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 300 | float32 | 156.83 | 142.58 | 9.09% |
| 300 | float64 | 192.46 | 176.90 | 8.08% |
| 1,000 | float32 | 493.61 | 448.68 | 9.10% |
| 1,000 | float64 | 604.93 | 559.10 | 7.58% |
| 3,000 | float32 | 1,269.48 | 1,156.07 | 8.93% |
| 3,000 | float64 | 1,551.34 | 1,429.13 | 7.88% |

In each of these six groups, the slowest optimized first-transform
measurement was faster than the fastest baseline measurement.
This describes the observed samples, not a statistical confidence interval.

Against KNNImputer in this same run, the optimized complete-donor build
achieved **2.14–2.73×** first-transform speedups. Fitting itself was slower
than KNNImputer; including fit, speedups were **2.08–2.72×**.

## Available-donor control

Available-donor first-transform medians, in milliseconds:

| Queries | Dtype | Before | After |
| ---: | --- | ---: | ---: |
| 300 | float32 | 95.47 | 95.78 |
| 300 | float64 | 110.41 | 110.52 |
| 1,000 | float32 | 300.64 | 301.13 |
| 1,000 | float64 | 353.38 | 351.07 |
| 3,000 | float32 | 883.86 | 880.09 |
| 3,000 | float64 | 1,038.15 | 1,044.87 |

Changes ranged from approximately −0.65% to +0.65%.
These small differences do not demonstrate an available-donor improvement.

## Timing and memory interpretation

Fit, first transform, and repeated transforms were timed separately after
a small untimed warmup. Timings exclude process startup, input generation,
validation, RSS sampling, and garbage collection between timed phases.

Fit-plus-transform results use the median of each worker's combined
fit and first-transform time. They do not measure a separate
`fit_transform()` call.

At 3,000 queries, median whole-worker peak RSS was:

| Policy | Dtype | Before | After |
| --- | --- | ---: | ---: |
| complete | float32 | 149.95 MiB | 149.90 MiB |
| complete | float64 | 154.54 MiB | 152.45 MiB |
| available | float32 | 257.57 MiB | 257.34 MiB |
| available | float64 | 260.58 MiB | 262.97 MiB |

Whole-worker peak RSS includes imports, data generation, warmup,
validation, and output handling. It is not retained fitted-model memory
or an isolated transform's peak memory.

Raw files include individual timings, repeated-transform results,
summary minima and maxima, and post-fit RSS changes. RSS changes include
allocator effects and should not be treated as exact model sizes.

## Output checks and quality

All workers passed output shape, dtype, finiteness, observed-value
preservation, input-preservation, and repeated-output checks.

Before and after output hashes matched in all 108 paired measurements.
Maximum differences between builds and between repetitions were zero.

Against KNNImputer, maximum absolute output differences were:

| Policy | Dtype | Maximum difference |
| --- | --- | ---: |
| complete | float32 | 0 |
| complete | float64 | 0 |
| available | float32 | 4.768 × 10⁻⁷ |
| available | float64 | 8.882 × 10⁻¹⁶ |

RMSE and MAE against hidden synthetic ground truth are recorded separately
in the raw files. Output agreement does not establish general numerical
equivalence or improved imputation quality.

## Provenance and reproduction

- [GitHub Actions run 35058305479, attempt 1](https://github.com/ScionKim/FaissImputer/actions/runs/35058305479)
- Candidate commit: `fde59b1dd160be9dad64acb785376e9d2454261d`
- Baseline commit: `0772effd548011b08fd646eae80cfc3096673bab`
- [Workflow at the candidate commit](https://github.com/ScionKim/FaissImputer/blob/fde59b1dd160be9dad64acb785376e9d2454261d/.github/workflows/benchmark-released.yml)

The workflow used `candidate` mode with `baseline_commit` set to the
baseline commit above. Both builds were installed as wheels in separate
environments, and measurements ran outside the source checkout.

The query-count sweep used a shared 1,800-second measurement budget
and a 45-minute job timeout. Each query count completed all 108 workers.

Raw results:

- [300 queries](../../benchmarks/results/complete-distance-checks-fde59b1d-q300.json)
- [1,000 queries](../../benchmarks/results/complete-distance-checks-fde59b1d-q1000.json)
- [3,000 queries](../../benchmarks/results/complete-distance-checks-fde59b1d-q3000.json)

Fixed query-count order leaves possible runner drift and order effects.
The conclusions cover this synthetic workload, three seeds, one runner,
and one native thread. They do not establish performance for other
datasets, larger query counts, callable metrics, or numerical edge cases.