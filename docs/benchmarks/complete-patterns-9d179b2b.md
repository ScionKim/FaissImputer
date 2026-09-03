# Complete-donor missingness-pattern benchmark

This experiment evaluates the cost of many distinct query missingness patterns when `FaissImputer` uses `donor_policy="complete"`.

The candidate changes the fitted donor array from the default C-order copy to a Fortran-order copy:

```python
self.donors_ = X[mask].copy(order="F")
```

This candidate is development code and is not part of the published PyPI 0.3.0 package.

## Method

The benchmark used:

- complete training rows with 20 features;
- 300 query rows, each with 4 missing features;
- 1, 8, or approximately 286–296 distinct query missingness patterns;
- `n_neighbors=5`, `metric="l2"`, `index_factory="Flat"`, and `strategy="mean"`;
- one native-library thread;
- fresh estimators for every timed repetition;
- three repetitions after an untimed warm-up;
- alternating FaissImputer/KNNImputer execution order.

The expanded sweep covered 1,000, 5,000, and 20,000 training rows with five seeds. Each value below is the mean of the five per-seed three-run medians.

The GitHub Actions environment reported Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.0, and Faiss 1.15.0.

## Correctness

All 45 conditions and 135 recorded output validations passed.

For these datasets, the maximum absolute difference between the missing values imputed by FaissImputer and KNNImputer was `0`. The checks also confirmed that:

- outputs were finite `float32` arrays;
- observed query values were unchanged;
- training and query inputs were not modified;
- first and repeated transforms agreed;
- instrumented and uninstrumented FaissImputer runs agreed.

These results apply to the benchmarked configuration and do not establish equivalence for every possible dataset or parameter combination.

## Before-and-after pilot

The initial pilot used 20,000 training rows, 300 queries, seed 101, and three repetitions.

| Query patterns | Baseline first transform | Candidate first transform | Change | Baseline fit + first | Candidate fit + first | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.364 ms | 30.994 ms | -1.2% | 34.193 ms | 34.481 ms | +0.8% |
| 8 | 36.559 ms | 33.678 ms | -7.9% | 39.319 ms | 37.159 ms | -5.5% |
| 296 | 215.861 ms | 131.080 ms | -39.3% | 218.613 ms | 134.564 ms | -38.4% |

The baseline and candidate were measured in separate GitHub Actions runs. KNNImputer control timings differed by approximately 1–2%, so these percentages should be treated as observed CI results rather than hardware-independent guarantees.

The candidate added approximately 0.66–0.76 ms to FaissImputer fitting in this pilot. No peak-memory measurement was performed.

For the 296-pattern case, the instrumented residual outside the recorded Faiss index operations fell from approximately 162.0 ms to 77.2 ms. This residual also includes validation, copying, grouping, aggregation, and instrumentation overhead, so the reduction cannot be attributed entirely to donor-column extraction.

## Expanded candidate sweep

The following table reports fit plus the first transform. `KNN/Faiss` values above 1 mean FaissImputer was faster.

| Training rows | Query patterns | FaissImputer | KNNImputer | KNN/Faiss |
|---:|---:|---:|---:|---:|
| 1,000 | 1 | 6.76 ms | 19.04 ms | 2.82× |
| 1,000 | 8 | 7.58 ms | 21.97 ms | 2.90× |
| 1,000 | 286–296 | 31.07 ms | 22.63 ms | 0.73× |
| 5,000 | 1 | 12.74 ms | 75.56 ms | 5.93× |
| 5,000 | 8 | 14.27 ms | 76.18 ms | 5.34× |
| 5,000 | 286–296 | 54.17 ms | 77.23 ms | 1.43× |
| 20,000 | 1 | 34.84 ms | 311.11 ms | 8.93× |
| 20,000 | 8 | 37.87 ms | 314.76 ms | 8.31× |
| 20,000 | 286–296 | 133.43 ms | 313.60 ms | 2.35× |

FaissImputer was faster in 40 of the 45 per-seed conditions. All five slower conditions were the 1,000-row random-pattern cases, where FaissImputer took approximately 37% more total time than KNNImputer.

No matched pre-change sweep was run for the additional training sizes and seeds, so the slower 1,000-row result must not be interpreted as a regression caused by the layout change.

## Interpretation

The Fortran-order donor layout is a small, bounded optimization that preserves the existing algorithm and adds no retained cache per query pattern. It substantially reduced the measured many-pattern cost in the 20,000-row pilot.

Per-pattern index construction remains significant when nearly every query has a different missingness pattern. For small training sets, that fixed overhead can make KNNImputer faster. More algorithmic work would therefore be required to improve this corner without compromising exact Flat L2 behavior or allowing memory use to grow with every new pattern.

These measurements support retaining the layout change as a limited optimization. They do not support a claim that FaissImputer is faster for every dataset, and they are not directly comparable with numeric targets measured in the historical 0.2.0 benchmark environment.

## Raw results

- [Pre-change baseline](../../benchmarks/results/complete-patterns-baseline-6b9fcf3e.json)
- [Single-seed candidate pilot](../../benchmarks/results/complete-patterns-candidate-eaddb771.json)
- [Expanded candidate sweep](../../benchmarks/results/complete-patterns-sweep-9d179b2b.json)
