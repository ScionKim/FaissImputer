# Partial-donor benchmark: 0a3cc077

This report measures the public `FaissImputer(donor_policy="available")` at [commit `0a3cc077`](https://github.com/ScionKim/FaissImputer/tree/0a3cc077a06dd56eb87de2f1799c31730211400b). This is an unreleased development snapshot, not the PyPI 0.2.2 implementation. Its package metadata still reports version 0.2.2.

[Raw results](../../benchmarks/results/partial-donor-sweep-0a3cc077.json) · [Benchmark source](https://github.com/ScionKim/FaissImputer/blob/0a3cc077a06dd56eb87de2f1799c31730211400b/benchmarks/benchmark_partial_donor_sweep.py)

## Method

- Linux x86_64 (Azure), Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.0, FAISS 1.15.0; FAISS and BLAS limited to one thread.
- Both methods use 5 neighbors with uniform mean aggregation. Only the public available-donor mode is compared with KNNImputer; the default complete-donor mode and median strategy are not evaluated here.
- Correlated synthetic float32 data with 20 features. Training sets have 1,000, 5,000, or 20,000 rows and target missingness of 10%, 30%, or 50%. Each held-out set has 500 rows, each with exactly 4 missing features.
- Seeds: 101, 202, 303, 404, 505; 3 timed repeats per seed and condition, with fresh estimators, untimed warmups, and rotating method order.
- Training prefixes and masks are nested; held-out queries are identical across conditions within each seed. Scaling uses complete maximum-training truth before masking. This is not an end-to-end preprocessing benchmark.
- For each method and condition, take the median of the 3 total times within each seed, then average those 5 medians. Time reduction is `100 * (1 - Faiss_time / KNN_time)`, calculated before rounding.

## Held-out fit + transform

Times include estimator fitting and transformation. Data preparation, warmups, the benchmark harness's input copies, explicit garbage collection, and result checks are outside the timed region.

| Training rows | Training missingness | KNNImputer (ms) | FaissImputer (ms) | Time reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 10% | 32.530 | 23.380 | 28.13% |
| 1,000 | 30% | 30.776 | 25.479 | 17.21% |
| 1,000 | 50% | 28.801 | 30.415 | -5.61% |
| 5,000 | 10% | 117.401 | 80.583 | 31.36% |
| 5,000 | 30% | 107.038 | 81.633 | 23.73% |
| 5,000 | 50% | 100.446 | 91.457 | 8.95% |
| 20,000 | 10% | 472.146 | 324.742 | 31.22% |
| 20,000 | 30% | 422.084 | 329.771 | 21.87% |
| 20,000 | 50% | 407.972 | 346.621 | 15.04% |

FaissImputer took less time in 8 of 9 condition aggregates. At 1,000 training rows and 50% missingness it took 5.61% more time. Each condition's direction held for all 5 seed medians, not necessarily every individual timing.

## Same-data fit_transform

One public `fit_transform` call imputes the same 5,000-row training input, with 20 features and 10% target missingness. Using the same aggregation:

- KNNImputer: 844.892 ms.
- FaissImputer: 699.780 ms.
- Time reduction: 17.18%; faster for all 5 seed medians.

## Quality and limitations

- All 135 held-out comparisons and 15 same-data comparisons passed `rtol=1e-6, atol=1e-6`. The maximum absolute output difference was `4.76837158203125e-7`.
- All 300 timed model runs passed finite-output, observed-value preservation, and input non-mutation checks. Both benchmark completion flags are true.
- These comparisons include repeats of the same data; they are not 150 independent datasets. Agreement within tolerance is not bitwise equality or a guarantee for arbitrary data.
- At 50% training missingness, mean held-out RMSE increased from 0.484760 at 5,000 rows to 0.535588 at 20,000 rows for both methods, rounded to 6 decimals. More donors did not guarantee better accuracy.
- Results describe one synthetic workload family on one single-threaded environment. They do not establish universal speedups, real-world accuracy, or peak-memory usage. Benchmark quality checks do not replace the separate test suite.

## Reproduce

Use a fresh environment with Python 3.12.14 and the measured commit above. From that checkout's root:

```sh
python -m pip install -e . "numpy==2.5.2" "scikit-learn==1.9.0" "faiss-cpu==1.15.0"
python -u -m benchmarks.benchmark_partial_donor_sweep
```

The script limits thread counts, writes `benchmark_outputs/partial_donor_sweep.json`, and fails if output comparisons exceed the stated tolerance. Timing results are measurements, not pass/fail thresholds.
