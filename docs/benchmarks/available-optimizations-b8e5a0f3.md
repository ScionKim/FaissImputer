# Combined available-donor optimizations

This benchmark directly compares both available-donor optimizations
with PyPI FaissImputer 0.3.19 in the same workflow run.

Available-policy first-transform time decreased by 50–54%, and peak
process RSS decreased by 13.0–14.5%. Measured imputation outputs were
identical to 0.3.19.

The candidate includes distance-buffer reuse, floating-point overlap
counting, and bounded search-preparation arrays.

## Versions and conditions

- Baseline: PyPI FaissImputer 0.3.19.
- Candidate: unpublished `0.3.19+bench.b8e5a0f3c8e6`.
- Candidate commit: `b8e5a0f3c8e6d4e5702379dd7cce46924bf512ba`.
- Reference estimator: KNNImputer from scikit-learn 1.9.1.
- CPU: AMD EPYC 9V74; one native thread.
- Python 3.12.14, NumPy 2.5.3, Faiss 1.15.0.
- Training data: 20,000 rows and 20 features.
- Queries: 300 rows, each missing four randomly selected features.
- Complete policy: fully observed training data.
- Available policy: training data with 10% MCAR missingness.
- Both float32 and float64; float64 generated without a float32 round trip.
- Five neighbors, uniform-weight mean aggregation, Flat L2 search.
- Seeds: 101, 202, and 303; three fresh workers per method and seed.
- Sequential workers with method order rotated across repetitions.
- Each worker measured fit, the first transform, and two additional transforms.

Each table entry is the median of nine workers per method and condition.
Timings exclude process startup, data generation, warmup, and validation.

## First-transform time

| Policy | dtype | KNNImputer | 0.3.19 | Candidate |
| --- | --- | ---: | ---: | ---: |
| complete | float32 | 297.67 ms | 139.38 ms | 139.78 ms |
| complete | float64 | 358.24 ms | 174.55 ms | 174.79 ms |
| available | float32 | 286.12 ms | 204.17 ms | 93.23 ms |
| available | float64 | 339.00 ms | 216.84 ms | 108.31 ms |

For available donors, first-transform time decreased by 54.3% for float32
and 50.1% for float64 versus 0.3.19. The candidate was respectively
3.07× and 3.13× as fast as KNNImputer.

Repeated-transform time decreased by 55.7% for float32 and 51.6% for
float64 versus 0.3.19.

## Fit plus first-transform time

Totals are calculated per worker before taking the median.

| Policy | dtype | KNNImputer | 0.3.19 | Candidate |
| --- | --- | ---: | ---: | ---: |
| complete | float32 | 299.27 ms | 144.21 ms | 144.51 ms |
| complete | float64 | 360.11 ms | 180.71 ms | 180.88 ms |
| available | float32 | 287.87 ms | 216.57 ms | 105.19 ms |
| available | float64 | 341.34 ms | 228.50 ms | 120.55 ms |

Available-policy total time decreased by 51.4% for float32 and 47.2%
for float64 versus 0.3.19. Including fit, the candidate was respectively
2.74× and 2.83× as fast as KNNImputer.

Complete-policy total time differed from 0.3.19 by less than 0.3%.

## Peak process memory

| Policy | dtype | KNNImputer | 0.3.19 | Candidate |
| --- | --- | ---: | ---: | ---: |
| complete | float32 | 235.62 MiB | 149.63 MiB | 149.65 MiB |
| complete | float64 | 249.29 MiB | 152.18 MiB | 152.35 MiB |
| available | float32 | 235.43 MiB | 290.51 MiB | 248.27 MiB |
| available | float64 | 249.26 MiB | 288.95 MiB | 251.48 MiB |

Available-policy peak RSS decreased by 14.5% for float32 and 13.0%
for float64 versus 0.3.19. It remained slightly higher than KNNImputer
in these available-policy cases.

Peak RSS covers the entire worker process, including validation and
repeated transforms. It is not an isolated array or fitted-model size.

## Output agreement

All 108 workers completed successfully and passed the benchmark checks.

- Candidate versus 0.3.19: maximum absolute imputed-value difference
  was zero in every measured case.
- Repeated runs produced identical outputs for matching cases.
- Complete-policy outputs matched KNNImputer exactly.
- Available-policy maximum absolute differences from KNNImputer were
  `4.76837158203125e-7` for float32 and `8.881784197001252e-16` for float64.

These are direct comparisons within one run, not percentages combined
from the earlier optimization benchmarks. Results describe this workload
and runner; performance depends on data, settings, and hardware.

## Reproduction and supporting results

Workflow: `benchmark-released.yml`, using `mode=candidate` with
`baseline_commit` left blank to select PyPI 0.3.19.

- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/34918843313)
- [Raw results](../../benchmarks/results/available-optimizations-b8e5a0f3.json)
- [Earlier distance-buffer benchmark](available-distance-buffers-94adbf66.md)
- [Earlier search-memory benchmark](available-distance-memory-7c62738b.md)