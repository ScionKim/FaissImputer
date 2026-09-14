# Released-version benchmark: FaissImputer 0.3.19

FaissImputer 0.3.19 produced identical outputs to 0.3.16 in all measured
cases. Their median fit-plus-first-transform times differed by less than
0.5%. Compared with KNNImputer, 0.3.19 completed those operations
1.17–1.82 times faster in this run.

All 108 workers completed successfully and passed their output checks.

## Setup

- Run: [GitHub Actions 34804828938](https://github.com/ScionKim/FaissImputer/actions/runs/34804828938)
- Benchmark commit: `4bd784b132857afde6cc6ebb50106b9355ce9434`
- CPU: Intel Xeon Platinum 8370C, 2.80 GHz.
- Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, Faiss 1.15.0.
- Released packages: FaissImputer 0.3.16 and 0.3.19.
- Separate virtual environments with identical shared dependencies.
- Training rows: 20,000; query rows: 300; features: 20.
- Five neighbors, uniform mean aggregation, built-in L2/nan_euclidean distances.
- Input types: float32 and float64, generated directly at the target precision.
- Complete-policy cases: fully observed training data.
- Available-policy cases: training data with 10% MCAR missingness.
- Query mask: the benchmark's `random` pattern.
- Seeds: 101, 202, 303; three repetitions per seed.
- Fresh sequential workers on one runner, with method order rotated.
- One native thread; scikit-learn working memory set to 256 MiB.
- Each fitted model receives a first transform and two additional transforms.

Within each policy, dtype, and seed, all methods receive identical inputs.

## Timing

Values below are milliseconds. Each entry is the median across nine workers
for that method and condition.

The repeated-transform column takes the median of two additional transforms
within each worker, then the median across workers.

| Policy | Input | Method | Fit | First transform | Fit + first transform | Repeated transform |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| complete | float32 | KNNImputer | 1.76 | 339.88 | 341.65 | 333.52 |
| complete | float32 | FaissImputer 0.3.16 | 4.98 | 181.73 | 186.78 | 179.92 |
| complete | float32 | FaissImputer 0.3.19 | 4.85 | 182.53 | 187.46 | 180.37 |
| complete | float64 | KNNImputer | 2.22 | 445.11 | 447.28 | 432.97 |
| complete | float64 | FaissImputer 0.3.16 | 7.05 | 248.76 | 255.81 | 245.73 |
| complete | float64 | FaissImputer 0.3.19 | 7.13 | 249.50 | 256.22 | 246.25 |
| available | float32 | KNNImputer | 1.92 | 318.34 | 320.26 | 316.96 |
| available | float32 | FaissImputer 0.3.16 | 12.61 | 260.61 | 273.37 | 255.43 |
| available | float32 | FaissImputer 0.3.19 | 12.38 | 260.23 | 272.82 | 256.41 |
| available | float64 | KNNImputer | 2.44 | 412.59 | 415.43 | 405.79 |
| available | float64 | FaissImputer 0.3.16 | 13.74 | 279.11 | 292.73 | 272.42 |
| available | float64 | FaissImputer 0.3.19 | 13.57 | 277.83 | 291.41 | 272.37 |

KNNImputer had lower fit times in every condition. FaissImputer's lower
transform times gave it the lower combined times shown above.

Timings exclude process startup, data generation, warmup, validation,
and the garbage collection and RSS sampling between fit and transform.
Each column is summarized independently, so displayed medians need not sum.

## Memory

Median worker peak RSS, in MiB:

| Policy | Input | KNNImputer | FaissImputer 0.3.16 | FaissImputer 0.3.19 |
| --- | --- | ---: | ---: | ---: |
| complete | float32 | 235.93 | 150.04 | 149.95 |
| complete | float64 | 249.58 | 152.62 | 152.55 |
| available | float32 | 235.99 | 291.09 | 291.06 |
| available | float64 | 271.52 | 289.52 | 289.42 |

Compared with KNNImputer, 0.3.19 used approximately 36–39% less peak RSS
in complete-policy cases and 7–23% more in available-policy cases.

Median post-fit RSS increases for 0.3.19 were 2.93 and 5.98 MiB in complete
mode, and 9.48 and 13.06 MiB in available mode, for float32 and float64
respectively. KNNImputer's increases were 1.40 and 3.05 MiB.

Peak RSS covers the entire worker lifetime. Post-fit RSS changes include
allocator effects; neither measurement is an exact fitted-model size.

## Output agreement and imputation quality

- Outputs from FaissImputer 0.3.16 and 0.3.19 were identical.
- Repeated transforms and matching cases across worker repetitions
  produced identical outputs.
- Complete-policy outputs matched KNNImputer exactly.
- Available-policy outputs differed from KNNImputer by at most
  `4.76837158203125e-7` for float32 and `8.881784197001252e-16` for float64.
- Every worker passed checks for output shape, dtype, finite values,
  input preservation, and unchanged observed entries.

Against the hidden synthetic truth, all methods had the following median
scores when rounded to six decimal places:

| Training case | RMSE | MAE |
| --- | ---: | ---: |
| Complete | 0.176254 | 0.132619 |
| Available | 0.180675 | 0.136436 |

Output agreement measures differences between imputers. RMSE and MAE
measure error against the synthetic truth.

## Interpretation

This run found matching outputs and closely similar timing and memory
measurements between FaissImputer 0.3.16 and 0.3.19.

The results cover one synthetic workload and one runner execution.
Performance and memory usage can change with data size, missingness,
neighbor count, thread count, and hardware.