# Same-data callable metric benchmark — 76e0230

Benchmark source: `76e02301aed7202656680ff119b8d0f726e67ef4`.
Runner CPU: **Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz**.
GitHub Actions run: [35821117256](https://github.com/ScionKim/FaissImputer/actions/runs/35821117256).

[Raw JSON](../../benchmarks/results/fit-transform-callable-76e0230.json) · [Full-precision summary](../../benchmarks/results/fit-transform-callable-76e0230-summary.json) · [Analysis script](../../benchmarks/analyze_callable_metrics.py)

Raw-file SHA-256: `7672a71977691b5f879a9bf22c92a438a79117e6dd3e9f83ed686cca5042eb47`.

## Scope and aggregation

This source-build study measures 300 and 1,000 training rows, 20 features, 5 neighbors, 10% target MCAR missingness, uniform weights, and one native thread. Phase-memory sampling is disabled.

The 432 records cover three methods, two metrics, two APIs, two dtypes, three seeds, and three repeats at each size. All records passed validation. This report uses only this preserved run.

- APIs and dtypes are reported separately. Both APIs include fitting.
- Time and process peak RSS are median [min–max] over nine records (three seeds × three repeats) per size/method/metric/API/dtype.
- KNN/Faiss is the median of nine matched total_seconds ratios. Values above one favor FaissImputer. Times are not divided after aggregation.
- Pairing uses the same run, size, features, neighbors, target missing rate, missingness pattern, dtype, API, metric, seed, and repeat.
- Callable/builtin is a separate median of nine paired time ratios within the same method; values above one mean the callback takes longer.
- Reconstruction scores and donor counts summarize three seed datasets. Repeats and APIs are not independent accuracy observations.
- Min–max gives the observed range, not a confidence interval. The summary retains unrounded floats and zero-based raw-record indices.

## Metric paths

Builtin mode retains KNNImputer's nan_euclidean metric and FaissImputer's l2 metric. Callable mode supplies the same Python function to all methods. It returns the square root of the shared squared differences multiplied by original feature count / shared feature count, using float64 arithmetic for either input dtype. It returns NaN when no feature is shared.

Callable FaissImputer evaluates distances directly without a Faiss index. Complete mode restricts candidates to fully observed donors. Its timing advantage therefore accompanies a different donor pool; it is not evidence of Faiss index acceleration for callbacks.

## Timing: fit_transform / float32

Each time and RSS cell uses nine records. Each KNN/Faiss cell uses nine matched pairs.

| Metric | Rows | Method | Seconds, median [min–max] | KNN/Faiss | Peak RSS MiB, median [min–max] |
| --- | ---: | --- | ---: | ---: | ---: |
| builtin | 300 | KNNImputer | 0.010598 [0.010256–0.011598] | — | 138.67 [137.88–142.17] |
| builtin | 300 | FaissImputer[complete] | 0.021702 [0.020593–0.023186] | 0.4942× | 138.27 [134.23–142.27] |
| builtin | 300 | FaissImputer[available] | 0.012773 [0.010776–0.013772] | 0.8449× | 138.51 [137.91–142.03] |
| builtin | 1000 | KNNImputer | 0.047222 [0.045556–0.048126] | — | 150.41 [150.31–150.80] |
| builtin | 1000 | FaissImputer[complete] | 0.058894 [0.056650–0.060032] | 0.8026× | 139.98 [136.03–144.29] |
| builtin | 1000 | FaissImputer[available] | 0.054994 [0.053081–0.057278] | 0.8545× | 143.53 [143.40–144.02] |
| callable | 300 | KNNImputer | 0.763551 [0.724669–0.791610] | — | 138.71 [133.91–142.17] |
| callable | 300 | FaissImputer[complete] | 0.156951 [0.128623–0.177934] | 4.8649× | 138.41 [134.03–142.29] |
| callable | 300 | FaissImputer[available] | 1.175666 [1.129534–1.217962] | 0.6456× | 138.51 [134.16–142.03] |
| callable | 1000 | KNNImputer | 8.359955 [8.161847–8.421407] | — | 139.98 [138.34–144.13] |
| callable | 1000 | FaissImputer[complete] | 1.650892 [1.581932–1.690233] | 5.0620× | 139.98 [136.03–144.37] |
| callable | 1000 | FaissImputer[available] | 12.711000 [12.546200–12.791845] | 0.6577× | 139.98 [136.16–144.02] |

## Timing: fit_transform / float64

Each time and RSS cell uses nine records. Each KNN/Faiss cell uses nine matched pairs.

| Metric | Rows | Method | Seconds, median [min–max] | KNN/Faiss | Peak RSS MiB, median [min–max] |
| --- | ---: | --- | ---: | ---: | ---: |
| builtin | 300 | KNNImputer | 0.011050 [0.010969–0.011632] | — | 139.44 [137.91–143.12] |
| builtin | 300 | FaissImputer[complete] | 0.022374 [0.021370–0.023890] | 0.4903× | 139.44 [134.53–143.22] |
| builtin | 300 | FaissImputer[available] | 0.026223 [0.023939–0.027466] | 0.4241× | 139.44 [138.03–142.90] |
| builtin | 1000 | KNNImputer | 0.051844 [0.050426–0.054113] | — | 151.89 [151.67–152.07] |
| builtin | 1000 | FaissImputer[complete] | 0.061075 [0.058530–0.061778] | 0.8682× | 140.81 [137.13–145.30] |
| builtin | 1000 | FaissImputer[available] | 0.102008 [0.099958–0.103336] | 0.5123× | 144.08 [143.97–145.06] |
| callable | 300 | KNNImputer | 0.691996 [0.657254–0.714881] | — | 139.44 [134.53–143.12] |
| callable | 300 | FaissImputer[complete] | 0.145316 [0.121355–0.167504] | 4.7961× | 139.44 [134.53–143.25] |
| callable | 300 | FaissImputer[available] | 1.095502 [1.051943–1.121869] | 0.6329× | 139.44 [134.66–142.90] |
| callable | 1000 | KNNImputer | 7.511054 [7.397514–7.675771] | — | 144.61 [144.34–145.21] |
| callable | 1000 | FaissImputer[complete] | 1.546505 [1.467222–1.579331] | 4.8538× | 140.81 [137.13–145.32] |
| callable | 1000 | FaissImputer[available] | 11.704705 [11.608574–11.921317] | 0.6418× | 140.91 [137.25–145.06] |

## Timing: fit_then_transform / float32

Each time and RSS cell uses nine records. Each KNN/Faiss cell uses nine matched pairs.

| Metric | Rows | Method | Seconds, median [min–max] | KNN/Faiss | Peak RSS MiB, median [min–max] |
| --- | ---: | --- | ---: | ---: | ---: |
| builtin | 300 | KNNImputer | 0.010451 [0.010245–0.010720] | — | 138.84 [138.01–142.17] |
| builtin | 300 | FaissImputer[complete] | 0.021829 [0.020613–0.023177] | 0.4787× | 138.47 [134.16–142.29] |
| builtin | 300 | FaissImputer[available] | 0.012614 [0.011001–0.013630] | 0.8250× | 138.51 [137.80–142.03] |
| builtin | 1000 | KNNImputer | 0.046494 [0.045759–0.048908] | — | 150.49 [150.18–150.69] |
| builtin | 1000 | FaissImputer[complete] | 0.058475 [0.056291–0.059362] | 0.8129× | 139.98 [136.03–144.37] |
| builtin | 1000 | FaissImputer[available] | 0.055119 [0.053223–0.056569] | 0.8474× | 143.54 [143.25–144.07] |
| callable | 300 | KNNImputer | 0.754033 [0.736874–0.780901] | — | 138.86 [134.03–142.26] |
| callable | 300 | FaissImputer[complete] | 0.156756 [0.129681–0.178877] | 4.8064× | 138.51 [134.16–142.29] |
| callable | 300 | FaissImputer[available] | 1.183475 [1.129054–1.219829] | 0.6447× | 138.52 [134.16–142.16] |
| callable | 1000 | KNNImputer | 8.341263 [8.242065–8.572096] | — | 140.11 [138.23–144.29] |
| callable | 1000 | FaissImputer[complete] | 1.665498 [1.568271–1.697292] | 5.0156× | 139.98 [136.03–144.37] |
| callable | 1000 | FaissImputer[available] | 12.656333 [12.558310–12.823054] | 0.6593× | 139.98 [136.28–144.09] |

## Timing: fit_then_transform / float64

Each time and RSS cell uses nine records. Each KNN/Faiss cell uses nine matched pairs.

| Metric | Rows | Method | Seconds, median [min–max] | KNN/Faiss | Peak RSS MiB, median [min–max] |
| --- | ---: | --- | ---: | ---: | ---: |
| builtin | 300 | KNNImputer | 0.011194 [0.010970–0.011411] | — | 139.44 [137.92–143.12] |
| builtin | 300 | FaissImputer[complete] | 0.022215 [0.021241–0.023915] | 0.4958× | 139.44 [134.53–143.25] |
| builtin | 300 | FaissImputer[available] | 0.026357 [0.023914–0.027918] | 0.4216× | 139.44 [137.91–143.08] |
| builtin | 1000 | KNNImputer | 0.051722 [0.049909–0.052889] | — | 151.83 [151.59–151.94] |
| builtin | 1000 | FaissImputer[complete] | 0.061027 [0.058229–0.061720] | 0.8571× | 140.82 [137.25–145.32] |
| builtin | 1000 | FaissImputer[available] | 0.101550 [0.100312–0.103082] | 0.5089× | 144.11 [143.76–145.17] |
| callable | 300 | KNNImputer | 0.683258 [0.667872–0.703136] | — | 139.44 [134.53–143.20] |
| callable | 300 | FaissImputer[complete] | 0.145094 [0.119567–0.164959] | 4.7001× | 139.44 [134.66–143.25] |
| callable | 300 | FaissImputer[available] | 1.076188 [1.052357–1.118886] | 0.6338× | 139.44 [134.66–143.09] |
| callable | 1000 | KNNImputer | 7.534136 [7.440544–7.601825] | — | 144.96 [144.01–145.23] |
| callable | 1000 | FaissImputer[complete] | 1.536864 [1.464908–1.560936] | 4.9169× | 140.88 [137.25–145.32] |
| callable | 1000 | FaissImputer[available] | 11.796751 [11.597572–11.835584] | 0.6402× | 140.97 [137.38–145.18] |

## Callable cost relative to builtin

Every value is the median of nine same-method matched callable/builtin total_seconds ratios. Larger values mean more time.

| Rows | Method | fit_transform f32 | fit_transform f64 | fit_then_transform f32 | fit_then_transform f64 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 300 | KNNImputer | 71.38× | 62.29× | 71.96× | 61.43× |
| 300 | FaissImputer[complete] | 7.23× | 6.49× | 7.12× | 6.51× |
| 300 | FaissImputer[available] | 92.04× | 41.44× | 93.54× | 40.94× |
| 1000 | KNNImputer | 176.23× | 144.79× | 177.89× | 145.83× |
| 1000 | FaissImputer[complete] | 27.67× | 25.03× | 28.22× | 24.97× |
| 1000 | FaissImputer[available] | 229.69× | 115.63× | 229.77× | 114.98× |

## Reconstruction quality

RMSE and MAE measure reconstruction error against hidden ground truth at masked training entries. Each cell below is median [min–max] over three seeds. All API/repeat copies were checked for agreement before retaining one observation per seed.

Similar aggregate errors do not establish equality of individual imputed values or algorithmic equivalence. These measurements concern this one callback and these generated datasets.

### float32

| Metric | Rows | Method | RMSE, median [min–max] | MAE, median [min–max] |
| --- | ---: | --- | ---: | ---: |
| builtin | 300 | KNNImputer | 0.4218582568 [0.3992780790–0.4583194231] | 0.3158855546 [0.3016139144–0.3364907356] |
| builtin | 300 | FaissImputer[complete] | 0.6649903187 [0.6441457574–0.6744750532] | 0.5071523086 [0.5063724022–0.5087643469] |
| builtin | 300 | FaissImputer[available] | 0.4218582589 [0.3992780809–0.4583194218] | 0.3158855553 [0.3016139149–0.3364907349] |
| builtin | 1000 | KNNImputer | 0.3185942690 [0.3161342035–0.3273849846] | 0.2385577457 [0.2357162224–0.2405193804] |
| builtin | 1000 | FaissImputer[complete] | 0.4864270040 [0.4765146674–0.4871380296] | 0.3685694700 [0.3579635408–0.3780090455] |
| builtin | 1000 | FaissImputer[available] | 0.3185942703 [0.3161342058–0.3273849856] | 0.2385577460 [0.2357162234–0.2405193809] |
| callable | 300 | KNNImputer | 0.4218582568 [0.3992780790–0.4583194231] | 0.3158855546 [0.3016139144–0.3364907356] |
| callable | 300 | FaissImputer[complete] | 0.6649903187 [0.6441457574–0.6744750532] | 0.5071523086 [0.5063724022–0.5087643469] |
| callable | 300 | FaissImputer[available] | 0.4218582568 [0.3992780790–0.4583194231] | 0.3158855546 [0.3016139144–0.3364907356] |
| callable | 1000 | KNNImputer | 0.3185942690 [0.3161342035–0.3273849846] | 0.2385577457 [0.2357162224–0.2405193804] |
| callable | 1000 | FaissImputer[complete] | 0.4864270040 [0.4765146674–0.4871380296] | 0.3685694700 [0.3579635408–0.3780090455] |
| callable | 1000 | FaissImputer[available] | 0.3185942690 [0.3161342035–0.3273849846] | 0.2385577457 [0.2357162224–0.2405193804] |

### float64

| Metric | Rows | Method | RMSE, median [min–max] | MAE, median [min–max] |
| --- | ---: | --- | ---: | ---: |
| builtin | 300 | KNNImputer | 0.4218582602 [0.3992780786–0.4583194238] | 0.3158855573 [0.3016139136–0.3364907350] |
| builtin | 300 | FaissImputer[complete] | 0.6649903167 [0.6441457564–0.6744750527] | 0.5071523089 [0.5063723999–0.5087643452] |
| builtin | 300 | FaissImputer[available] | 0.4218582602 [0.3992780786–0.4583194238] | 0.3158855573 [0.3016139136–0.3364907350] |
| builtin | 1000 | KNNImputer | 0.3185942693 [0.3161342063–0.3273849846] | 0.2385577467 [0.2357162233–0.2405193814] |
| builtin | 1000 | FaissImputer[complete] | 0.4864270068 [0.4765146675–0.4871380309] | 0.3685694704 [0.3579635416–0.3780090484] |
| builtin | 1000 | FaissImputer[available] | 0.3185942693 [0.3161342063–0.3273849846] | 0.2385577467 [0.2357162233–0.2405193814] |
| callable | 300 | KNNImputer | 0.4218582602 [0.3992780786–0.4583194238] | 0.3158855573 [0.3016139136–0.3364907350] |
| callable | 300 | FaissImputer[complete] | 0.6649903167 [0.6441457564–0.6744750527] | 0.5071523089 [0.5063723999–0.5087643452] |
| callable | 300 | FaissImputer[available] | 0.4218582602 [0.3992780786–0.4583194238] | 0.3158855573 [0.3016139136–0.3364907350] |
| callable | 1000 | KNNImputer | 0.3185942693 [0.3161342063–0.3273849846] | 0.2385577467 [0.2357162233–0.2405193814] |
| callable | 1000 | FaissImputer[complete] | 0.4864270068 [0.4765146675–0.4871380309] | 0.3685694704 [0.3579635416–0.3780090484] |
| callable | 1000 | FaissImputer[available] | 0.3185942693 [0.3161342063–0.3273849846] | 0.2385577467 [0.2357162233–0.2405193814] |

## Donor counts

These counts agree across dtypes, metrics, methods, APIs, and repeats. Complete donors are fully observed training rows; this is not the available policy's query/feature-specific donor count. The first five training rows were kept complete.

| Rows | Seed | Complete donors | Masked cells | Actual missing rate |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 101 | 37 | 594 | 0.099000 |
| 300 | 202 | 29 | 601 | 0.100167 |
| 300 | 303 | 44 | 579 | 0.096500 |
| 1000 | 101 | 122 | 1991 | 0.099550 |
| 1000 | 202 | 115 | 2043 | 0.102150 |
| 1000 | 303 | 125 | 2009 | 0.100450 |

## Memory and reproduction

Worker peak RSS includes preparation and validation before JSON serialization. It is not fitted-model memory; phase-memory fields are null.

Run the **Analyze callable metric benchmark results** workflow to regenerate this report and the full-precision JSON summary. The script verifies the raw-file hash, provenance, full record grid, shared inputs, API/repeat consistency, and stored summaries. It does not execute imputers or modify the raw JSON.
