# Held-out real-data benchmarks - ef04b1b

Benchmark source: `ef04b1b0274d3abdd977440dd61e7d91e328c64d`.

[Full-precision summary](../../benchmarks/results/real-data-datasets-ef04b1b-summary.json) | [Analysis script](../../benchmarks/analyze_real_data_datasets.py)

## Scope and aggregation

These runs measure fitting on 1,000 or 3,000 training rows followed by the first transform of 1,000 disjoint held-out query rows. They do not measure same-data fit_transform. Datasets, dtypes, and missingness mechanisms are separate.

- Time uses total_seconds, including fit and first transform. Time and process peak RSS are median [min-max] across nine records (three seeds x three repeats).
- KNN/Faiss is the median of nine matched KNNImputer total_seconds / FaissImputer total_seconds ratios. Above one favors Faiss; it is not a ratio of median times.
- Matching uses dataset/run, feature count, training/query sizes, mechanism, dtype, missing rate, MAR driver/reference prefix, neighbors, weights, seed and repeat. The validated model settings fix five neighbors, uniform weights and Flat Faiss indexes.
- RMSE and MAE summarize reconstruction error against held-out float64 ground truth. Each quality cell is median [min-max] across three seeds after verifying equal repeat metrics. Predictions are not archived: this script reaggregates recorded errors.
- Output difference vs KNN is a separate comparison on masked query entries. Its table value is the maximum across three seed-level maxima, in standardized units. Similar reconstruction metrics do not establish prediction or algorithmic equivalence.
- Scalers use observed training values only and differ by case. Source-unit feature errors refer to values as supplied in the data file, not inferred physical units.
- The selected MAR driver remains observed under both MCAR and MAR. MAR uses its median over a common 1,000-row training prefix. Actual missing rates appear below.
- Peak RSS includes loading, preparation, warmup and validation; timing excludes them. It is whole-worker peak memory. Min-max is an observed range, not a confidence interval.
- Full-precision JSON includes input hashes, record indices, nine paired ratios, three-seed quality values, and per-feature standardized/source-unit errors.

## Wine Quality (white)

Source: [Cortez et al. (2009). Wine Quality. https://doi.org/10.24432/C56S3T](https://archive.ics.uci.edu/dataset/186/wine+quality). License: CC BY 4.0.
Features: 11; excluded columns: quality. Always observed: alcohol.
CPU model: **AMD EPYC 9V74 80-Core Processor**. [Actions run 36066134388](https://github.com/ScionKim/FaissImputer/actions/runs/36066134388).
[Raw JSON](../../benchmarks/results/real-data-datasets-ef04b1b/wine-quality-white.json); SHA-256: `2c4d748590efa26607cf8f6258de0b20cd58fe0bd90247a07363cc6598aec5f1`.

### Timing and memory: float32

Each time/RSS cell uses nine records; each KNN/Faiss cell uses nine matched pairs.

| Train | Pattern | Method | Total seconds, median [min-max] | KNN/Faiss | Peak RSS MiB, median [min-max] |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.001685 [0.001618-0.001773] | - | 148.86 [138.57-159.09] |
| 1000 | MAR | SimpleImputer[median] | 0.002097 [0.002087-0.002175] | - | 148.86 [138.70-159.09] |
| 1000 | MAR | KNNImputer | 0.027395 [0.026536-0.028743] | - | 150.95 [150.47-159.31] |
| 1000 | MAR | FaissImputer[complete] | 0.023397 [0.022790-0.023621] | 1.1893x | 149.15 [138.95-158.12] |
| 1000 | MAR | FaissImputer[available] | 0.043083 [0.042236-0.044443] | 0.6384x | 148.86 [147.11-158.21] |
| 1000 | MCAR | SimpleImputer[mean] | 0.001667 [0.001603-0.001743] | - | 146.02 [137.50-156.34] |
| 1000 | MCAR | SimpleImputer[median] | 0.002092 [0.002076-0.002225] | - | 146.10 [137.70-156.34] |
| 1000 | MCAR | KNNImputer | 0.028402 [0.027161-0.053953] | - | 151.68 [151.30-156.34] |
| 1000 | MCAR | FaissImputer[complete] | 0.019575 [0.019153-0.019804] | 1.4482x | 146.31 [138.43-156.34] |
| 1000 | MCAR | FaissImputer[available] | 0.044577 [0.042755-0.050521] | 0.6534x | 147.35 [147.09-156.34] |
| 3000 | MAR | SimpleImputer[mean] | 0.001925 [0.001879-0.002141] | - | 153.68 [143.11-163.96] |
| 3000 | MAR | SimpleImputer[median] | 0.002822 [0.002776-0.002947] | - | 153.68 [143.41-164.17] |
| 3000 | MAR | KNNImputer | 0.064069 [0.060934-0.065930] | - | 170.59 [169.68-171.35] |
| 3000 | MAR | FaissImputer[complete] | 0.029034 [0.027994-0.029690] | 2.1739x | 153.91 [143.68-163.82] |
| 3000 | MAR | FaissImputer[available] | 0.148245 [0.146094-0.151587] | 0.4236x | 157.19 [156.82-163.96] |
| 3000 | MCAR | SimpleImputer[mean] | 0.001855 [0.001815-0.001916] | - | 151.38 [141.25-160.43] |
| 3000 | MCAR | SimpleImputer[median] | 0.002839 [0.002791-0.002876] | - | 151.42 [141.45-160.62] |
| 3000 | MCAR | KNNImputer | 0.064836 [0.059226-0.066731] | - | 173.18 [172.14-173.48] |
| 3000 | MCAR | FaissImputer[complete] | 0.024721 [0.024334-0.025205] | 2.6227x | 151.59 [141.48-160.34] |
| 3000 | MCAR | FaissImputer[available] | 0.160052 [0.158476-0.160942] | 0.4047x | 157.20 [156.82-160.38] |

### Reconstruction quality and KNN output comparison: float32

RMSE/MAE use three seed observations. The final column is the maximum masked-cell output difference from KNN over those three seeds.

| Train | Pattern | Method | RMSE, median [min-max] | MAE, median [min-max] | Max output difference vs KNN |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.964395 [0.894847-0.969654] | 0.727739 [0.706717-0.744577] | 2.41378 |
| 1000 | MAR | SimpleImputer[median] | 0.959010 [0.883876-0.963386] | 0.703518 [0.685285-0.724817] | 2.55256 |
| 1000 | MAR | KNNImputer | 0.810133 [0.735840-0.813015] | 0.557533 [0.531330-0.571738] | 0 |
| 1000 | MAR | FaissImputer[complete] | 0.843844 [0.778935-0.858813] | 0.591405 [0.574535-0.613297] | 2.10627 |
| 1000 | MAR | FaissImputer[available] | 0.810133 [0.735840-0.813015] | 0.557533 [0.531330-0.571738] | 2.38419e-07 |
| 1000 | MCAR | SimpleImputer[mean] | 1.024886 [0.923903-1.060029] | 0.746989 [0.715064-0.785709] | 2.59556 |
| 1000 | MCAR | SimpleImputer[median] | 1.027281 [0.923925-1.070808] | 0.729334 [0.703806-0.776482] | 2.73491 |
| 1000 | MCAR | KNNImputer | 0.867112 [0.728826-0.905334] | 0.568102 [0.530399-0.602091] | 0 |
| 1000 | MCAR | FaissImputer[complete] | 0.907623 [0.773968-0.963929] | 0.604940 [0.558917-0.652433] | 2.68907 |
| 1000 | MCAR | FaissImputer[available] | 0.867112 [0.728826-0.905334] | 0.568102 [0.530399-0.602091] | 2.38419e-07 |
| 3000 | MAR | SimpleImputer[mean] | 0.967392 [0.911834-0.970593] | 0.725830 [0.719046-0.749753] | 2.71597 |
| 3000 | MAR | SimpleImputer[median] | 0.961961 [0.901149-0.965115] | 0.699361 [0.697341-0.727643] | 2.87733 |
| 3000 | MAR | KNNImputer | 0.780936 [0.702004-0.793664] | 0.513611 [0.503584-0.547645] | 0 |
| 3000 | MAR | FaissImputer[complete] | 0.825936 [0.735690-0.826810] | 0.562784 [0.532444-0.570192] | 2.84218 |
| 3000 | MAR | FaissImputer[available] | 0.780936 [0.702004-0.793664] | 0.513611 [0.503584-0.547645] | 2.38419e-07 |
| 3000 | MCAR | SimpleImputer[mean] | 1.022821 [0.944901-1.060926] | 0.745411 [0.730752-0.794249] | 5.83818 |
| 3000 | MCAR | SimpleImputer[median] | 1.027525 [0.946257-1.073214] | 0.728204 [0.720241-0.784323] | 5.97778 |
| 3000 | MCAR | KNNImputer | 0.801336 [0.694123-0.851476] | 0.504908 [0.499158-0.554542] | 0 |
| 3000 | MCAR | FaissImputer[complete] | 0.863473 [0.738746-0.870314] | 0.560912 [0.526046-0.595083] | 4.03089 |
| 3000 | MCAR | FaissImputer[available] | 0.801336 [0.694397-0.851476] | 0.504908 [0.499191-0.554542] | 0.263018 |

### Timing and memory: float64

Each time/RSS cell uses nine records; each KNN/Faiss cell uses nine matched pairs.

| Train | Pattern | Method | Total seconds, median [min-max] | KNN/Faiss | Peak RSS MiB, median [min-max] |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.001649 [0.001633-0.001713] | - | 150.10 [139.93-159.37] |
| 1000 | MAR | SimpleImputer[median] | 0.002088 [0.002049-0.002168] | - | 150.10 [140.01-159.46] |
| 1000 | MAR | KNNImputer | 0.026648 [0.026243-0.027834] | - | 151.54 [151.01-159.53] |
| 1000 | MAR | FaissImputer[complete] | 0.024040 [0.023556-0.024437] | 1.1051x | 150.39 [140.21-159.31] |
| 1000 | MAR | FaissImputer[available] | 0.070941 [0.069903-0.071549] | 0.3761x | 149.98 [147.52-159.33] |
| 1000 | MCAR | SimpleImputer[mean] | 0.001669 [0.001628-0.001915] | - | 147.27 [137.82-157.02] |
| 1000 | MCAR | SimpleImputer[median] | 0.002114 [0.002059-0.002190] | - | 147.35 [137.82-157.09] |
| 1000 | MCAR | KNNImputer | 0.027161 [0.026294-0.028207] | - | 152.19 [151.90-157.17] |
| 1000 | MCAR | FaissImputer[complete] | 0.019985 [0.019548-0.020492] | 1.3613x | 147.49 [138.59-156.91] |
| 1000 | MCAR | FaissImputer[available] | 0.074835 [0.072065-0.080407] | 0.3631x | 147.84 [147.58-156.93] |
| 3000 | MAR | SimpleImputer[mean] | 0.001891 [0.001827-0.001951] | - | 154.82 [144.60-165.08] |
| 3000 | MAR | SimpleImputer[median] | 0.003061 [0.003015-0.003167] | - | 154.82 [144.84-165.25] |
| 3000 | MAR | KNNImputer | 0.062338 [0.060346-0.066609] | - | 173.10 [172.07-174.07] |
| 3000 | MAR | FaissImputer[complete] | 0.029960 [0.029178-0.030728] | 2.0744x | 154.82 [144.97-165.08] |
| 3000 | MAR | FaissImputer[available] | 0.169603 [0.167399-0.175873] | 0.3681x | 157.66 [157.31-165.08] |
| 3000 | MCAR | SimpleImputer[mean] | 0.001914 [0.001844-0.001999] | - | 152.47 [141.89-161.82] |
| 3000 | MCAR | SimpleImputer[median] | 0.003048 [0.003021-0.003321] | - | 152.47 [141.92-161.88] |
| 3000 | MCAR | KNNImputer | 0.063852 [0.062507-0.065434] | - | 175.79 [174.84-176.24] |
| 3000 | MCAR | FaissImputer[complete] | 0.025790 [0.025145-0.026049] | 2.4783x | 152.47 [142.22-161.68] |
| 3000 | MCAR | FaissImputer[available] | 0.183020 [0.180115-0.184446] | 0.3492x | 157.82 [157.37-161.76] |

### Reconstruction quality and KNN output comparison: float64

RMSE/MAE use three seed observations. The final column is the maximum masked-cell output difference from KNN over those three seeds.

| Train | Pattern | Method | RMSE, median [min-max] | MAE, median [min-max] | Max output difference vs KNN |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.964395 [0.894847-0.969654] | 0.727739 [0.706717-0.744577] | 2.41378 |
| 1000 | MAR | SimpleImputer[median] | 0.959010 [0.883876-0.963386] | 0.703518 [0.685285-0.724817] | 2.55256 |
| 1000 | MAR | KNNImputer | 0.810133 [0.735840-0.813015] | 0.557533 [0.531330-0.571738] | 0 |
| 1000 | MAR | FaissImputer[complete] | 0.843844 [0.778935-0.858813] | 0.591405 [0.574535-0.613297] | 2.10627 |
| 1000 | MAR | FaissImputer[available] | 0.810133 [0.735840-0.813015] | 0.557533 [0.531330-0.571738] | 4.44089e-16 |
| 1000 | MCAR | SimpleImputer[mean] | 1.024886 [0.923903-1.060029] | 0.746989 [0.715064-0.785709] | 2.59556 |
| 1000 | MCAR | SimpleImputer[median] | 1.027281 [0.923925-1.070808] | 0.729334 [0.703806-0.776482] | 2.73491 |
| 1000 | MCAR | KNNImputer | 0.867112 [0.728826-0.905334] | 0.568102 [0.530399-0.602091] | 0 |
| 1000 | MCAR | FaissImputer[complete] | 0.907623 [0.773968-0.963929] | 0.604941 [0.558917-0.652433] | 2.68907 |
| 1000 | MCAR | FaissImputer[available] | 0.867112 [0.728826-0.905334] | 0.568102 [0.530399-0.602091] | 4.44089e-16 |
| 3000 | MAR | SimpleImputer[mean] | 0.967392 [0.911834-0.970593] | 0.725830 [0.719046-0.749753] | 2.71597 |
| 3000 | MAR | SimpleImputer[median] | 0.961961 [0.901149-0.965115] | 0.699361 [0.697341-0.727643] | 2.87733 |
| 3000 | MAR | KNNImputer | 0.780936 [0.702004-0.793664] | 0.513611 [0.503584-0.547645] | 0 |
| 3000 | MAR | FaissImputer[complete] | 0.825936 [0.735690-0.826810] | 0.562784 [0.532444-0.570192] | 2.84218 |
| 3000 | MAR | FaissImputer[available] | 0.780936 [0.702004-0.793664] | 0.513611 [0.503584-0.547645] | 4.44089e-16 |
| 3000 | MCAR | SimpleImputer[mean] | 1.022821 [0.944901-1.060926] | 0.745411 [0.730752-0.794249] | 5.83818 |
| 3000 | MCAR | SimpleImputer[median] | 1.027525 [0.946257-1.073214] | 0.728204 [0.720241-0.784323] | 5.97778 |
| 3000 | MCAR | KNNImputer | 0.801336 [0.694397-0.851476] | 0.504908 [0.499191-0.554542] | 0 |
| 3000 | MCAR | FaissImputer[complete] | 0.863473 [0.738746-0.870314] | 0.560912 [0.526046-0.595083] | 4.03089 |
| 3000 | MCAR | FaissImputer[available] | 0.801336 [0.694397-0.851476] | 0.504908 [0.499191-0.554542] | 4.44089e-16 |

### Donors and actual missingness

Each row is one seed dataset, shared by both dtypes and all timing repeats. Complete mode restricts candidates to complete training rows. Available mode can use observed entries in incomplete rows; per-feature availability is training size minus missing_per_feature in the full-precision JSON.

| Train | Pattern | Seed | Complete donors | Train missing % | Query missing % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | MCAR | 101 | 322 | 10.0455 | 10.0455 |
| 1000 | MCAR | 202 | 296 | 10.3091 | 10.0273 |
| 1000 | MCAR | 303 | 297 | 10.5364 | 9.9364 |
| 1000 | MAR | 101 | 366 | 9.9455 | 10.7091 |
| 1000 | MAR | 202 | 344 | 10.1636 | 10.3455 |
| 1000 | MAR | 303 | 355 | 10.6091 | 10.1636 |
| 3000 | MCAR | 101 | 943 | 9.8030 | 10.0455 |
| 3000 | MCAR | 202 | 964 | 9.8939 | 10.0273 |
| 3000 | MCAR | 303 | 910 | 10.4333 | 9.9364 |
| 3000 | MAR | 101 | 1096 | 9.7818 | 10.7091 |
| 3000 | MAR | 202 | 1099 | 9.9273 | 10.3455 |
| 3000 | MAR | 303 | 1084 | 10.3545 | 10.1636 |

## Abalone (numerical features)

Source: [Nash et al. (1994). Abalone. https://doi.org/10.24432/C55C7W](https://archive.ics.uci.edu/dataset/1/abalone). License: CC BY 4.0.
Features: 7; excluded columns: Sex, Rings. Always observed: Length.
CPU model: **AMD EPYC 9V74 80-Core Processor**. [Actions run 36073113007](https://github.com/ScionKim/FaissImputer/actions/runs/36073113007).
[Raw JSON](../../benchmarks/results/real-data-datasets-ef04b1b/abalone.json); SHA-256: `4ce80d0aad87ddc9831a2131d752feff3278d563343ef643096c07e7f87fd64c`.

### Timing and memory: float32

Each time/RSS cell uses nine records; each KNN/Faiss cell uses nine matched pairs.

| Train | Pattern | Method | Total seconds, median [min-max] | KNN/Faiss | Peak RSS MiB, median [min-max] |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.001724 [0.001659-0.001903] | - | 146.62 [138.24-155.53] |
| 1000 | MAR | SimpleImputer[median] | 0.002044 [0.002012-0.002193] | - | 146.62 [138.36-155.53] |
| 1000 | MAR | KNNImputer | 0.020212 [0.019320-0.020919] | - | 147.98 [147.40-155.53] |
| 1000 | MAR | FaissImputer[complete] | 0.008279 [0.008108-0.008434] | 2.3966x | 146.77 [138.49-155.38] |
| 1000 | MAR | FaissImputer[available] | 0.018627 [0.017857-0.019888] | 1.0598x | 146.79 [146.20-155.38] |
| 1000 | MCAR | SimpleImputer[mean] | 0.001717 [0.001648-0.001845] | - | 144.77 [136.61-153.44] |
| 1000 | MCAR | SimpleImputer[median] | 0.002157 [0.001975-0.002301] | - | 144.77 [136.69-153.54] |
| 1000 | MCAR | KNNImputer | 0.020336 [0.019836-0.061209] | - | 148.38 [148.05-153.54] |
| 1000 | MCAR | FaissImputer[complete] | 0.007538 [0.007219-0.007913] | 2.6887x | 144.78 [137.72-153.29] |
| 1000 | MCAR | FaissImputer[available] | 0.018352 [0.017839-0.020427] | 1.1087x | 146.80 [146.66-153.44] |
| 3000 | MAR | SimpleImputer[mean] | 0.001962 [0.001849-0.002137] | - | 151.05 [142.12-160.32] |
| 3000 | MAR | SimpleImputer[median] | 0.002617 [0.002484-0.002800] | - | 151.05 [142.16-160.43] |
| 3000 | MAR | KNNImputer | 0.045861 [0.043494-0.047354] | - | 162.78 [162.28-164.00] |
| 3000 | MAR | FaissImputer[complete] | 0.011257 [0.011025-0.011816] | 4.0398x | 151.18 [142.37-160.17] |
| 3000 | MAR | FaissImputer[available] | 0.039187 [0.036750-0.044044] | 1.1494x | 154.93 [153.74-160.32] |
| 3000 | MCAR | SimpleImputer[mean] | 0.001903 [0.001839-0.002124] | - | 148.98 [140.48-156.85] |
| 3000 | MCAR | SimpleImputer[median] | 0.002558 [0.002501-0.002764] | - | 149.10 [140.55-156.90] |
| 3000 | MCAR | KNNImputer | 0.047393 [0.044726-0.049518] | - | 164.76 [164.17-165.01] |
| 3000 | MCAR | FaissImputer[complete] | 0.010236 [0.009952-0.010826] | 4.6055x | 149.24 [140.76-156.77] |
| 3000 | MCAR | FaissImputer[available] | 0.032871 [0.031808-0.034183] | 1.4269x | 154.99 [154.86-156.77] |

### Reconstruction quality and KNN output comparison: float32

RMSE/MAE use three seed observations. The final column is the maximum masked-cell output difference from KNN over those three seeds.

| Train | Pattern | Method | RMSE, median [min-max] | MAE, median [min-max] | Max output difference vs KNN |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 1.024186 [1.011513-1.112732] | 0.812019 [0.806228-0.878720] | 3.28628 |
| 1000 | MAR | SimpleImputer[median] | 1.056201 [1.018443-1.136425] | 0.834859 [0.804785-0.894761] | 3.34511 |
| 1000 | MAR | KNNImputer | 0.343608 [0.321129-0.381125] | 0.221159 [0.218137-0.229097] | 0 |
| 1000 | MAR | FaissImputer[complete] | 0.344080 [0.323592-0.371216] | 0.219993 [0.219448-0.224970] | 1.64517 |
| 1000 | MAR | FaissImputer[available] | 0.343323 [0.321141-0.380775] | 0.220860 [0.218212-0.229015] | 0.15887 |
| 1000 | MCAR | SimpleImputer[mean] | 0.961054 [0.959982-1.037995] | 0.773762 [0.770438-0.820342] | 3.54697 |
| 1000 | MCAR | SimpleImputer[median] | 0.971406 [0.965500-1.048390] | 0.773582 [0.768547-0.824114] | 3.45822 |
| 1000 | MCAR | KNNImputer | 0.308190 [0.283881-0.318781] | 0.183925 [0.181692-0.186941] | 0 |
| 1000 | MCAR | FaissImputer[complete] | 0.305248 [0.288107-0.327859] | 0.184682 [0.180389-0.186969] | 1.58731 |
| 1000 | MCAR | FaissImputer[available] | 0.308210 [0.283881-0.318781] | 0.183925 [0.181743-0.186941] | 0.0394038 |
| 3000 | MAR | SimpleImputer[mean] | 1.064268 [0.977922-1.094517] | 0.849371 [0.778448-0.866979] | 3.40204 |
| 3000 | MAR | SimpleImputer[median] | 1.097258 [1.003975-1.121179] | 0.874202 [0.797657-0.887180] | 3.53168 |
| 3000 | MAR | KNNImputer | 0.339817 [0.311046-0.353879] | 0.222797 [0.208009-0.226281] | 0 |
| 3000 | MAR | FaissImputer[complete] | 0.329381 [0.283377-0.332362] | 0.211427 [0.192857-0.213847] | 1.30664 |
| 3000 | MAR | FaissImputer[available] | 0.340084 [0.311418-0.353680] | 0.222147 [0.208310-0.226698] | 0.529608 |
| 3000 | MCAR | SimpleImputer[mean] | 0.993344 [0.931979-1.028329] | 0.799918 [0.750682-0.814425] | 3.24669 |
| 3000 | MCAR | SimpleImputer[median] | 1.004117 [0.937656-1.035482] | 0.803667 [0.746166-0.815661] | 3.24487 |
| 3000 | MCAR | KNNImputer | 0.274563 [0.258503-0.310641] | 0.177164 [0.176123-0.181760] | 0 |
| 3000 | MCAR | FaissImputer[complete] | 0.274509 [0.252064-0.326616] | 0.164758 [0.164215-0.184293] | 4.60586 |
| 3000 | MCAR | FaissImputer[available] | 0.274644 [0.258503-0.310641] | 0.177400 [0.176123-0.181760] | 0.106646 |

### Timing and memory: float64

Each time/RSS cell uses nine records; each KNN/Faiss cell uses nine matched pairs.

| Train | Pattern | Method | Total seconds, median [min-max] | KNN/Faiss | Peak RSS MiB, median [min-max] |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 0.001806 [0.001695-0.001932] | - | 147.57 [139.36-156.53] |
| 1000 | MAR | SimpleImputer[median] | 0.002085 [0.002026-0.002343] | - | 147.61 [139.50-156.53] |
| 1000 | MAR | KNNImputer | 0.018853 [0.018572-0.020645] | - | 148.41 [147.71-156.64] |
| 1000 | MAR | FaissImputer[complete] | 0.008545 [0.008376-0.008620] | 2.2365x | 147.82 [139.63-156.39] |
| 1000 | MAR | FaissImputer[available] | 0.042360 [0.041907-0.044592] | 0.4438x | 147.50 [146.83-156.39] |
| 1000 | MCAR | SimpleImputer[mean] | 0.001774 [0.001692-0.001893] | - | 145.91 [137.24-154.28] |
| 1000 | MCAR | SimpleImputer[median] | 0.002078 [0.001989-0.002138] | - | 145.91 [137.35-154.60] |
| 1000 | MCAR | KNNImputer | 0.019312 [0.018856-0.019708] | - | 148.79 [148.41-154.60] |
| 1000 | MCAR | FaissImputer[complete] | 0.007665 [0.007319-0.007991] | 2.5072x | 145.91 [137.78-154.28] |
| 1000 | MCAR | FaissImputer[available] | 0.045063 [0.043233-0.052132] | 0.4298x | 147.06 [146.59-154.28] |
| 3000 | MAR | SimpleImputer[mean] | 0.001955 [0.001830-0.002110] | - | 152.21 [143.22-160.71] |
| 3000 | MAR | SimpleImputer[median] | 0.002723 [0.002633-0.002927] | - | 152.21 [143.22-160.71] |
| 3000 | MAR | KNNImputer | 0.041793 [0.039553-0.042865] | - | 164.61 [164.28-165.95] |
| 3000 | MAR | FaissImputer[complete] | 0.011579 [0.011190-0.011705] | 3.6144x | 152.37 [143.34-160.71] |
| 3000 | MAR | FaissImputer[available] | 0.062640 [0.061160-0.070440] | 0.6554x | 155.75 [154.62-160.71] |
| 3000 | MCAR | SimpleImputer[mean] | 0.001946 [0.001870-0.002128] | - | 150.06 [141.66-159.36] |
| 3000 | MCAR | SimpleImputer[median] | 0.002733 [0.002596-0.002981] | - | 150.06 [141.66-159.36] |
| 3000 | MCAR | KNNImputer | 0.043944 [0.042548-0.045363] | - | 166.86 [166.04-166.95] |
| 3000 | MCAR | FaissImputer[complete] | 0.010677 [0.010253-0.011452] | 4.1935x | 150.18 [141.66-159.36] |
| 3000 | MCAR | FaissImputer[available] | 0.059146 [0.058851-0.065299] | 0.7275x | 155.49 [155.20-159.36] |

### Reconstruction quality and KNN output comparison: float64

RMSE/MAE use three seed observations. The final column is the maximum masked-cell output difference from KNN over those three seeds.

| Train | Pattern | Method | RMSE, median [min-max] | MAE, median [min-max] | Max output difference vs KNN |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | MAR | SimpleImputer[mean] | 1.024186 [1.011513-1.112732] | 0.812019 [0.806228-0.878720] | 3.28628 |
| 1000 | MAR | SimpleImputer[median] | 1.056201 [1.018443-1.136425] | 0.834859 [0.804785-0.894761] | 3.34511 |
| 1000 | MAR | KNNImputer | 0.343581 [0.321129-0.381125] | 0.221024 [0.218137-0.229097] | 0 |
| 1000 | MAR | FaissImputer[complete] | 0.344080 [0.323592-0.371216] | 0.219993 [0.219448-0.224970] | 1.64517 |
| 1000 | MAR | FaissImputer[available] | 0.343323 [0.321177-0.380775] | 0.220860 [0.218306-0.229015] | 0.182645 |
| 1000 | MCAR | SimpleImputer[mean] | 0.961054 [0.959982-1.037995] | 0.773762 [0.770438-0.820342] | 3.54697 |
| 1000 | MCAR | SimpleImputer[median] | 0.971406 [0.965500-1.048390] | 0.773582 [0.768547-0.824114] | 3.45822 |
| 1000 | MCAR | KNNImputer | 0.308210 [0.283881-0.318781] | 0.183925 [0.181743-0.186941] | 0 |
| 1000 | MCAR | FaissImputer[complete] | 0.305248 [0.288107-0.327859] | 0.184682 [0.180389-0.186969] | 1.58731 |
| 1000 | MCAR | FaissImputer[available] | 0.308210 [0.283881-0.318781] | 0.183925 [0.181743-0.186941] | 4.44089e-16 |
| 3000 | MAR | SimpleImputer[mean] | 1.064268 [0.977922-1.094517] | 0.849371 [0.778448-0.866979] | 3.40204 |
| 3000 | MAR | SimpleImputer[median] | 1.097258 [1.003975-1.121179] | 0.874202 [0.797657-0.887180] | 3.53168 |
| 3000 | MAR | KNNImputer | 0.340366 [0.311132-0.351832] | 0.222191 [0.207974-0.226971] | 0 |
| 3000 | MAR | FaissImputer[complete] | 0.329381 [0.283377-0.332362] | 0.211427 [0.192857-0.213847] | 1.30664 |
| 3000 | MAR | FaissImputer[available] | 0.340084 [0.311221-0.353438] | 0.221920 [0.208035-0.226698] | 0.372807 |
| 3000 | MCAR | SimpleImputer[mean] | 0.993344 [0.931979-1.028329] | 0.799918 [0.750682-0.814425] | 3.24669 |
| 3000 | MCAR | SimpleImputer[median] | 1.004117 [0.937656-1.035482] | 0.803667 [0.746166-0.815661] | 3.24487 |
| 3000 | MCAR | KNNImputer | 0.274644 [0.258503-0.310641] | 0.177400 [0.176123-0.181760] | 0 |
| 3000 | MCAR | FaissImputer[complete] | 0.274509 [0.252064-0.326616] | 0.164757 [0.164215-0.184293] | 4.60586 |
| 3000 | MCAR | FaissImputer[available] | 0.274563 [0.258503-0.310641] | 0.177164 [0.176123-0.181760] | 0.106646 |

### Donors and actual missingness

Each row is one seed dataset, shared by both dtypes and all timing repeats. Complete mode restricts candidates to complete training rows. Available mode can use observed entries in incomplete rows; per-feature availability is training size minus missing_per_feature in the full-precision JSON.

| Train | Pattern | Seed | Complete donors | Train missing % | Query missing % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | MCAR | 101 | 482 | 10.0571 | 10.1571 |
| 1000 | MCAR | 202 | 458 | 10.4429 | 9.9000 |
| 1000 | MCAR | 303 | 449 | 10.3000 | 10.1143 |
| 1000 | MAR | 101 | 505 | 9.9429 | 10.3286 |
| 1000 | MAR | 202 | 484 | 10.3143 | 9.9143 |
| 1000 | MAR | 303 | 484 | 10.5571 | 9.5857 |
| 3000 | MCAR | 101 | 1444 | 9.8048 | 10.1571 |
| 3000 | MCAR | 202 | 1419 | 10.1429 | 9.9000 |
| 3000 | MCAR | 303 | 1383 | 10.3286 | 10.1143 |
| 3000 | MAR | 101 | 1536 | 9.8048 | 10.3286 |
| 3000 | MAR | 202 | 1512 | 10.0714 | 9.9143 |
| 3000 | MAR | 303 | 1490 | 10.2905 | 9.5857 |
