# Wine Quality White: released 0.3.22 and 0.3.21 — float64, MAR, distance weights

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Wine Quality White data with `weights="distance"`. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. Each dtype has its own records and tables.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_wine_quality_distance_mar_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_wine_quality_distance_mar_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37572753465/attempts/1).
- Benchmark source commit: `949e3ff6478fba8273e20fb77a72eef60d00e2a7`. Measured packages were installed outside the checkout.
- Archive SHA-256: `9ff8a529b3afb7129e7a78dc8d47e07aa8d42b1910c7431591436c5321fe8a8f`.
- `version_comparison_wine_quality_white_distance_mar.json` SHA-256: `7544ca8ab3247334f8678799f4047afc82dfd8cea3a394e097c00d4ab160e3ef`.
- Runner: AMD EPYC 7763 64-Core Processor; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 11 numerical features; k=5; distance weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
- 10% target overall MAR missingness in training and query inputs. `alcohol` stays observed; the other features are eligible for masking.
- 27 successful workers: 3 seeds × 3 repeats × 3 methods × 1 dtype. Variants rotate within each seed/repeat. Only float64 is measured in this run.

The environment freezes differ only in the FaissImputer release. KNNImputer uses the 0.3.22 environment. Prepared cases, source rows, masks, scaler parameters and float64 scoring truth match across variants and repeats. Scaling uses observed training values only.

### MAR missingness by seed

The benchmark sets the cutoff to the median raw `alcohol` value in the first 1,000 training rows, before scaling. Query values do not determine it. For each eligible feature, rows at or below the cutoff use half the base masking probability; rows above it use 1.5 times the base probability. The driver stays observed. Realized missingness rates are reported with donor counts below.

| Seed | Reference training rows | Driver cutoff (source units) | Base probability (%) | At/below cutoff (%) | Above cutoff (%) |
| --- | --- | --- | --- | --- | --- |
| 101 | 1000 | 10.35 | 11.0000 | 5.5000 | 16.5000 |
| 202 | 1000 | 10.3 | 11.0000 | 5.5000 | 16.5000 |
| 303 | 1000 | 10.3 | 11.0000 | 5.5000 | 16.5000 |

This analysis validates the saved MAR metadata and case consistency. It does not regenerate masks or recompute the cutoff from source rows.

## Aggregation methodology

All timing and memory cells are **median [min–max] across nine records per method and dtype: three seeds × three repeats**. `total_seconds` is fit plus first transform. `transform_seconds` is the first transform alone. Fit and transform are consecutive, with no explicit garbage collection or RSS sampling between them. Preparation, warmup, worker startup, validation and serialization are outside the timed interval; no additional transforms were measured.

Ratios are **numerator time / denominator time for each matched record**, then median [min–max] across nine pairs. Matching includes the archived run, dataset, training/query sizes, features, neighbors, donor policy, validated metric/weights, missingness, dtype, API, threads, seed, repeat and prepared inputs. **Ratios are not ratios of displayed median times.** Values above one favor the denominator. Dtypes are never pooled.

Duration change is `100 * (denominator time / numerator time - 1)` for each pair, summarized by median [min–max]. Negative values mean less time. These observed ranges are not confidence intervals.

RMSE and MAE summarize reconstruction error against held-out ground truth in standardized units. Quality uses three seed datasets, taking repeat 1 after verifying consistency across repeats. Similar aggregate errors do not establish identical predictions or algorithmic equivalence. The analyzer aggregates stored worker metrics; it does not regenerate truth arrays or recompute the underlying ground-truth errors.

Output differences are recomputed from saved `imputed_values`. Full-output hashes are recorded worker hashes. Seed-level difference counts use repeat 1, so timing repetitions do not multiply affected entries. The `1e-5` threshold is descriptive, not an equivalence test. The full-precision JSON retains original seconds, samples, pair indices, numerators, denominators, ratios, percentage changes, feature errors and fingerprints; only Markdown is rounded.

## float64

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.612 [0.561–0.679] | 61.375 [58.756–63.058] | 61.981 [59.317–63.655] |
| FaissImputer 0.3.21 | 1.456 [1.415–1.494] | 181.477 [178.923–188.776] | 182.893 [180.344–190.240] |
| FaissImputer 0.3.22 | 1.424 [1.363–1.460] | 156.338 [154.979–160.084] | 157.752 [156.401–161.447] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0175 [0.9941–1.0681] | 1.1605 [1.1545–1.1800] | 1.1594 [1.1531–1.1786] |
| KNNImputer / FaissImputer 0.3.22 | 0.4297 [0.3949–0.4803] | 0.3914 [0.3791–0.4045] | 0.3921 [0.3793–0.4047] |
| KNNImputer / FaissImputer 0.3.21 | 0.4321 [0.3953–0.4634] | 0.3354 [0.3284–0.3500] | 0.3362 [0.3289–0.3505] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -1.7247 [-6.3752–0.5956] | 6/9 |
| First transform | -13.8272 [-15.2578–-13.3825] | 9/9 |
| Fit + first transform | -13.7461 [-15.1536–-13.2763] | 9/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 172.863 [172.305–174.074] |
| FaissImputer 0.3.21 | 157.551 [157.355–157.750] |
| FaissImputer 0.3.22 | 157.465 [157.363–157.668] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.7398262794 [0.6394005462–0.7493364640] | 0.4260925552 [0.4059979548–0.4676263856] |
| FaissImputer 0.3.21 | 0.7397523610 [0.6403205154–0.7493364640] | 0.4260925509 [0.4069146913–0.4674705588] |
| FaissImputer 0.3.22 | 0.7397523610 [0.6403205154–0.7493364640] | 0.4260925509 [0.4069146913–0.4674705588] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.307759628858 | 0.000919969198384 | 0.000916736537962 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 1178 | 929 | 8 | 0.307759628858 | +0.000919969198384 | +0.000916736537962 |
| 202 | 1138 | 911 | 5 | 0.0487435450699 | -7.39184533606e-05 | -0.000155826833398 |
| 303 | 1118 | 847 | 0 | 3.50474007149e-07 | -5.55111512313e-16 | -4.32409225093e-09 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.15937×, with 9/9 pairs favoring the current release. Median paired total-duration change: -13.7461%.
- KNN/0.3.22 median paired total-time ratio: 0.39208×, with 0/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## Donor counts and observed missingness

One observation per seed after input consistency checks across variants and repeats. Complete donors are fully observed training rows. Available mode also uses partially observed rows separately for each missing feature.

| Seed | Complete donors | Training missing (%) | Query missing (%) | Queries with missing values | Scored cells |
| --- | --- | --- | --- | --- | --- |
| 101 | 1096 | 9.7818 | 10.7091 | 660 | 1178 |
| 202 | 1099 | 9.9273 | 10.3455 | 643 | 1138 |
| 303 | 1084 | 10.3545 | 10.1636 | 628 | 1118 |

### Training rows observed for each feature

Counts are training rows minus missing entries in each feature. They do not guarantee a defined distance to every query. Per-feature reconstruction errors in standardized and source units are preserved separately for each dtype in the full-precision summary.

| Feature | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- |
| fixed acidity | 2682 | 2700 | 2657 |
| volatile acidity | 2690 | 2672 | 2674 |
| citric acid | 2657 | 2637 | 2613 |
| residual sugar | 2694 | 2674 | 2660 |
| chlorides | 2665 | 2643 | 2686 |
| free sulfur dioxide | 2701 | 2667 | 2659 |
| total sulfur dioxide | 2653 | 2688 | 2657 |
| density | 2681 | 2708 | 2645 |
| pH | 2676 | 2663 | 2660 |
| sulphates | 2673 | 2672 | 2672 |
| alcohol | 3000 | 3000 | 3000 |

## Limits

These are descriptive measurements of one configuration on one runner. No statistical significance, universal speedup or all-input output equivalence is established. Output differences alone do not identify their numerical or neighbor-selection cause. Earlier diagnostics on other seeds or missingness mechanisms do not establish the cause here.

The MAR uniform and distance archives were measured in separate runs. A matching CPU model does not make them a single controlled weights comparison. Earlier MCAR results are separate experiments; cross-run timings do not isolate a missingness or weights effect. MCAR output diagnostics do not establish the cause of MAR output differences. These reports keep each run separate.

## Dataset provenance

[Wine Quality (white)](https://archive.ics.uci.edu/dataset/186/wine+quality). Cortez et al. (2009). Wine Quality. https://doi.org/10.24432/C56S3T

The source file `winequality-white.csv` contains 4,898 rows. Excluded columns: quality. The original source ZIP is preserved; the recorded dataset license is CC BY 4.0. Target values are not used for neighbor search or downstream scoring.

- Source ZIP SHA-256: `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`.
- Source data SHA-256: `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836`.
- Parsed numerical-array fingerprint: `518c625f745e3807da855507e3b47c6b3d9dd499699b55dd0c5a88105d96674c`.

## Reproduction

The standard-library analysis script validates the archive and source hashes, complete per-dtype worker grids, matching inputs, dependency freezes, repeated outputs and all stored aggregates. It recomputes every displayed statistic from the saved records without installing or running imputers.

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the corresponding matrix entry uses `--dataset wine_quality_white --dtype float64 --weights distance --mechanism MAR`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-wine-quality-mar-0.3.22`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require both files; a partially present pair fails. This is saved-data analysis, not a new benchmark.
