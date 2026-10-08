# Abalone: released 0.3.22 and 0.3.21 — float32 and float64, MAR, uniform weights

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Abalone numerical data. Float32 and float64 use separate records and tables. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. No same-data API or complete-donor measurements are included.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_abalone_mar_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_abalone_mar_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37679710940/attempts/1).
- Benchmark source commit: `99cdc057d844f5dec7c4a98da9c5a719d71ccafe`. Measured packages were installed outside the checkout.
- Archive SHA-256: `8b1c14831c25d53a3a7e9756761ad71553e931b244cdf5ea8ad1bc5e44de3924`.
- `version_comparison_abalone_float32_mar.json` SHA-256: `9b95acde7b68f5860bab1c3f48f0fb5d2cc1785bd770d587f69df1c695e2b4b9`.
- `version_comparison_abalone_float64_mar.json` SHA-256: `d69abc34dae2c384125bf5a68b801b5552684b965acdea9f8fb8c8cf375761cd`.
- Runner: AMD EPYC 9V74 80-Core Processor; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.15; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 7 numerical features; k=5; uniform weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
- 10% target overall MAR missingness in training and query inputs. `Length` stays observed; the other features are eligible for masking.
- 54 successful workers: 3 seeds × 3 repeats × 3 methods × 2 dtypes. Variants rotate within each seed/repeat. Float32 and float64 run sequentially as separate invocations on the same runner.

The environment freezes differ only in the FaissImputer release. KNNImputer uses the 0.3.22 environment. Source rows, masks, scaler parameters and float64 scoring truth match across dtypes. Within each dtype, prepared cases match across variants and repeats. Scaling uses observed training values only.

### MAR missingness by seed

The benchmark sets the cutoff to the median raw `Length` value in the first 1,000 training rows, before scaling. Query values do not determine it. For each eligible feature, rows at or below the cutoff use half the base masking probability; rows above it use 1.5 times the base probability. The driver stays observed. Realized missingness rates are reported with donor counts below.

| Seed | Reference training rows | Driver cutoff (source units) | Base probability (%) | At/below cutoff (%) | Above cutoff (%) |
| --- | --- | --- | --- | --- | --- |
| 101 | 1000 | 0.54 | 11.6667 | 5.8333 | 17.5000 |
| 202 | 1000 | 0.545 | 11.6667 | 5.8333 | 17.5000 |
| 303 | 1000 | 0.55 | 11.6667 | 5.8333 | 17.5000 |

This analysis validates the saved MAR metadata and case consistency. It does not regenerate masks or recompute the cutoff from source rows.

## Aggregation methodology

All timing and memory cells are **median [min–max] across nine records per method and dtype: three seeds × three repeats**. `total_seconds` is fit plus first transform. `transform_seconds` is the first transform alone. Fit and transform are consecutive, with no explicit garbage collection or RSS sampling between them. Preparation, warmup, worker startup, validation and serialization are outside the timed interval; no additional transforms were measured.

Ratios are **numerator time / denominator time for each matched record**, then median [min–max] across nine pairs. Matching includes the archived run, dataset, training/query sizes, features, neighbors, donor policy, validated metric/weights, missingness, dtype, API, threads, seed, repeat and prepared inputs. **Ratios are not ratios of displayed median times.** Values above one favor the denominator. Dtypes are never pooled.

Duration change is `100 * (denominator time / numerator time - 1)` for each pair, summarized by median [min–max]. Negative values mean less time. These observed ranges are not confidence intervals.

RMSE and MAE summarize reconstruction error against held-out ground truth in standardized units. Quality uses three seed datasets, taking repeat 1 after verifying consistency across repeats. Similar aggregate errors do not establish identical predictions or algorithmic equivalence. The analyzer aggregates stored worker metrics; it does not regenerate truth arrays or recompute the underlying ground-truth errors.

Output differences are recomputed from saved `imputed_values`. Full-output hashes are recorded worker hashes. Seed-level difference counts use repeat 1, so timing repetitions do not multiply affected entries. The `1e-5` threshold is descriptive, not an equivalence test. The full-precision JSON retains original seconds, samples, pair indices, numerators, denominators, ratios, percentage changes, feature errors and fingerprints; only Markdown is rounded.

## float32

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.721 [0.650–0.783] | 44.199 [42.817–46.103] | 44.982 [43.468–46.817] |
| FaissImputer 0.3.21 | 1.212 [1.131–1.243] | 37.738 [34.533–41.652] | 38.950 [35.738–42.840] |
| FaissImputer 0.3.22 | 1.262 [1.219–1.279] | 37.134 [35.288–41.906] | 38.398 [36.550–43.130] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 0.9656 [0.8843–0.9945] | 0.9978 [0.9783–1.0526] | 0.9974 [0.9773–1.0496] |
| KNNImputer / FaissImputer 0.3.22 | 0.5725 [0.5128–0.6127] | 1.1729 [1.0229–1.2871] | 1.1527 [1.0083–1.2624] |
| KNNImputer / FaissImputer 0.3.21 | 0.5814 [0.5394–0.6372] | 1.1783 [1.0283–1.2753] | 1.1610 [1.0152–1.2519] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | 3.5658 [0.5573–13.0865] | 0/9 |
| First transform | 0.2215 [-4.9966–2.2196] | 4/9 |
| Fit + first transform | 0.2630 [-4.7240–2.3198] | 4/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 162.797 [162.559–164.066] |
| FaissImputer 0.3.21 | 154.215 [153.680–155.102] |
| FaissImputer 0.3.22 | 153.984 [153.781–154.957] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.3398170325 [0.3110463685–0.3538790786] | 0.2227969540 [0.2080085813–0.2262806054] |
| FaissImputer 0.3.21 | 0.3400843186 [0.3114176616–0.3536796673] | 0.2221468531 [0.2083098577–0.2266981474] |
| FaissImputer 0.3.22 | 0.3400843186 [0.3114176616–0.3536796673] | 0.2221468531 [0.2083098577–0.2266981474] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.529607668519 | 0.000371293103344 | 0.000650100822903 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 723 | 130 | 10 | 0.278636157513 | +0.000267286117763 | +0.000417541984329 |
| 202 | 694 | 159 | 9 | 0.116158291698 | +0.000371293103344 | +0.000301276358417 |
| 303 | 671 | 134 | 7 | 0.529607668519 | -0.000199411315853 | -0.000650100822903 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 0.99738×, with 4/9 pairs favoring the current release. Median paired total-duration change: +0.2630%.
- KNN/0.3.22 median paired total-time ratio: 1.15273×, with 9/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## float64

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.698 [0.665–0.747] | 40.031 [38.313–42.529] | 40.763 [39.035–43.257] |
| FaissImputer 0.3.21 | 1.370 [1.324–1.389] | 60.271 [58.644–68.030] | 61.619 [60.014–69.410] |
| FaissImputer 0.3.22 | 1.335 [1.286–1.388] | 40.128 [37.570–47.365] | 41.452 [38.891–48.753] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0200 [0.9662–1.0651] | 1.4983 [1.4278–1.5940] | 1.4839 [1.4152–1.5719] |
| KNNImputer / FaissImputer 0.3.22 | 0.5250 [0.4917–0.5693] | 0.9918 [0.8979–1.0433] | 0.9759 [0.8873–1.0239] |
| KNNImputer / FaissImputer 0.3.21 | 0.5089 [0.4794–0.5438] | 0.6482 [0.6167–0.6826] | 0.6452 [0.6145–0.6792] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -1.9642 [-6.1098–3.4984] | 7/9 |
| First transform | -33.2588 [-37.2634–-29.9598] | 9/9 |
| Fit + first transform | -32.6122 [-36.3834–-29.3362] | 9/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 164.559 [164.340–165.852] |
| FaissImputer 0.3.21 | 154.609 [154.492–155.949] |
| FaissImputer 0.3.22 | 154.812 [154.598–155.727] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.3403656704 [0.3111316758–0.3518318477] | 0.2221907184 [0.2079739768–0.2269711044] |
| FaissImputer 0.3.21 | 0.3400843198 [0.3112210596–0.3534383765] | 0.2219198322 [0.2080348718–0.2266981467] |
| FaissImputer 0.3.22 | 0.3400843198 [0.3112210596–0.3534383765] | 0.2219198322 [0.2080348718–0.2266981467] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.372807062814 | 0.00160652882545 | 0.000272957643491 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 723 | 124 | 8 | 0.218632140596 | -0.000281350588284 | -0.000272957643491 |
| 202 | 694 | 133 | 7 | 0.208238461086 | +8.9383804844e-05 | +6.08949482455e-05 |
| 303 | 671 | 128 | 9 | 0.372807062814 | +0.00160652882545 | -0.000270886210404 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.48395×, with 9/9 pairs favoring the current release. Median paired total-duration change: -32.6122%.
- KNN/0.3.22 median paired total-time ratio: 0.97588×, with 3/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## Donor counts and observed missingness

These counts are shared across dtypes; the analyzer verifies identical source rows, masks and scaler metadata. Complete donors are fully observed training rows. Available mode also uses partially observed rows separately for each missing feature.

| Seed | Complete donors | Training missing (%) | Query missing (%) | Queries with missing values | Scored cells |
| --- | --- | --- | --- | --- | --- |
| 101 | 1536 | 9.8048 | 10.3286 | 515 | 723 |
| 202 | 1512 | 10.0714 | 9.9143 | 485 | 694 |
| 303 | 1490 | 10.2905 | 9.5857 | 488 | 671 |

### Training rows observed for each feature

Counts are training rows minus missing entries in each feature. They do not guarantee a defined distance to every query. Per-feature reconstruction errors in standardized and source units are preserved separately for each dtype in the full-precision summary.

| Feature | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- |
| Length | 3000 | 3000 | 3000 |
| Diameter | 2655 | 2649 | 2620 |
| Height | 2663 | 2638 | 2647 |
| Whole_weight | 2633 | 2645 | 2626 |
| Shucked_weight | 2676 | 2674 | 2647 |
| Viscera_weight | 2632 | 2632 | 2639 |
| Shell_weight | 2682 | 2647 | 2660 |

## Limits

These are descriptive measurements of one configuration on one runner. No statistical significance, universal speedup or all-input output equivalence is established. Output differences alone do not identify their numerical or neighbor-selection cause. Earlier diagnostics on other seeds or missingness mechanisms do not establish the cause here.

The Abalone MAR uniform and distance archives were measured in separate runs on different CPU models. They are not a hardware-controlled weights comparison. Earlier MCAR results are separate experiments; cross-run timings do not isolate a missingness or weights effect. MCAR output diagnostics do not establish the cause of MAR output differences. These reports keep each run separate.

## Dataset provenance

[Abalone (numerical features)](https://archive.ics.uci.edu/dataset/1/abalone). Nash et al. (1994). Abalone. https://doi.org/10.24432/C55C7W

The source file `abalone.data` contains 4,177 rows. Excluded columns: Sex, Rings. The original source ZIP is preserved; the recorded dataset license is CC BY 4.0. Target values are not used for neighbor search or downstream scoring.

- Source ZIP SHA-256: `755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4`.
- Source data SHA-256: `de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f`.
- Parsed numerical-array fingerprint: `6d71332f0d6a22320c7eff7b18909fab5565975da5a68e7ceef0b49f4572e1a4`.

## Reproduction

The standard-library analysis script validates the archive and source hashes, complete per-dtype worker grids, matching inputs, dependency freezes, repeated outputs and all stored aggregates. It recomputes every displayed statistic from the saved records without installing or running imputers.

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the corresponding matrix entry uses `--dataset abalone --weights uniform --mechanism MAR`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-abalone-mar-0.3.22`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require both files; a partially present pair fails. This is saved-data analysis, not a new benchmark.
