# Abalone: released 0.3.22 and 0.3.21 — float32 and float64, MAR, distance weights

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Abalone data with `weights="distance"`. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. Each dtype has its own records and tables.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_abalone_distance_mar_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_abalone_distance_mar_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37680429397/attempts/1).
- Benchmark source commit: `99cdc057d844f5dec7c4a98da9c5a719d71ccafe`. Measured packages were installed outside the checkout.
- Archive SHA-256: `52ccd209302777d1c16dcfe159ae541c34c793f37d41fc035ad3e720fcf05695`.
- `version_comparison_abalone_float32_distance_mar.json` SHA-256: `2541eff5f3d7fe147194947e72ad8a5ad711028b0e514b631a21aa243d87f41d`.
- `version_comparison_abalone_float64_distance_mar.json` SHA-256: `031772e7e0a44f086bd46db3240a4f8b164a2789d6c3c2a20d86b2bd5f3d2fd4`.
- Runner: INTEL(R) XEON(R) PLATINUM 8573C; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.15; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 7 numerical features; k=5; distance weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
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
| KNNImputer | 0.821 [0.522–1.064] | 50.898 [44.968–120.877] | 51.867 [45.686–121.698] |
| FaissImputer 0.3.21 | 1.285 [1.084–1.422] | 47.820 [40.545–53.045] | 49.139 [41.629–54.330] |
| FaissImputer 0.3.22 | 1.232 [1.123–1.358] | 45.245 [41.233–50.681] | 46.526 [42.429–52.040] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0588 [0.9066–1.2228] | 1.0691 [0.9260–1.1055] | 1.0689 [0.9258–1.1082] |
| KNNImputer / FaissImputer 0.3.22 | 0.6459 [0.3922–0.8893] | 1.1524 [0.9926–2.3851] | 1.1416 [0.9757–2.3386] |
| KNNImputer / FaissImputer 0.3.21 | 0.6386 [0.3674–0.9809] | 1.0779 [0.9106–2.2788] | 1.0680 [0.8956–2.2400] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -5.5548 [-18.2207–10.3006] | 6/9 |
| First transform | -6.4603 [-9.5430–7.9865] | 7/9 |
| Fit + first transform | -6.4422 [-9.7669–8.0112] | 7/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 163.055 [162.762–164.262] |
| FaissImputer 0.3.21 | 154.523 [153.930–155.684] |
| FaissImputer 0.3.22 | 154.500 [153.984–155.535] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.3566299297 [0.3307440518–0.3589793974] | 0.2272482920 [0.2170209639–0.2353910092] |
| FaissImputer 0.3.21 | 0.3551326412 [0.3313509496–0.3616681492] | 0.2276317982 [0.2172476420–0.2350590466] |
| FaissImputer 0.3.22 | 0.3551326412 [0.3313509496–0.3616681492] | 0.2276317982 [0.2172476420–0.2350590466] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.457599282265 | 0.00268875175675 | 0.000383506195528 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 723 | 577 | 54 | 0.457599282265 | -0.00149728848091 | -0.000331962608677 |
| 202 | 694 | 551 | 37 | 0.289220035076 | +0.000606897739501 | +0.000226678113689 |
| 303 | 671 | 546 | 37 | 0.360154181719 | +0.00268875175675 | +0.000383506195528 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.06886×, with 7/9 pairs favoring the current release. Median paired total-duration change: -6.4422%.
- KNN/0.3.22 median paired total-time ratio: 1.14155×, with 6/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## float64

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.723 [0.613–0.907] | 44.189 [41.693–52.295] | 44.822 [42.319–53.202] |
| FaissImputer 0.3.21 | 1.306 [1.142–1.440] | 65.245 [60.031–72.233] | 66.549 [61.270–73.566] |
| FaissImputer 0.3.22 | 1.353 [1.238–1.409] | 47.677 [43.096–55.466] | 49.030 [44.334–56.847] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0008 [0.8107–1.0639] | 1.3748 [1.2487–1.4660] | 1.3661 [1.2376–1.4549] |
| KNNImputer / FaissImputer 0.3.22 | 0.5234 [0.4559–0.7233] | 0.9074 [0.8195–1.1019] | 0.8979 [0.8123–1.0921] |
| KNNImputer / FaissImputer 0.3.21 | 0.5567 [0.4745–0.6957] | 0.7022 [0.6118–0.8015] | 0.7005 [0.6093–0.7994] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -0.0764 [-6.0039–23.3536] | 5/9 |
| First transform | -27.2600 [-31.7876–-19.9189] | 9/9 |
| Fit + first transform | -26.8007 [-31.2673–-19.1997] | 9/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 164.586 [164.371–165.922] |
| FaissImputer 0.3.21 | 155.031 [154.715–155.918] |
| FaissImputer 0.3.22 | 155.117 [154.867–156.410] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.3553378194 [0.3358917360–0.3624505307] | 0.2290302720 [0.2185757926–0.2353842404] |
| FaissImputer 0.3.21 | 0.3551326133 [0.3313509574–0.3615428404] | 0.2275085245 [0.2172476432–0.2350590427] |
| FaissImputer 0.3.22 | 0.3551326133 [0.3313509574–0.3615428404] | 0.2275085245 [0.2172476432–0.2350590427] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.404908118778 | 0.00454077858857 | 0.00152174746288 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 723 | 655 | 11 | 0.156274882784 | -0.000205206105601 | -0.000325197738141 |
| 202 | 694 | 613 | 14 | 0.404908118778 | -0.00454077858857 | -0.00132814947694 |
| 303 | 671 | 603 | 15 | 0.403510601853 | -0.000907690323088 | -0.00152174746288 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.36613×, with 9/9 pairs favoring the current release. Median paired total-duration change: -26.8007%.
- KNN/0.3.22 median paired total-time ratio: 0.89794×, with 2/9 pairs favoring the current release. Fit alone is reported separately above.
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

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the corresponding matrix entry uses `--dataset abalone --weights distance --mechanism MAR`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-abalone-mar-0.3.22`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require both files; a partially present pair fails. This is saved-data analysis, not a new benchmark.
