# Abalone: released 0.3.22 and 0.3.21 — distance weights

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Abalone data with `weights="distance"`. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. Each dtype has its own records and tables.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_abalone_distance_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_abalone_distance_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37241332456/attempts/1).
- Benchmark source commit: `cd8117708f0bb43a2f79e15adf1e2e543ec6cda3`. Measured packages were installed outside the checkout.
- Archive SHA-256: `c9239069766885e67b3c11c8f56d4a6268f9d1c9b1d27c431ea56d35f6dca3c0`.
- `version_comparison_abalone_float32_distance.json` SHA-256: `41297a57f6d7a101527f731efb9c6571f90b706da810e5ec3f2d42af590959d5`.
- `version_comparison_abalone_float64_distance.json` SHA-256: `c908b062021185483b6f72551d264c96c77290a97bebed93144bbda7c8f87a7f`.
- Runner: AMD EPYC 7763 64-Core Processor; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 7 numerical features; k=5; distance weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
- 10% target overall MCAR missingness in training and query inputs. `Length` stays observed; the other features are eligible for masking.
- 54 successful workers: 3 seeds × 3 repeats × 3 methods × 2 dtypes. Variants rotate within each seed/repeat. Float32 and float64 run sequentially as separate invocations on the same runner.

The environment freezes differ only in the FaissImputer release. KNNImputer uses the 0.3.22 environment. Source rows, masks, scaler parameters and float64 scoring truth match across dtypes. Within each dtype, prepared cases match across variants and repeats. Scaling uses observed training values only.

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
| KNNImputer | 0.611 [0.541–0.666] | 44.644 [42.265–45.581] | 45.185 [42.931–46.191] |
| FaissImputer 0.3.21 | 1.157 [1.101–1.207] | 33.114 [32.498–34.455] | 34.232 [33.601–35.605] |
| FaissImputer 0.3.22 | 1.151 [1.100–1.206] | 33.021 [32.323–34.134] | 34.141 [33.474–35.296] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0012 [0.9426–1.0937] | 1.0028 [0.9812–1.0300] | 1.0022 [0.9819–1.0292] |
| KNNImputer / FaissImputer 0.3.22 | 0.5298 [0.4704–0.5990] | 1.3404 [1.2926–1.3854] | 1.3121 [1.2698–1.3579] |
| KNNImputer / FaissImputer 0.3.21 | 0.5314 [0.4672–0.5696] | 1.3382 [1.2615–1.3650] | 1.3115 [1.2381–1.3344] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -0.1220 [-8.5688–6.0946] | 5/9 |
| First transform | -0.2815 [-2.9136–1.9179] | 6/9 |
| Fit + first transform | -0.2192 [-2.8384–1.8454] | 5/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 164.750 [164.316–164.910] |
| FaissImputer 0.3.21 | 155.004 [154.770–155.062] |
| FaissImputer 0.3.22 | 155.035 [154.629–155.086] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.2754324098 [0.2625402355–0.3139563012] | 0.1794806585 [0.1784881503–0.1868146455] |
| FaissImputer 0.3.21 | 0.2755078430 [0.2625663859–0.3140038889] | 0.1796286521 [0.1785040755–0.1868538436] |
| FaissImputer 0.3.22 | 0.2755078430 [0.2625663859–0.3140038889] | 0.1796286521 [0.1785040755–0.1868538436] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.0799499750137 | 7.54332001783e-05 | 0.000147993608742 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 711 | 552 | 21 | 0.0799499750137 | +7.54332001783e-05 | +0.000147993608742 |
| 202 | 693 | 535 | 17 | 0.00844883918762 | +2.6150359599e-05 | +1.5925266612e-05 |
| 303 | 708 | 556 | 6 | 0.0202132463455 | +4.75876853023e-05 | +3.91981056707e-05 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.00220×, with 5/9 pairs favoring the current release. Median paired total-duration change: -0.2192%.
- KNN/0.3.22 median paired total-time ratio: 1.31213×, with 9/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## float64

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.633 [0.568–0.702] | 40.659 [39.781–42.799] | 41.284 [40.391–43.501] |
| FaissImputer 0.3.21 | 1.284 [1.203–1.300] | 67.372 [66.153–69.782] | 68.611 [67.433–71.071] |
| FaissImputer 0.3.22 | 1.268 [1.216–1.320] | 37.066 [35.967–37.762] | 38.367 [37.205–39.083] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 0.9951 [0.9246–1.0626] | 1.8235 [1.7870–1.9069] | 1.7992 [1.7597–1.8754] |
| KNNImputer / FaissImputer 0.3.22 | 0.5043 [0.4325–0.5437] | 1.1058 [1.0535–1.1833] | 1.0869 [1.0335–1.1613] |
| KNNImputer / FaissImputer 0.3.21 | 0.4990 [0.4409–0.5467] | 0.6036 [0.5895–0.6206] | 0.6017 [0.5873–0.6192] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | 0.4907 [-5.8878–8.1511] | 3/9 |
| First transform | -45.1598 [-47.5583–-44.0397] | 9/9 |
| Fit + first transform | -44.4193 [-46.6788–-43.1722] | 9/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 166.820 [166.031–166.973] |
| FaissImputer 0.3.21 | 155.391 [155.289–155.562] |
| FaissImputer 0.3.22 | 155.418 [155.277–155.539] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.2751931977 [0.2625663822–0.3140038854] | 0.1795368390 [0.1785040733–0.1868538424] |
| FaissImputer 0.3.21 | 0.2754407044 [0.2625663830–0.3140038887] | 0.1794819672 [0.1785040744–0.1868538449] |
| FaissImputer 0.3.22 | 0.2754407044 [0.2625663830–0.3140038887] | 0.1794819672 [0.1785040744–0.1868538449] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.15553150591 | 0.000247506619977 | 5.48718226448e-05 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 711 | 655 | 7 | 0.15553150591 | +0.000247506619977 | -5.48718226448e-05 |
| 202 | 693 | 619 | 0 | 4.11225168515e-07 | +8.32184543498e-10 | +1.05125411154e-09 |
| 303 | 708 | 637 | 0 | 1.20465203413e-06 | +3.29204530303e-09 | +2.52591922378e-09 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.79919×, with 9/9 pairs favoring the current release. Median paired total-duration change: -44.4193%.
- KNN/0.3.22 median paired total-time ratio: 1.08686×, with 9/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## Donor counts and observed missingness

These counts are shared across dtypes; the analyzer verifies identical source rows, masks and scaler metadata. Complete donors are fully observed training rows. Available mode also uses partially observed rows separately for each missing feature.

| Seed | Complete donors | Training missing (%) | Query missing (%) | Queries with missing values | Scored cells |
| --- | --- | --- | --- | --- | --- |
| 101 | 1444 | 9.8048 | 10.1571 | 536 | 711 |
| 202 | 1419 | 10.1429 | 9.9000 | 522 | 693 |
| 303 | 1383 | 10.3286 | 10.1143 | 534 | 708 |

### Training rows observed for each feature

Counts are training rows minus missing entries in each feature. They do not guarantee a defined distance to every query. Per-feature reconstruction errors in standardized and source units are preserved separately for each dtype in the full-precision summary.

| Feature | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- |
| Length | 3000 | 3000 | 3000 |
| Diameter | 2666 | 2629 | 2633 |
| Height | 2657 | 2641 | 2634 |
| Whole_weight | 2632 | 2637 | 2617 |
| Shucked_weight | 2664 | 2664 | 2646 |
| Viscera_weight | 2632 | 2655 | 2644 |
| Shell_weight | 2690 | 2644 | 2657 |

## Limits

These are descriptive measurements of one configuration on one runner. No statistical significance, universal speedup or all-input output equivalence is established. Output differences alone do not identify their numerical or neighbor-selection cause. Earlier diagnostics on other seeds or missingness mechanisms do not establish the cause here.

The earlier uniform-weight comparisons and the other dataset run are separate experiments. A matching CPU model does not make them a single controlled run. Cross-run timing differences do not isolate a weights or release effect. Uniform-weight output diagnostics do not establish the cause of the distance-weighted differences reported here.

## Dataset provenance

[Abalone (numerical features)](https://archive.ics.uci.edu/dataset/1/abalone). Nash et al. (1994). Abalone. https://doi.org/10.24432/C55C7W

The source file `abalone.data` contains 4,177 rows. Excluded columns: Sex, Rings. The original source ZIP is preserved; the recorded dataset license is CC BY 4.0. Target values are not used for neighbor search or downstream scoring.

- Source ZIP SHA-256: `755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4`.
- Source data SHA-256: `de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f`.
- Parsed numerical-array fingerprint: `6d71332f0d6a22320c7eff7b18909fab5565975da5a68e7ceef0b49f4572e1a4`.

## Reproduction

The standard-library analysis script validates the archive and source hashes, complete per-dtype worker grids, matching inputs, dependency freezes, repeated outputs and all stored aggregates. It recomputes every displayed statistic from the saved records without installing or running imputers.

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the corresponding matrix entry uses `--dataset abalone --weights distance`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-distance-weights`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require both files; a partially present pair fails. This is saved-data analysis, not a new benchmark.
