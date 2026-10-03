# Abalone: released 0.3.22 and 0.3.21

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Abalone numerical data. Float32 and float64 use separate records and tables. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. No same-data API or complete-donor measurements are included.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_abalone_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_abalone_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37099197073/attempts/1).
- Benchmark source commit: `4acd09dfa1aca339f38d401bdb2a1076bf3abe8c`. Measured packages were installed outside the checkout.
- Archive SHA-256: `8d2e491b9f066312627e5ea8ad6b955269c1634e7ae059b71d1aecac8579b638`.
- `version_comparison_abalone_float32.json` SHA-256: `50d92bbc54652408795939f377f862a191a1a4919bd27418e3caf9f76b347bea`.
- `version_comparison_abalone_float64.json` SHA-256: `d4e5814a2fd5b773f0d268e940fddeccc67fdf0e459edd2dc678a43a94d089ef`.
- Runner: INTEL(R) XEON(R) PLATINUM 8573C; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 7 numerical features; k=5; uniform weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
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
| KNNImputer | 0.607 [0.582–0.686] | 46.637 [44.875–65.463] | 47.222 [45.493–66.149] |
| FaissImputer 0.3.21 | 1.157 [1.117–1.214] | 36.366 [36.251–37.363] | 37.547 [37.399–38.552] |
| FaissImputer 0.3.22 | 1.148 [1.111–1.194] | 36.721 [36.115–37.368] | 37.835 [37.271–38.516] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0094 [0.9419–1.0896] | 1.0014 [0.9723–1.0274] | 1.0016 [0.9723–1.0268] |
| KNNImputer / FaissImputer 0.3.22 | 0.5310 [0.5053–0.6130] | 1.2571 [1.2067–1.8048] | 1.2345 [1.1866–1.7691] |
| KNNImputer / FaissImputer 0.3.21 | 0.5229 [0.4955–0.6073] | 1.2633 [1.2214–1.8023] | 1.2406 [1.2015–1.7662] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -0.9314 [-8.2257–6.1713] | 7/9 |
| First transform | -0.1374 [-2.6648–2.8463] | 5/9 |
| Fit + first transform | -0.1614 [-2.6077–2.8448] | 5/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 164.996 [164.449–165.188] |
| FaissImputer 0.3.21 | 155.363 [155.094–155.535] |
| FaissImputer 0.3.22 | 155.332 [155.070–155.512] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.2745597659 [0.2585029216–0.3106408829] | 0.1771440647 [0.1761232181–0.1817598461] |
| FaissImputer 0.3.21 | 0.2746437054 [0.2585029206–0.3106408855] | 0.1773996637 [0.1761232157–0.1817598464] |
| FaissImputer 0.3.22 | 0.2746437054 [0.2585029206–0.3106408855] | 0.1773996637 [0.1761232157–0.1817598464] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.106646299362 | 8.39395573218e-05 | 0.000255598951886 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 711 | 137 | 2 | 0.106646299362 | +8.39395573218e-05 | +0.000255598951886 |
| 202 | 693 | 121 | 0 | 2.38418579102e-07 | -9.25999277257e-10 | -2.45180872827e-09 |
| 303 | 708 | 143 | 0 | 2.38418579102e-07 | +2.54133852851e-09 | +3.03222363884e-10 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.00162×, with 5/9 pairs favoring the current release. Median paired total-duration change: -0.1614%.
- KNN/0.3.22 median paired total-time ratio: 1.23451×, with 9/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## float64

### Timing

Milliseconds, median [min–max] across nine workers per method.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.621 [0.563–0.695] | 45.111 [43.810–47.791] | 45.749 [44.372–48.486] |
| FaissImputer 0.3.21 | 1.206 [1.185–1.289] | 59.172 [57.496–60.103] | 60.371 [58.681–61.391] |
| FaissImputer 0.3.22 | 1.194 [1.164–1.244] | 39.691 [39.347–41.095] | 40.934 [40.542–42.301] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0044 [0.9586–1.0857] | 1.4839 [1.4496–1.5062] | 1.4723 [1.4368–1.4920] |
| KNNImputer / FaissImputer 0.3.22 | 0.5194 [0.4570–0.5879] | 1.1358 [1.0945–1.2134] | 1.1176 [1.0755–1.1952] |
| KNNImputer / FaissImputer 0.3.21 | 0.5235 [0.4521–0.5795] | 0.7744 [0.7338–0.8077] | 0.7694 [0.7280–0.8031] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -0.4343 [-7.8958–4.3175] | 5/9 |
| First transform | -32.6086 [-33.6065–-31.0153] | 9/9 |
| Fit + first transform | -32.0780 [-32.9759–-30.3989] | 9/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 166.773 [166.125–167.074] |
| FaissImputer 0.3.21 | 155.691 [155.531–155.910] |
| FaissImputer 0.3.22 | 155.812 [155.605–156.070] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.2746437061 [0.2585029221–0.3106408858] | 0.1773996625 [0.1761232158–0.1817598460] |
| FaissImputer 0.3.21 | 0.2745630451 [0.2585029221–0.3106408858] | 0.1771637007 [0.1761232158–0.1817598460] |
| FaissImputer 0.3.22 | 0.2745630451 [0.2585029221–0.3106408858] | 0.1771637007 [0.1761232158–0.1817598460] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.106646246792 | 8.06609438118e-05 | 0.000235961832035 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 711 | 147 | 4 | 0.106646246792 | -8.06609438118e-05 | -0.000235961832035 |
| 202 | 693 | 121 | 0 | 4.4408920985e-16 | +0 | +0 |
| 303 | 708 | 157 | 0 | 8.881784197e-16 | +5.55111512313e-17 | +2.77555756156e-17 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.47228×, with 9/9 pairs favoring the current release. Median paired total-duration change: -32.0780%.
- KNN/0.3.22 median paired total-time ratio: 1.11760×, with 9/9 pairs favoring the current release. Fit alone is reported separately above.
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

The [Wine Quality release comparison](released_wine_quality_0.3.22.md), [synthetic release comparison](released_versions_0.3.22.md) and [earlier real-data coverage](real-data-datasets-ef04b1b.md) are separate experiments. Cross-run absolute timing differences do not isolate a release effect.

## Dataset provenance

[Abalone (numerical features)](https://archive.ics.uci.edu/dataset/1/abalone). Nash et al. (1994). Abalone. https://doi.org/10.24432/C55C7W

The source file `abalone.data` contains 4,177 rows. Excluded columns: Sex, Rings. The original source ZIP is preserved; the recorded dataset license is CC BY 4.0. Target values are not used for neighbor search or downstream scoring.

- Source ZIP SHA-256: `755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4`.
- Source data SHA-256: `de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f`.
- Parsed numerical-array fingerprint: `6d71332f0d6a22320c7eff7b18909fab5565975da5a68e7ceef0b49f4572e1a4`.

## Reproduction

The standard-library analysis script validates the archive and source hashes, complete per-dtype worker grids, matching inputs, dependency freezes, repeated outputs and all stored aggregates. It recomputes every displayed statistic from the saved records without installing or running imputers.

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the Abalone matrix entry uses `--dataset abalone`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-abalone-results-0.3.22`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require the output pair, and a partially present pair fails. This is saved-data analysis, not a new benchmark.
