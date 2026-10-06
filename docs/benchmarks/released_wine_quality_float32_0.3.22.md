# Wine Quality White: released 0.3.22 and 0.3.21 — float32, uniform weights

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed FaissImputer releases and KNNImputer on held-out Wine Quality White data with `weights="uniform"`. Each worker measures consecutive `fit(train)` and first `transform(query)` calls with available donors. Each dtype has its own records and tables.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_wine_quality_float32_0.3.22.zip).
- [Full-precision summary](../../benchmarks/results/released_wine_quality_float32_0.3.22-summary.json) and [analysis script](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37403070692/attempts/1).
- Benchmark source commit: `6d533d37c8173058ce2cfb314701377ee67b4b59`. Measured packages were installed outside the checkout.
- Archive SHA-256: `0654259363cd7c1d76f467fc06e5c3fd104f24198f9a7ef865569bdd2c4833bd`.
- `version_comparison_wine_quality_white_float32.json` SHA-256: `e4b16d9794a944afc8e63dceffe2cb04a97132ff7f7a7228be562999757a7ece`.
- Runner: AMD EPYC 7763 64-Core Processor; 4 logical CPUs, 4 in affinity; one native thread.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out queries; 11 numerical features; k=5; uniform weights; mean aggregation; `donor_policy="available"`; `index_factory="Flat"`.
- 10% target overall MCAR missingness in training and query inputs. `alcohol` stays observed; the other features are eligible for masking.
- 27 successful workers: 3 seeds × 3 repeats × 3 methods × 1 dtype. Variants rotate within each seed/repeat. Only float32 is measured in this run.

The environment freezes differ only in the FaissImputer release. KNNImputer uses the 0.3.22 environment. Prepared cases, source rows, masks, scaler parameters and float64 scoring truth match across variants and repeats. Scaling uses observed training values only.

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
| KNNImputer | 0.647 [0.639–0.685] | 63.444 [61.378–65.097] | 64.085 [62.020–65.742] |
| FaissImputer 0.3.21 | 1.346 [1.314–1.407] | 164.850 [162.601–167.287] | 166.223 [163.938–168.684] |
| FaissImputer 0.3.22 | 1.345 [1.292–1.410] | 164.195 [163.049–167.723] | 165.573 [164.364–169.015] |

### Matched timing ratios

Median [min–max] of nine matched numerator/denominator ratios.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0212 [0.9481–1.0635] | 1.0040 [0.9768–1.0096] | 1.0041 [0.9766–1.0093] |
| KNNImputer / FaissImputer 0.3.22 | 0.4891 [0.4544–0.5257] | 0.3811 [0.3744–0.3967] | 0.3818 [0.3752–0.3974] |
| KNNImputer / FaissImputer 0.3.21 | 0.4822 [0.4573–0.5212] | 0.3893 [0.3720–0.3949] | 0.3901 [0.3727–0.3956] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched percentage changes; negative means less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -2.0794 [-5.9749–5.4703] | 6/9 |
| First transform | -0.3970 [-0.9515–2.3720] | 5/9 |
| Fit + first transform | -0.4112 [-0.9255–2.3973] | 5/9 |

### Memory

Whole-worker peak RSS in MiB, median [min–max] across nine workers. The sample is taken after validation and before JSON serialization, including imports, preparation and warmup. It is neither isolated transform memory nor retained model size. The working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 173.078 [172.020–173.375] |
| FaissImputer 0.3.21 | 157.113 [156.984–157.328] |
| FaissImputer 0.3.22 | 157.195 [156.641–157.438] |

### Reconstruction quality

Median [min–max] of three seed-level errors against ground truth.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.8013359890 [0.6941226106–0.8514761198] | 0.5049077624 [0.4991583771–0.5545421515] |
| FaissImputer 0.3.21 | 0.8013359884 [0.6943968043–0.8514761211] | 0.5049077623 [0.4991907277–0.5545421522] |
| FaissImputer 0.3.22 | 0.8013359884 [0.6943968043–0.8514761211] | 0.5049077623 [0.4991907277–0.5545421522] |

### Output agreement

Hash counts cover nine matched pairs but three distinct seed inputs. Other columns are maximum absolute differences across those pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 9/9 | 0 | 0 | 0 |
| KNNImputer / FaissImputer 0.3.22 | 0/9 | 0.263017714024 | 0.000274193702368 | 3.23506258102e-05 |

### 0.3.22 versus KNNImputer by seed

One observation per seed after repeat-consistency checks. Signed error differences are current release minus KNNImputer; negative means lower reconstruction error on that seed.

| Seed | Scored cells | Different hidden entries | Entries differing >1e-5 | Max hidden-entry difference | RMSE difference | MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| 101 | 1105 | 249 | 2 | 0.263017714024 | +0.000274193702368 | +3.23506258102e-05 |
| 202 | 1103 | 264 | 0 | 2.38418579102e-07 | +1.25172094911e-09 | +7.5790529408e-10 |
| 303 | 1093 | 225 | 0 | 4.76837158203e-07 | -6.36695252076e-10 | -8.69120331259e-11 |

### Interpretation

- 0.3.21/0.3.22 median paired total-time ratio: 1.00413×, with 5/9 pairs favoring the current release. Median paired total-duration change: -0.4112%.
- KNN/0.3.22 median paired total-time ratio: 0.38176×, with 0/9 pairs favoring the current release. Fit alone is reported separately above.
- 0.3.21/0.3.22 recorded full-output hashes match in 9/9 pairs; maximum saved hidden-entry difference is 0. The seed table separately records differences from KNNImputer.

## Donor counts and observed missingness

One observation per seed after input consistency checks across variants and repeats. Complete donors are fully observed training rows. Available mode also uses partially observed rows separately for each missing feature.

| Seed | Complete donors | Training missing (%) | Query missing (%) | Queries with missing values | Scored cells |
| --- | --- | --- | --- | --- | --- |
| 101 | 943 | 9.8030 | 10.0455 | 706 | 1105 |
| 202 | 964 | 9.8939 | 10.0273 | 699 | 1103 |
| 303 | 910 | 10.4333 | 9.9364 | 680 | 1093 |

### Training rows observed for each feature

Counts are training rows minus missing entries in each feature. They do not guarantee a defined distance to every query. Per-feature reconstruction errors in standardized and source units are preserved separately for each dtype in the full-precision summary.

| Feature | Seed 101 | Seed 202 | Seed 303 |
| --- | --- | --- | --- |
| fixed acidity | 2681 | 2700 | 2665 |
| volatile acidity | 2703 | 2639 | 2674 |
| citric acid | 2671 | 2654 | 2611 |
| residual sugar | 2689 | 2680 | 2664 |
| chlorides | 2672 | 2641 | 2672 |
| free sulfur dioxide | 2687 | 2677 | 2657 |
| total sulfur dioxide | 2660 | 2692 | 2657 |
| density | 2670 | 2704 | 2629 |
| pH | 2657 | 2653 | 2659 |
| sulphates | 2675 | 2695 | 2669 |
| alcohol | 3000 | 3000 | 3000 |

## Limits

These are descriptive measurements of one configuration on one runner. No statistical significance, universal speedup or all-input output equivalence is established. Output differences alone do not identify their numerical or neighbor-selection cause. Earlier diagnostics on other seeds or missingness mechanisms do not establish the cause here.

The float32 uniform and distance archives were measured in separate runs on different CPU models. They are not a hardware-controlled weights comparison. The earlier float64 archives are also separate experiments. Cross-run timing differences do not isolate a weights, dtype or release effect. Diagnostics on other configurations do not establish the cause of the output differences reported here.

## Dataset provenance

[Wine Quality (white)](https://archive.ics.uci.edu/dataset/186/wine+quality). Cortez et al. (2009). Wine Quality. https://doi.org/10.24432/C56S3T

The source file `winequality-white.csv` contains 4,898 rows. Excluded columns: quality. The original source ZIP is preserved; the recorded dataset license is CC BY 4.0. Target values are not used for neighbor search or downstream scoring.

- Source ZIP SHA-256: `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`.
- Source data SHA-256: `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836`.
- Parsed numerical-array fingerprint: `518c625f745e3807da855507e3b47c6b3d9dd499699b55dd0c5a88105d96674c`.

## Reproduction

The standard-library analysis script validates the archive and source hashes, complete per-dtype worker grids, matching inputs, dependency freezes, repeated outputs and all stored aggregates. It recomputes every displayed statistic from the saved records without installing or running imputers.

In the [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml), the corresponding matrix entry uses `--dataset wine_quality_white --dtype float32 --weights uniform`. It produces this Markdown report and the full-precision summary. On the first push or manual analysis run on `bench/released-wine-quality-float32-results`, both outputs may initially be absent. Commit them together; subsequent runs compare them byte-for-byte. Pull requests require both files; a partially present pair fails. This is saved-data analysis, not a new benchmark.
