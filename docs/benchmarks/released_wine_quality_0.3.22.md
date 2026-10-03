# Wine Quality White: released 0.3.22 and 0.3.21

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares two installed FaissImputer releases and KNNImputer on one held-out Wine Quality White workload. It measures consecutive `fit(train)` and first `transform(query)` calls with available donors, float64 inputs, and MCAR missingness. It does not pool same-data APIs, dtypes, donor policies, or runs.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_wine_quality_0.3.22.zip).
- [Full-precision analysis](../../benchmarks/results/released_wine_quality_0.3.22-summary.json) and [generator](../../benchmarks/analyze_released_real_data.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/37076320747/attempts/1).
- Benchmark source commit: `2e07c594b60f7b755c726d991a6548adae6f0331`. Package measurements use installed distributions outside the checkout.
- Archive SHA-256: `d6a2c52875db764124fbea5b312138356476b7ea55d503b3ee0ae2a514db0a39`.
- `version_comparison_wine_quality_white.json` SHA-256: `3bc7e665f8ddfb66b453dc930f1eedd2a92e716701e3b18ce9b5a63b0fb2bf1d`.
- Runner: AMD EPYC 9V74 80-Core Processor; 4 logical CPUs, 4 in affinity; all recorded native thread counts are one.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 3,000 training rows; 1,000 held-out query rows; 11 numerical features; k=5; uniform weights. Faiss uses built-in L2, mean aggregation, `donor_policy="available"`, and `index_factory="Flat"`.
- Nominal overall missingness is 10% in training and query inputs. `alcohol` remains observed; missingness is sampled on the other features. Actual rates are reported below.
- Seeds: 101, 202, 303; 3 fresh workers per seed and variant; 27 successful workers. Variant order rotates across repetitions.

The archived dependency freezes differ only in the FaissImputer release. KNNImputer uses the 0.3.22 environment. Inputs, split identities, masks, scaler metadata and fingerprints agree across all variants and repeats for each seed. Standardization uses observed training values only; held-out ground truth retains float64 precision.

## Aggregation methodology

Timing and memory use **9 records = 3 seeds × 3 repeats** per method. Each table entry is median [min–max]. `transform_seconds` measures the first held-out transform. `total_seconds` measures fit plus that transform. Fit and transform are consecutive, with no explicit garbage collection or RSS sampling between them. Preparation, warmup, validation, process startup and serialization are outside the measured interval. No additional transform calls are measured in this experiment.

Speedup is **numerator record time / denominator record time**, calculated for each matching pair, then summarized as median [min–max]. Each comparison contains **9 pairs** with the same archived run, dataset, training/query sizes, features, neighbors, donor regime, metric/weights configuration, missingness configuration, dtype, API, thread settings, seed, repeat, and prepared inputs. The ratios are **not obtained by dividing method-level median times**. Values above one favor the denominator.

Duration change is computed for each pair as `100 * (denominator time / numerator time - 1)` and then summarized. Negative values mean the denominator took less time. Observed ranges are not confidence intervals. Timing repetitions reuse each seed dataset; they do not create additional independent datasets.

RMSE and MAE use **3 seed metrics**, taking repeat 1 after verifying repeat consistency of metrics, stored imputed values and output hashes. They measure error on masked held-out entries against ground truth, in units standardized using the training data. They do not measure method-to-method differences.

The analyzer uses unrounded JSON numbers for all calculations. The summary preserves full-precision timing samples, pair indices, numerators, denominators, ratios, duration changes, quality samples and input/output fingerprints. Only this Markdown presentation is rounded. Stored worker RMSE/MAE values are aggregated; ground-truth and mask arrays are not stored as arrays in the result JSON, so those underlying errors are not recomputed by this analyzer. Pairwise hidden-entry output differences are recomputed directly from the archived `imputed_values` arrays.

## Timing

Milliseconds, median [min–max] across 9 workers per method. The full-precision summary keeps the original seconds.

| Method | Fit (ms) | First transform (ms) | Fit + first transform (ms) |
| --- | --- | --- | --- |
| KNNImputer | 0.734 [0.710–0.902] | 66.142 [64.246–73.515] | 66.876 [64.971–74.247] |
| FaissImputer 0.3.21 | 1.492 [1.433–1.521] | 182.435 [180.449–182.986] | 183.947 [181.970–184.452] |
| FaissImputer 0.3.22 | 1.465 [1.418–1.560] | 160.886 [159.351–163.483] | 162.362 [160.769–165.042] |

## Matched timing ratios

Median [min–max] of 9 numerator/denominator ratios per comparison. Above one favors the denominator; below one favors the numerator.

| Numerator / denominator | Fit ratio | First-transform ratio | Total ratio |
| --- | --- | --- | --- |
| FaissImputer 0.3.21 / FaissImputer 0.3.22 | 1.0036 [0.9252–1.0527] | 1.1334 [1.1083–1.1411] | 1.1322 [1.1065–1.1403] |
| KNNImputer / FaissImputer 0.3.22 | 0.5085 [0.4692–0.6159] | 0.4084 [0.4032–0.4515] | 0.4091 [0.4041–0.4530] |
| KNNImputer / FaissImputer 0.3.21 | 0.4998 [0.4769–0.6150] | 0.3616 [0.3533–0.4058] | 0.3628 [0.3544–0.4066] |

### 0.3.22 duration change relative to 0.3.21

Median [min–max] of matched per-pair percentage changes. Negative values mean less time.

| Phase | Duration change (%) | Pairs favoring 0.3.22 |
| --- | --- | --- |
| Fit | -0.3623 [-5.0085–8.0845] | 6/9 |
| First transform | -11.7723 [-12.3674–-9.7689] | 9/9 |
| Fit + first transform | -11.6797 [-12.3072–-9.6278] | 9/9 |

## Memory

MiB, median [min–max] across 9 workers per method. Peak RSS is sampled after validation and before worker JSON serialization. It includes imports, input preparation, warmup, fit, transform and validation. It is neither isolated transform memory nor retained model size. The sklearn working-memory setting is not a process RAM limit.

| Method | Worker peak RSS (MiB) |
| --- | --- |
| KNNImputer | 175.887 [174.855–176.316] |
| FaissImputer 0.3.21 | 157.621 [157.539–157.824] |
| FaissImputer 0.3.22 | 157.738 [157.539–157.953] |

## Reconstruction quality

Median [min–max] across 3 seeds, using each seed's masked held-out query entries. Scored-cell counts are listed below. Similar aggregate errors do not establish equality of individual imputed values or algorithmic equivalence.

| Method | RMSE | MAE |
| --- | --- | --- |
| KNNImputer | 0.8013359904 [0.6943968041–0.8514761195] | 0.5049077639 [0.4991907279–0.5545421516] |
| FaissImputer 0.3.21 | 0.8013359904 [0.6943968041–0.8514761195] | 0.5049077639 [0.4991907279–0.5545421516] |
| FaissImputer 0.3.22 | 0.8013359904 [0.6943968041–0.8514761195] | 0.5049077639 [0.4991907279–0.5545421516] |

### Output agreement

Hash counts cover 9 matched timing pairs but only 3 distinct seed inputs. Hashes are those recorded for the full output arrays. Hidden-entry differences are recomputed from stored imputed values. All differences below are maxima across matching pairs.

| Comparison | Matching full-output hashes | Max hidden-entry difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- |
| FaissImputer 0.3.22 vs FaissImputer 0.3.21 | 9/9 | 0 | 0 | 0 |
| FaissImputer 0.3.22 vs KNNImputer | 0/9 | 4.4408920985e-16 | 0 | 0 |

## Donor counts and observed missingness

One row per seed, after input and case metadata checks across variants and repeats. Complete donors are fully observed training rows; available mode also uses partially observed rows, separately for each missing feature.

| Seed | Complete donors | Training missing (%) | Query missing (%) | Queries with missing values | Scored cells |
| --- | --- | --- | --- | --- | --- |
| 101 | 943 | 9.8030 | 10.0455 | 706 | 1105 |
| 202 | 964 | 9.8939 | 10.0273 | 699 | 1103 |
| 303 | 910 | 10.4333 | 9.9364 | 680 | 1093 |

### Training rows observed for each feature

Counts are training rows minus missing training entries in each feature. They describe observed donor values, not a guarantee of a defined distance to every query. Feature-level reconstruction errors in standardized and source units are preserved in the full-precision summary.

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

## Interpretation and limits

- 0.3.22 first-transform speedup relative to 0.3.21 is 1.13343×; 9/9 matched first transforms favor the current release. Median paired duration change is -11.7723% for first transform and -11.6797% for fit plus first transform.
- KNN/0.3.22 total-time ratio is 0.40913×. In this workload KNN takes less time in every matched total-time pair; the version-to-version improvement does not make FaissImputer faster than KNN here.
- 0.3.21/0.3.22 full-output hashes match in 9/9 pairs, with maximum hidden-entry difference 0. Compared with KNN, 0.3.22 has 0/9 matching full-output hashes and maximum hidden-entry difference 4.4408920985e-16. These observations apply to these inputs and versions only.
- Native thread counts, package origins and archived dependency versions are checked. Timings remain descriptive observations from one runner; no statistical significance, universal speedup or all-input equivalence is established.
- The [published synthetic comparison](released_versions_0.3.22.md) and [earlier real-data coverage](real-data-datasets-ef04b1b.md) are separate experiments. Their hardware, versions or workloads differ. Cross-run absolute timing changes do not isolate a release effect.

## Dataset provenance

[Wine Quality (white)](https://archive.ics.uci.edu/dataset/186/wine+quality). Cortez et al. (2009). Wine Quality. https://doi.org/10.24432/C56S3T

The preserved artifact includes the original source ZIP and identifies the dataset license as CC BY 4.0. `winequality-white.csv` contains 4,898 rows; `quality` is excluded from the predictor matrix. No target values are used for neighbor search or downstream prediction scoring.

- Source ZIP SHA-256: `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`.
- Source CSV SHA-256: `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836`.
- Parsed predictor-array fingerprint: `518c625f745e3807da855507e3b47c6b3d9dd499699b55dd0c5a88105d96674c`.

## Reproduction

The standard-library-only generator reads the preserved artifact and reuses validation and formatting helpers from the published synthetic analysis. It neither installs nor executes imputers. It checks archive and member hashes, the complete worker grid, input consistency, dependency freezes, repeated outputs, and every stored summary and comparison before writing this report and the full-precision JSON.

The [Analyze released-version benchmark results workflow](../../.github/workflows/analyze-released-versions.yml) regenerates both historical synthetic reports and this Wine report. On the first push to `bench/released-wine-quality-0.3.22`, it may generate the Wine outputs when both committed files are absent. Commit the Markdown and summary JSON together; subsequent runs compare them byte-for-byte. Pull requests require both Wine outputs and both synthetic output pairs. A partially present pair fails. The workflow runs analysis of saved data, not a new benchmark.
