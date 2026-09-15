# FaissImputer 0.3.20: released-package benchmark

[Benchmark reports](README.md)
· [Project README](../../README.md#performance)
· [Raw results](../../benchmarks/results/released_versions_0.3.20.json)

## Findings

On this available-donor workload, the published FaissImputer 0.3.20
package achieved a **2.71–2.73× first-transform speedup** over
scikit-learn KNNImputer. Fit plus first transform was **2.44–2.49× as fast**.

Fit itself took longer than KNNImputer. Available-donor worker peak RSS
was also slightly higher than KNNImputer.

Compared with FaissImputer 0.3.19, available-donor first-transform time
fell by **45–49%**, and worker peak RSS fell by **13–14%**.
Complete-donor timings remained close, with slightly higher transform
times for 0.3.20 in this run.

These results describe one synthetic workload on one runner. They do
not establish performance across other data sizes, missingness patterns,
metrics, or hardware.

## Conditions and provenance

| Item | Configuration |
| --- | --- |
| Measurement date | 2026-09-15 |
| FaissImputer packages | Published PyPI versions 0.3.19 and 0.3.20 |
| Reference | scikit-learn KNNImputer 1.9.1 |
| Runtime | Python 3.12.14, NumPy 2.5.3, Faiss 1.15.0 |
| Platform | Linux x86_64, AMD EPYC 7763, four logical CPUs available |
| Native threads | 1 |
| Training rows / query rows | 20,000 / 300 |
| Features / neighbors | 20 / 5 |
| Aggregation | Mean with uniform weights |
| Search | FaissImputer Flat L2; KNNImputer nan-euclidean |
| Complete-donor case | Fully observed training data |
| Available-donor case | Training data with 10% MCAR missingness |
| Query missingness | Four randomly selected missing features per row |
| Input dtypes | float32 and float64 |
| Seeds | 101, 202, 303 |
| Repetitions | Three fresh workers per seed and configuration |
| Repeated transforms | Two additional transforms per worker |
| scikit-learn working memory | 256 MiB |

The run completed **108 of 108 workers** successfully:
three methods × two donor policies × two dtypes × three seeds ×
three repetitions. All workers passed output and input-preservation checks.

Each worker ran in a fresh subprocess, sequentially. Method order was
rotated. Matching cases used identical input fingerprints, and package
environments used matching dependency versions.

A small untimed warmup preceded the measured fit and transforms.
“First transform” therefore means the first transform of the measured
fitted estimator, not process cold-start latency.

- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/35015576984)
- Benchmark-code commit: `bbc74347d70d122da237b2836a0883e3bb809a12`
- [Workflow at the measured commit](https://github.com/ScionKim/FaissImputer/blob/bbc74347d70d122da237b2836a0883e3bb809a12/.github/workflows/benchmark-released.yml)

The commit identifies the benchmark code and workflow. The measured
FaissImputer implementations were the installed PyPI releases.

## Timing results

All times below are **milliseconds**.

Each value is a median across nine workers: three seeds with three
repetitions each. Repeated-transform time is the median of each worker's
two-transform median.

Fit plus first transform is calculated within each worker before taking
the median. It is not the sum of the separately reported medians, and it
is not a separate measurement of `fit_transform()`.

| Donor case | Dtype | Method | Fit | First transform | Repeated transform | Fit + first transform |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Complete | float32 | KNNImputer | 1.753 | 292.451 | 283.983 | 294.174 |
| Complete | float32 | FaissImputer 0.3.19 | 4.932 | 148.309 | 140.939 | 153.080 |
| Complete | float32 | FaissImputer 0.3.20 | 4.924 | 148.552 | 141.515 | 153.414 |
| Complete | float64 | KNNImputer | 1.984 | 358.581 | 347.819 | 360.990 |
| Complete | float64 | FaissImputer 0.3.19 | 7.038 | 204.688 | 196.194 | 212.572 |
| Complete | float64 | FaissImputer 0.3.20 | 7.031 | 207.709 | 198.491 | 214.739 |
| Available | float32 | KNNImputer | 1.811 | 279.931 | 272.959 | 281.724 |
| Available | float32 | FaissImputer 0.3.19 | 11.735 | 201.626 | 196.481 | 213.700 |
| Available | float32 | FaissImputer 0.3.20 | 11.915 | 103.402 | 96.675 | 115.565 |
| Available | float64 | KNNImputer | 2.089 | 334.845 | 329.521 | 336.841 |
| Available | float64 | FaissImputer 0.3.19 | 14.021 | 224.154 | 217.435 | 238.391 |
| Available | float64 | FaissImputer 0.3.20 | 12.867 | 122.476 | 116.230 | 135.213 |

Speedups use the ratio of unrounded medians:
KNNImputer time divided by FaissImputer 0.3.20 time.

### First-transform variation

The following are observed minimum–maximum times across the nine workers,
in milliseconds. These ranges are not confidence intervals; the workers
cover three distinct data seeds, each repeated three times.

| Donor case | Dtype | KNNImputer | FaissImputer 0.3.19 | FaissImputer 0.3.20 |
| --- | --- | ---: | ---: | ---: |
| Complete | float32 | 289.005–300.829 | 145.288–153.805 | 144.071–152.420 |
| Complete | float64 | 352.016–367.905 | 202.783–209.846 | 204.322–212.887 |
| Available | float32 | 274.254–286.983 | 199.122–205.444 | 101.316–104.695 |
| Available | float64 | 327.482–353.223 | 222.310–228.613 | 120.094–126.176 |

## Process memory

Values are medians in **MiB**.

Post-fit RSS change is process RSS after fit minus process RSS before fit.
It includes allocator effects and is not a measurement of fitted-model
storage alone.

Worker peak RSS covers the whole worker, including imports, data
preparation, warmup, transforms, and validation. It is not a separate peak
for fit or transform.

| Donor case | Dtype | Method | Post-fit RSS change | Worker peak RSS |
| --- | --- | --- | ---: | ---: |
| Complete | float32 | KNNImputer | 1.40 | 235.89 |
| Complete | float32 | FaissImputer 0.3.19 | 3.00 | 149.87 |
| Complete | float32 | FaissImputer 0.3.20 | 2.93 | 149.89 |
| Complete | float64 | KNNImputer | 3.05 | 249.57 |
| Complete | float64 | FaissImputer 0.3.19 | 6.04 | 152.52 |
| Complete | float64 | FaissImputer 0.3.20 | 5.98 | 152.36 |
| Available | float32 | KNNImputer | 1.40 | 235.91 |
| Available | float32 | FaissImputer 0.3.19 | 9.94 | 285.99 |
| Available | float32 | FaissImputer 0.3.20 | 9.48 | 248.60 |
| Available | float64 | KNNImputer | 3.05 | 249.59 |
| Available | float64 | FaissImputer 0.3.19 | 10.30 | 293.14 |
| Available | float64 | FaissImputer 0.3.20 | 13.06 | 251.77 |

The available-donor peak RSS reduction versus 0.3.19 does not imply
a reduction in every memory measure. In particular, the float64 post-fit
RSS change was higher in this run.

## Output agreement and imputation quality

FaissImputer 0.3.20 and 0.3.19 produced identical output values in every
measured case. Repeated transforms also produced identical values.

Maximum absolute differences between FaissImputer 0.3.20 and KNNImputer:

| Donor case | Dtype | Maximum absolute difference |
| --- | --- | ---: |
| Complete | float32 | 0 |
| Complete | float64 | 0 |
| Available | float32 | 4.76837158203125e-7 |
| Available | float64 | 8.881784197001252e-16 |

Agreement with another imputer is separate from error against hidden
ground truth. The benchmark scored the 1,200 deliberately hidden query
values in each worker.

Median quality results, rounded to six decimal places, were the same
for all three methods within each case:

| Donor case | Dtype | RMSE | MAE |
| --- | --- | ---: | ---: |
| Complete | float32 | 0.176254 | 0.132619 |
| Complete | float64 | 0.176254 | 0.132619 |
| Available | float32 | 0.180675 | 0.136436 |
| Available | float64 | 0.180675 | 0.136436 |

Matching rounded quality metrics do not imply identical output values.
The raw results retain the unrounded metrics and output differences.

Complete and available cases used different training missingness, so
their quality scores should not be interpreted as a controlled comparison
of donor policies on identical training inputs.

These checks cover the measured workload. They do not establish general
numerical equivalence with KNNImputer.

## Reproduction

Use the recorded workflow and benchmark-code revision, with:

- Workflow: `Version comparison benchmark`
- Mode: `released`
- `baseline_commit`: empty
- Current PyPI version: `0.3.20`
- Previous PyPI version: `0.3.19`

The package versions are configured in the workflow, not additional
workflow-dispatch inputs. Later workflow defaults may differ.

The workflow creates separate environments, installs the released
packages with matching dependencies, and runs the benchmark outside the
source checkout. It invokes:

```bash
"$CURRENT_ENV/bin/python" -u -m benchmarks.benchmark_released_versions \
  --previous-python "$PREVIOUS_ENV/bin/python" \
  --previous-version "0.3.19" \
  --current-version "0.3.20" \
  --train-size 20000 \
  --queries 300 \
  --seeds 101 202 303 \
  --repeats 3 \
  --repeated-transforms 2 \
  --timeout-seconds 180 \
  --budget-seconds 1200 \
  --output "$GITHUB_WORKSPACE/benchmark_outputs/version_comparison.json"
```

The remaining workload settings are those listed above and implemented
by the recorded benchmark revision.

The archived [raw results](../../benchmarks/results/released_versions_0.3.20.json)
retain individual timings, memory measurements, input fingerprints,
environment details, and validation results.