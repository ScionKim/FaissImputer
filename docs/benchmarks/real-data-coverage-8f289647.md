# Held-out real-data comparison: source 8f289647

[Benchmark index](README.md) ·
[Raw results](../../benchmarks/results/real-data-coverage-8f289647.json) ·
[Workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/35287620647)

This report measures a development build on California Housing features
with artificially introduced missing values. The prediction target is
unused.

These are source-candidate measurements, not results for the published
FaissImputer 0.3.20 package.

## Main findings

Across 10,000 and 15,000 training rows, 3,000 held-out queries,
MCAR/MAR missingness, and float32/float64 inputs:

- Available mode completed fit plus first transform 1.37–1.82× as fast
  as KNNImputer. First transform alone was 1.38–1.84× as fast.
- Available-mode process peak RSS medians were 47.7–63.0% lower.
- Complete mode completed fit plus first transform 7.36–10.46× as fast
  as KNNImputer, with 58.2–73.2% lower process peak RSS medians.
- Complete-mode reconstruction RMSE medians were 15.9–27.5% lower
  than KNNImputer in this workload. Complete donors represented
  42.59–47.60% of training rows.
- Available-mode aggregate reconstruction errors closely matched
  KNNImputer, but one float32 case had a material individual-output
  difference. See the output-agreement section.

Ratios compare condition-level medians. They are not medians of
seed-by-seed ratios.

All 360 runs passed output and input-preservation checks. Controller
elapsed time was 501.86 seconds, excluding workflow setup.

## Build and environment

| Item | Value |
| --- | --- |
| Source commit | `8f289647dcb8e7bd0c7e32ed29153db73a33dc26` |
| Candidate version | `0.3.20+bench.8f289647dcb8` |
| Workflow run / attempt | `35287620647` / `1` |
| CPU | AMD EPYC 7763 64-Core Processor |
| Visible logical CPUs | 4 |
| Native threads | 1 |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Faiss | 1.15.1 |
| Platform | Linux 6.17.0-1022-azure, x86_64, glibc 2.39 |
| scikit-learn working memory | 256 MiB |

The working-memory setting does not cap total process memory.

Candidate wheel:

```text
faiss_imputer-0.3.20+bench.8f289647dcb8-py3-none-any.whl
```

Wheel SHA256:

```text
1ac4cc1d849fc51bc6414e6306c51ae48797f0da502af2bbdef63ff0dbbb6b17
```

## Data and protocol

California Housing contains 20,640 rows and eight numerical features:

`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
`AveOccup`, `Latitude`, and `Longitude`.

The benchmark uses the features only. It does not evaluate housing-price
prediction.

Dataset fingerprint, including array shape, dtype, and logical values:

```text
ddea995a7713b3236dac0634457255ff07ff08ed63a4062d91a6e257fcdc5099
```

For each seed:

- Select 3,000 held-out query rows.
- Select disjoint training sets of 10,000 and 15,000 rows as nested
  prefixes of the remaining shuffled rows.
- Keep raw query rows and query missingness masks fixed across training
  sizes and input dtypes.
- Keep `MedInc` observed. Apply missingness to the other seven features.
- Fit `StandardScaler` on observed training values only.
- Convert model inputs to the requested dtype while retaining float64
  scoring truth.

MCAR uses a missingness probability of `0.10 × 8 / 7` for each eligible
entry. MAR multiplies that probability by 0.5 or 1.5 according to whether
the row's original `MedInc` is at or below, or above, the median of the
common first 10,000 training rows.

The nominal overall missing rate is 10%. Actual query rates were
9.84–10.58%; complete counts and feature-level missingness are recorded
in the raw results.

Each training size and mechanism has its own training-fitted scaler.
Standardized inputs and scoring units can therefore change across cases,
even when the raw query rows are shared.

Methods:

- `SimpleImputer(strategy="mean")`
- `SimpleImputer(strategy="median")`
- `KNNImputer(n_neighbors=5, weights="uniform")`
- FaissImputer with complete donors, five neighbors, uniform mean
  aggregation, built-in L2, and `index_factory="Flat"`
- FaissImputer with available donors and the same aggregation settings

The run covers:

```text
2 training sizes × 2 mechanisms × 2 dtypes
× 3 seeds × 3 repetitions × 5 methods = 360 workers
```

Seeds are `101`, `202`, and `303`. Each measurement uses a fresh,
sequential worker with a separate small untimed warmup. Method order
rotates across seeds and repetitions.

Fit and first transform are timed consecutively. Loading, preprocessing,
scoring, and validation are outside the measured intervals. This study
does not measure repeated transforms or same-data `fit_transform()`.

## Timing

Each timing cell is the median of nine runs: three seeds with three
repetitions each. Brackets show the observed minimum and maximum,
not confidence intervals.

### First transform

Times are milliseconds.

| Train rows | Missingness | Dtype | KNNImputer | Complete | Available |
| ---: | --- | --- | ---: | ---: | ---: |
| 10,000 | MCAR | float32 | 401.24 [389.51–415.33] | 49.49 [48.58–51.47] | 228.42 [215.55–232.95] |
| 10,000 | MCAR | float64 | 462.86 [459.63–482.54] | 50.37 [49.29–52.34] | 336.06 [332.60–344.08] |
| 10,000 | MAR | float32 | 390.52 [383.11–405.27] | 51.55 [50.45–56.35] | 215.85 [210.35–223.36] |
| 10,000 | MAR | float64 | 452.12 [445.48–472.62] | 52.56 [51.38–57.34] | 326.71 [315.98–334.31] |
| 15,000 | MCAR | float32 | 618.56 [597.73–643.98] | 65.98 [65.61–68.72] | 342.53 [337.80–346.91] |
| 15,000 | MCAR | float64 | 726.64 [715.28–732.59] | 67.04 [66.24–69.25] | 452.25 [448.25–462.36] |
| 15,000 | MAR | float32 | 600.14 [586.15–650.68] | 68.98 [67.86–74.78] | 326.95 [319.36–344.13] |
| 15,000 | MAR | float64 | 697.39 [689.82–727.67] | 69.85 [68.79–75.16] | 433.98 [428.54–451.25] |

### Fit plus first transform

Times are median milliseconds. Speed ratios are KNNImputer time divided
by the corresponding FaissImputer time.

| Train rows | Missingness | Dtype | KNNImputer | Complete | Available | Complete speed | Available speed |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 | MCAR | float32 | 401.99 | 51.14 | 231.41 | 7.86× | 1.74× |
| 10,000 | MCAR | float64 | 463.68 | 52.08 | 339.51 | 8.90× | 1.37× |
| 10,000 | MAR | float32 | 391.29 | 53.17 | 218.88 | 7.36× | 1.79× |
| 10,000 | MAR | float64 | 452.97 | 54.25 | 330.21 | 8.35× | 1.37× |
| 15,000 | MCAR | float32 | 619.51 | 68.25 | 346.76 | 9.08× | 1.79× |
| 15,000 | MCAR | float64 | 727.64 | 69.57 | 456.87 | 10.46× | 1.59× |
| 15,000 | MAR | float32 | 601.06 | 71.36 | 331.09 | 8.42× | 1.82× |
| 15,000 | MAR | float64 | 698.36 | 72.25 | 438.40 | 9.67× | 1.59× |

Fit itself was slower for FaissImputer. Condition-level fit medians were
0.77–0.98 ms for KNNImputer, 1.65–2.36 ms for complete mode, and
2.98–4.54 ms for available mode.

Simple mean and median imputation were substantially cheaper:
fit-plus-transform medians ranged from 2.57–3.14 ms and 4.86–7.48 ms,
respectively. Their reconstruction errors are included below.

## Process peak memory

Values are median worker peak RSS in MiB.

| Train rows | Missingness | Dtype | KNNImputer | Complete | Available |
| ---: | --- | --- | ---: | ---: | ---: |
| 10,000 | MCAR | float32 | 366.3 | 145.3 | 184.2 |
| 10,000 | MCAR | float64 | 421.7 | 146.4 | 185.2 |
| 10,000 | MAR | float32 | 352.2 | 147.3 | 184.2 |
| 10,000 | MAR | float64 | 406.1 | 148.9 | 185.2 |
| 15,000 | MCAR | float32 | 477.0 | 149.8 | 206.7 |
| 15,000 | MCAR | float64 | 562.3 | 150.8 | 208.1 |
| 15,000 | MAR | float32 | 444.9 | 151.6 | 206.7 |
| 15,000 | MAR | float64 | 538.5 | 152.7 | 208.1 |

RSS includes interpreter and library overhead, dataset preparation,
warmup, fitting, transformation, validation, and comparison-payload
preparation. It is sampled before JSON serialization.

These values are not retained-model memory or transform-only peaks.
Simple-imputation peak RSS medians were approximately 145–153 MiB.

## Reconstruction quality

Errors are measured only at artificially hidden entries in held-out
queries, against the original float64 truth after training-fitted
standardization. Lower values are better.

The table shows float64 **RMSE / MAE**, with each metric summarized
by its median across the three seeds. Repeated timings do not add
independent quality observations.

| Train rows | Missingness | Mean baseline | Median baseline | KNNImputer | Complete | Available |
| ---: | --- | --- | --- | --- | --- | --- |
| 10,000 | MCAR | 0.900855 / 0.610832 | 0.958816 / 0.578476 | 0.771187 / 0.433458 | 0.648734 / 0.336981 | 0.771187 / 0.433458 |
| 10,000 | MAR | 0.942527 / 0.612844 | 0.997604 / 0.584962 | 0.804430 / 0.453292 | 0.635186 / 0.347323 | 0.804430 / 0.453292 |
| 15,000 | MCAR | 0.915218 / 0.599737 | 0.969070 / 0.574070 | 0.735273 / 0.424112 | 0.579649 / 0.324703 | 0.735273 / 0.424112 |
| 15,000 | MAR | 0.924906 / 0.606041 | 0.983528 / 0.576330 | 0.789150 / 0.448465 | 0.572256 / 0.329783 | 0.789150 / 0.448465 |

Complete mode had lower aggregate RMSE than KNNImputer in every matched
seed and condition in this run. Its condition-level RMSE medians were
15.9–27.5% lower across both dtypes.

This differs from the earlier synthetic
[same-data study](fit-transform-110e37bf.md), where complete mode had
higher RMSE. The datasets, feature counts, donor retention, and evaluation
protocols differ. These results do not establish a generally superior
donor policy.

The raw results include per-feature RMSE and MAE in both standardized
and original units. `MedInc` has zero scored cells and null error metrics
because it remains observed.

Aggregate improvements do not imply improvement for every feature.
Cross-size standardized errors also use different training-fitted scales.

## Agreement with KNNImputer

Output agreement is a separate measurement from reconstruction quality.

For float64 available mode, the maximum absolute difference from
KNNImputer over scored entries was at most `8.88e-16`. Recorded aggregate
RMSE values matched in all corresponding cases.

For float32 available mode, the maximum difference was at most
`4.77e-7` except for this case:

| Item | Value |
| --- | --- |
| Training rows | 15,000 |
| Query rows | 3,000 |
| Missingness | MAR |
| Dtype | float32 |
| Seed | 303 |
| Scored entries | 2,371 |
| Maximum absolute output difference | 0.612522468 standardized units |
| KNNImputer RMSE | 0.789031678 |
| Available-mode RMSE | 0.789150378 |
| KNNImputer MAE | 0.448263854 |
| Available-mode MAE | 0.448465015 |

The difference recurred in all three repetitions. Aggregate RMSE differed
by approximately `0.000118700`, and MAE by `0.000201161`.

A follow-up diagnostic reproduced the original float32 outputs.
Only four missing cells across two query rows differed by more than
`1e-5` between the implementations.

To isolate numerical precision, the diagnostic promoted the same
prepared float32 inputs to float64 without changing their values.
With these float64 inputs, both implementations produced identical
outputs.

| Implementation | Maximum output change from float32 to float64 input |
| --- | --- |
| FaissImputer, available mode | 1.61e-7 |
| KNNImputer | 0.6125 |

An independent check calculated missing-aware distances directly in
float64 and matched FaissImputer's neighbor selection. For the affected
rows, KNNImputer's float32 distance calculations changed the neighbor
ordering. Different donors therefore contributed to the averages,
explaining the output differences.

In this benchmark case, FaissImputer's float32 results stayed consistent
with the higher-precision reference within small rounding differences.
The discrepancy reflects a change in neighbor selection, rather than
a comparably large rounding difference in the final average.

Complete mode uses a different donor population and is not expected to
match KNNImputer outputs.

## Validation

- All 360 workers completed successfully; none failed, timed out, or
  remained unrun because of the time budget.
- Every worker passed finite-output, shape, dtype, observed-value,
  and input-preservation checks.
- Prepared-case metadata and fingerprints matched across methods and
  repetitions in all 24 cases.
- Output hashes were identical across the three repetitions within
  each of the 120 method/case groups.
- Dataset fingerprints, environment metadata, and native thread counts
  were consistent.
- Raw query rows and masks matched across training sizes and dtypes.
- The 40 summary groups agree with the recorded measurements.

These checks establish execution consistency. They do not establish
numerical identity between different methods.

## Reproduction

The exact workflow is available at
[source 8f289647](https://github.com/ScionKim/FaissImputer/blob/8f289647dcb8e7bd0c7e32ed29153db73a33dc26/.github/workflows/benchmark-real-data-coverage.yml).

The workflow artifact is named:

```text
real-data-coverage-35287620647-1
```

It contains the candidate wheel, `candidate-build.json`,
`dependencies.txt`, `environment.json`, `install-report.json`,
`dataset.json`, and the raw results. GitHub artifact retention is limited;
the raw results are also archived in this repository.

To reproduce using the artifact:

1. Obtain a clean source checkout at the recorded commit.
2. Extract the artifact and verify the wheel SHA256 shown above.
3. Use Python 3.12 and a fresh environment with the recorded dependencies.
4. Copy the benchmark scripts outside the source checkout so imports
   resolve to the installed candidate wheel.
5. Download the dataset once, then run the controller with the recorded
   settings.

The following Linux commands use three paths that must be supplied:

```bash
SOURCE=/absolute/path/to/checkout-at-8f289647
ARTIFACT=/absolute/path/to/extracted-artifact
WORK=/absolute/path/to/new-reproduction-directory

export GITHUB_SHA=8f289647dcb8e7bd0c7e32ed29153db73a33dc26
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$GITHUB_SHA"

mkdir -p "$WORK/run" "$WORK/data"
python3.12 -m venv "$WORK/venv"
BENCH_PY="$WORK/venv/bin/python"

"$BENCH_PY" -m pip install \
  --only-binary=:all: \
  -r "$ARTIFACT/dependencies.txt" \
  "$ARTIFACT/wheels/faiss_imputer-0.3.20+bench.8f289647dcb8-py3-none-any.whl"

"$BENCH_PY" -m pip check
cp -R "$SOURCE/benchmarks" "$WORK/run/benchmarks"
cd "$WORK/run"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export BENCH_DATA_HOME="$WORK/data"

"$BENCH_PY" - <<'PY'
import os

from benchmarks.benchmark_real_data_cases import load_dataset

load_dataset(
    os.environ["BENCH_DATA_HOME"],
    download_if_missing=True,
)
PY

"$BENCH_PY" -u -m benchmarks.benchmark_real_data_coverage \
  --expected-version 0.3.20+bench.8f289647dcb8 \
  --provenance "$ARTIFACT/candidate-build.json" \
  --data-home "$BENCH_DATA_HOME" \
  --train-sizes 10000 15000 \
  --query-size 3000 \
  --seeds 101 202 303 \
  --repeats 3 \
  --timeout-seconds 300 \
  --budget-seconds 5400 \
  --output "$WORK/real_data_coverage.json"
```

The pinned workflow also documents how to rebuild the candidate from
source if the original artifact is unavailable. A rebuilt wheel must
have its own recorded fingerprint.

New runs should retain their actual environment and provenance.
Timings on different hardware or dependency versions are new
measurements, not replacements for this archived run.

## Limits

This study covers one real feature dataset, artificial missingness,
three seeds, eight features, five neighbors, and one runner environment.

It does not evaluate naturally missing data, downstream prediction,
other datasets, callable metrics, approximate indexes, GPU execution,
or larger query workloads.

The float32 diagnosis is limited to this reproduced case and software
environment. It does not establish that FaissImputer is always more
accurate or that KNNImputer is generally incorrect. Agreement with a
higher-precision distance reference is separate from reconstruction
quality: KNNImputer had slightly lower aggregate reconstruction errors
in this case, as reported above.
Published-package performance claims require measurements of the
corresponding published wheel.