# Benchmark reports

[Project README](../../README.md#performance)
· [API reference](../api.md)
· [Usage examples](../usage.md)

## Latest release comparison

### FaissImputer 0.3.22

Comparison of published FaissImputer 0.3.22, FaissImputer 0.3.21,
and KNNImputer on one Intel Xeon Platinum 8370C runner, with 20,000
training rows and 300 held-out queries. Both donor policies and
float32/float64 inputs are reported separately. Complete-donor cases
use fully observed training data; available-donor cases use 10% MCAR
training missingness.

For available-donor float64, the median matched 0.3.21/0.3.22 timing
ratio was 1.106× for the first transform and 1.101× for fit plus first
transform. All nine first-transform pairs were faster in 0.3.22.
First-transform ratios for the other policy/dtype combinations were
close to 1. Available-mode whole-worker peak RSS medians were higher
than the corresponding KNNImputer baseline for both dtypes.

All 108 workers passed their checks. Recorded output hashes matched
between 0.3.21 and 0.3.22 in all 36 paired comparisons. This observation
is limited to the measured inputs and does not establish general
prediction equivalence.

Timings summarize nine workers per version, policy, and dtype:
three seeds and three repeats. Speedups are medians of matched
record-level timing ratios, not ratios of separately reported medians.
The report covers first and repeated transforms, fit time, timing
variation, whole-worker memory, reconstruction error, and recorded
output agreement.

A standard-library analysis script regenerates the report and
full-precision summary from the preserved ZIP, including environment
files. This Intel run and the earlier AMD 0.3.21 run are separate
experiments; differences between those runs do not establish a
version-to-version performance change.

[Read the report](released_versions_0.3.22.md)
· [Original benchmark evidence](../../benchmarks/results/released_versions_0.3.22.zip)
· [Full-precision summary](../../benchmarks/results/released_versions_0.3.22-summary.json)

## Historical version comparisons

| Report | Coverage |
| --- | --- |
| [FaissImputer 0.3.21](released_versions_0.3.21.md) | Comparison with released 0.3.20 and KNNImputer on an AMD EPYC 7763 runner, separating donor policies and dtypes and reporting timing, memory, reconstruction error, and output agreement. |
| [FaissImputer 0.3.20](released_versions_0.3.20.md) | Comparison with released 0.3.19 and KNNImputer under both donor policies and input dtypes, including timing, process memory, output agreement, and synthetic-data quality. |
| [FaissImputer 0.3.19](released_versions_0.3.19.md) | Comparison with released 0.3.16 and KNNImputer under both donor policies and input dtypes, including timing, process memory, output agreement, and synthetic-data quality. |
| [FaissImputer 0.3.10](released_versions_0.3.10.md) | Released-package comparisons with KNNImputer: an AMD 20,000-row run and an Intel training-size sweep. Includes tables, conditions, output agreement, and memory observations. |
| [FaissImputer 0.2.0](v0.2.0.md) | Historical measurements of the earlier complete-donor-only implementation. |

Speedup figures depend on whether fit is included. The 0.3.22, 0.3.21,
0.3.20, and 0.3.19 reports provide both first-transform and
fit-plus-first-transform timings. The historical 0.3.10 speedup tables
measure the first transform only. Use same-run paired measurements
for version-to-version claims.

## Workload and implementation experiments

| Report | Focus |
| --- | --- |
| [Available float64 selected-distance batching](available-selected-distances-c02b71d.md) | Matched before/after timings, output agreement, worker peak RSS, and a same-code control examining the complete float64 third-call discrepancy. |
| [Available-donor batching and threads](available-batching-90c8cfb8.md) | Batch memory budgets, thread counts, repeated 500,000-row measurements, and a million-row pilot. |
| [Available-donor candidate expansion](available-expansion-0.3.8.md) | Comparison of 0.3.7 and 0.3.8 available-donor candidate expansion. |
| [Complete-aggregation recovery](complete-aggregation-e5ef482.md) | Source comparison between 0.3.8 and the implementation prepared for 0.3.10, across query and feature shapes. |
| [Complete-donor query patterns](complete-patterns-9d179b2b.md) | Effects of training size and query missingness patterns. |
| [Real-data imputation](real-data-a3bd1ce3.md) | A small-data comparison under MCAR and selected MAR missingness, including imputation quality. |
| [Partial-donor development snapshot](partial-donors-0a3cc077.md) | Historical measurements for commit `0a3cc077` during partial-donor development. |

## Code and archived data

- [Benchmark code](../../benchmarks)
- [Archived result files](../../benchmarks/results)

Reports provide readable tables and explanations. Archived JSON and ZIP
files contain the underlying measurements.

Compare hardware, dependencies, inputs, donor policies, and timing
definitions before drawing conclusions across reports. Results from
different environments do not establish a version-to-version performance
change.

## Development benchmarks

- [Combined available-donor optimizations (`b8e5a0f3`)](available-optimizations-b8e5a0f3.md):
  Direct comparison with PyPI 0.3.19 showed 50–54% less transform time
  and 13.0–14.5% lower peak process RSS in the measured workload,
  with identical imputation outputs.

- [Available-donor search memory (`7c62738b`)](available-distance-memory-7c62738b.md):
  Compared with the earlier candidate `94adbf66`, peak process RSS decreased
  by approximately 13% and transform time by 13–15% in the measured workload,
  with identical imputation outputs.

- [Available-donor distance buffers (`94adbf66`)](available-distance-buffers-94adbf66.md):
  Compared with released 0.3.19, available-donor transform time decreased
  by 38–41% in the measured workload, with identical imputation outputs.

## Unreleased candidate benchmarks

These reports identify development builds by source commit.
Published-package comparisons are listed separately.

### Query-count studies

These studies cover 20,000 training rows and 300, 1,000, and 3,000 queries,
using both donor policies and float32/float64 inputs.

- [Query-count sweep: 0772effd versus 0.3.20](query-count-sweep-0772effd.md).
  Complete-donor first-transform time increased by 10.0–13.0%;
  available-donor time changed by less than 0.5%.

- [Distance-check optimization: fde59b1d versus 0772effd](complete-distance-checks-fde59b1d.md).
  The follow-up optimization reduced complete-donor first-transform time
  by 7.6–9.1%; available-donor time changed by less than 0.7%.

Each query-count study completed 324 workers with output and
input-preservation checks. Candidate outputs matched the corresponding
FaissImputer baseline outputs.

### Same-data fit_transform

[Same-data comparison: source 110e37bf](fit-transform-110e37bf.md)
compares `fit_transform(X)` with `fit(X)` followed by `transform(X)`
for both donor policies and KNNImputer.

Primary conclusions use **10,000 and 20,000 rows**:

- Available mode completed `fit_transform()` 1.72–2.18× as fast as
  KNNImputer, with closely matching aggregate reconstruction errors.
- Available-mode process peak RSS medians were 55.4–71.8% lower.
  These measurements are not retained-model memory.
- Complete mode was faster than available mode but used only about 12% of rows
  as donors and had 38.2–41.7% higher median reconstruction RMSE.

The 3,000-row case shows the crossover region. The 1,000-row case is
retained only for small-data regression tracking.

[Phase-separated memory: source ac84bfa1abec](fit-transform-ac84bfa1abec-phase-memory.md)
distinguishes retained fitted memory and fit/transform phase-peak RSS
from whole-process peak RSS for the same workload. Retained fitted
memory is 0.3–3.8% of peak RSS at 20,000 rows; transform dominates
peak RSS for KNNImputer and available-donor.

All 432 runs passed output and input-preservation checks. Within each
method, both APIs produced identical outputs, with primary-workload
total-time medians differing by less than 1%.

The report includes conditions, observed timing ranges, quality and
memory measurements, limitations, reproduction instructions, and raw results.

### Held-out real-data imputation

[California Housing comparison: source 8f289647](real-data-coverage-8f289647.md)
covers 10,000 and 15,000 training rows with 3,000 held-out queries,
MCAR/MAR missingness, and float32/float64 inputs. Methods include
mean and median baselines, KNNImputer, and both FaissImputer donor policies.

Compared with KNNImputer, using condition-level medians:

- Available mode completed fit plus first transform 1.37–1.82× as fast,
  with 47.7–63.0% lower process peak RSS.
- Complete mode completed fit plus first transform 7.36–10.46× as fast,
  with 15.9–27.5% lower reconstruction RMSE on this dataset.
  Complete donors represented 42.59–47.60% of training rows.

All 360 runs passed output and input-preservation checks.

A follow-up diagnostic reproduced the float32 discrepancy: four missing
cells across two rows differed by more than `1e-5`, with a maximum
difference of 0.6125 standardized units.

In this case, FaissImputer's float32 results stayed consistent with an
independent float64 direct-distance reference, while KNNImputer's float32
distance calculations changed the neighbor ordering. Both implementations
agreed when the same inputs were promoted to float64.

This diagnosis is specific to the reproduced case and does not establish
a general advantage in reconstruction accuracy.

The report includes timing variation, simple-baseline quality,
memory measurements, reproduction instructions, and raw results.

## Same-data OFAT sweep — 9683d03

[Report](fit-transform-ofat-9683d03.md) ·
[Raw benchmark JSON files](../../benchmarks/results/ofat-2026-09-22/) ·
[Full-precision analysis summary](../../benchmarks/results/ofat-2026-09-22-summary.json) ·
[Analysis script](../../benchmarks/analyze_ofat.py)

Twelve runs compare KNNImputer with FaissImputer's complete and available
donor policies across row counts, feature counts, missing rates,
missingness patterns, and neighbor counts.

Results separate `fit_transform` from `fit_then_transform`, and float32
from float64. Timings are median [min–max]; speedups are medians of
matched record-level KNN/Faiss timing ratios.

The Intel k=30 observation and Intel 50,000-row stress result are reported
separately from the AMD sweeps. The neighbors sweep also varies the
number of guaranteed complete rows with k.

RMSE and MAE describe reconstruction error against ground truth.
Close aggregate metrics do not establish identical predictions or
algorithmic equivalence.

## Same-data callable metrics — 76e0230

[Report](fit-transform-callable-76e0230.md) ·
[Raw benchmark JSON](../../benchmarks/results/fit-transform-callable-76e0230.json) ·
[Full-precision analysis summary](../../benchmarks/results/fit-transform-callable-76e0230-summary.json) ·
[Analysis script](../../benchmarks/analyze_callable_metrics.py)

Compares built-in metrics with a shared Python nan-Euclidean callable
on 300 and 1,000 training rows, using KNNImputer and both FaissImputer
donor policies. Both metric modes were measured on the same Intel
Xeon Platinum 8370C runner.

Results separate `fit_transform` from `fit_then_transform`, and float32
from float64. Timings are median [min–max] across three seeds and three
repeats. KNN/Faiss speedups and callable/builtin time multipliers are
medians of matched record-level ratios.

Under these tested callable conditions, available-donor mode was slower
than KNNImputer. Complete-donor mode was faster, with fewer eligible
donors and higher reconstruction error. Callable metrics do not build
a Faiss index.

RMSE and MAE measure reconstruction error against hidden ground truth.
Quality summaries use three seed datasets after verifying consistency
across APIs and repeats. Similar aggregate errors do not establish
identical predictions or algorithmic equivalence.

## Held-out Wine Quality and Abalone — ef04b1b

[Report](real-data-datasets-ef04b1b.md) ·
[Raw benchmark evidence](../../benchmarks/results/real-data-datasets-ef04b1b/) ·
[Full-precision analysis summary](../../benchmarks/results/real-data-datasets-ef04b1b-summary.json) ·
[Analysis script](../../benchmarks/analyze_real_data_datasets.py)

Compares mean and median baselines, KNNImputer, and both FaissImputer
donor policies on Wine Quality (white) and Abalone. Each dataset uses
1,000 and 3,000 training rows, 1,000 held-out queries, MCAR/MAR
missingness, and float32/float64 inputs. All 720 worker records passed
the benchmark checks.

Datasets and dtypes are reported separately. Fit-plus-first-transform
time and whole-worker peak RSS are median [min–max] across three seeds
and three repeats. KNN/Faiss speedups are medians of nine matched
record-level total-time ratios.

Available mode was slower than KNNImputer in Wine Quality and in
Abalone float64, but faster in Abalone float32. Complete mode was
faster on both datasets; reconstruction quality and donor counts
are reported separately.

RMSE and MAE measure error on masked held-out query entries against
ground truth in standardized units. Quality summaries use three seed
datasets after verifying consistency across repeats. Per-feature
errors in standardized and source units are preserved in the
full-precision summary.

The report separately records differences between imputed outputs,
including available-mode differences from KNNImputer in Abalone
float64. Similar aggregate reconstruction errors do not establish
identical predictions or algorithmic equivalence.

## Abalone float64 output diagnostic - 1969f4b

[Report](abalone-output-1969f4b.md) | [Evidence archive](../../benchmarks/results/abalone-output-1969f4b.zip) | [Diagnostic script](../../benchmarks/diagnose_abalone_output.py)

Reproduces the original outputs for Abalone with 3000 training rows, 1000 held-out queries, MAR, float64 and seed 303.

Independent exact arithmetic on the prepared binary64 inputs explains all nine masked entries with output differences above 1e-5: six involve different admissible selections among exact boundary ties, and three involve near-equal distance ordering reversals in the captured KNNImputer calculation.

The findings apply to this case. They do not establish prediction equivalence, general reconstruction-quality superiority, or an explanation for other benchmark cases.

## Wine Quality float32 output diagnostic - 71a6e60

[Report](wine-quality-output-71a6e60.md) | [Evidence archive](../../benchmarks/results/wine-quality-output-71a6e60.zip) | [Diagnostic script](../../benchmarks/diagnose_wine_quality_output.py)

Reproduces the original outputs for Wine Quality (white) with 3000 training rows, 1000 held-out queries, MCAR, float32 and seed 101.

Independent exact arithmetic on the prepared binary32 inputs explains both masked entries with output differences above 1e-5. The captured KNNImputer float32 distances reverse the ordering of two near-equal candidates, while Faiss selects the exact fifth neighbor.

These findings concern the examined case. Exact-neighbor agreement is distinct from reconstruction quality: KNNImputer has slightly lower RMSE and MAE in this case.