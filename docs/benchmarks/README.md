# Benchmark reports

[Project README](../../README.md#performance)
· [API reference](../api.md)
· [Usage examples](../usage.md)

## Latest release comparison

### FaissImputer 0.3.20

Comparison of published FaissImputer 0.3.20, FaissImputer 0.3.19,
and KNNImputer with complete and available donor policies and
float32/float64 inputs.

On the measured available-donor workload, 0.3.20 achieved a
2.71–2.73× first-transform speedup over KNNImputer.
Fit itself was slower; fit plus first transform was 2.44–2.49× as fast.

The report covers fit time, first and repeated transforms, timing
variation, process memory, output agreement, error against synthetic
ground truth, and reproduction instructions.

[Read the report](released_versions_0.3.20.md)
· [Raw measurements](../../benchmarks/results/released_versions_0.3.20.json)

## Historical version comparisons

| Report | Coverage |
| --- | --- |
| [FaissImputer 0.3.19](released_versions_0.3.19.md) | Comparison with released 0.3.16 and KNNImputer under both donor policies and input dtypes, including timing, process memory, output agreement, and synthetic-data quality. |
| [FaissImputer 0.3.10](released_versions_0.3.10.md) | Released-package comparisons with KNNImputer: an AMD 20,000-row run and an Intel training-size sweep. Includes tables, conditions, output agreement, and memory observations. |
| [FaissImputer 0.2.0](v0.2.0.md) | Historical measurements of the earlier complete-donor-only implementation. |

Speedup figures depend on whether fit is included. The 0.3.20 and
0.3.19 reports provide both first-transform and fit-plus-first-transform
timings. The historical 0.3.10 speedup tables measure the first transform
only.

## Workload and implementation experiments

| Report | Focus |
| --- | --- |
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