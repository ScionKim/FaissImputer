# Benchmark reports

[Project README](../../README.md#benchmarks)
· [API reference](../api.md)
· [Usage examples](../usage.md)

## Latest release comparison

### FaissImputer 0.3.19

Comparison of released FaissImputer 0.3.19, FaissImputer 0.3.16,
and KNNImputer with complete and available donor policies and
float32/float64 inputs.

The report covers fit time, first and repeated transforms, process memory,
output agreement, and error against synthetic ground truth.

[Read the report](released_versions_0.3.19.md)
· [Raw measurements](../../benchmarks/results/released_versions_0.3.19.json)

## Historical version comparisons

| Report | Coverage |
| --- | --- |
| [FaissImputer 0.3.10](released_versions_0.3.10.md) | Released-package comparisons with KNNImputer: an AMD 20,000-row run and an Intel training-size sweep. Includes tables, conditions, output agreement, and memory observations. |
| [FaissImputer 0.2.0](v0.2.0.md) | Historical measurements of the earlier complete-donor-only implementation. |

The 0.3.19 report presents fit-plus-first-transform totals as well as
individual timings. The 0.3.10 speedup tables measure the first transform
only. Their headline speedups use different timing definitions.

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