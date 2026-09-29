# Wine Quality float32 output diagnostic - 71a6e60

This diagnostic reproduces the recorded Wine Quality (white) float32 outputs and explains both masked entries whose absolute difference between KNNImputer and FaissImputer[available] exceeds 1e-5 in the selected case.

Both differences arise from reversed ordering of near-equal distances in the captured KNNImputer float32 calculation. Faiss selects the fifth neighbor identified by exact arithmetic on the prepared input values.

These findings concern neighbor selection in this case. They do not establish general reconstruction-quality superiority.

[Evidence archive](../../benchmarks/results/wine-quality-output-71a6e60.zip) | [Original benchmark report](real-data-datasets-ef04b1b.md) | [Diagnostic script](https://github.com/ScionKim/FaissImputer/blob/71a6e60b4885e7ac5e8b7a3a34e4b8b541e337b3/benchmarks/diagnose_wine_quality_output.py) | [Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/36506128325)

## Scope and provenance

| Field | Value |
| --- | --- |
| Library source commit | `ef04b1b0274d3abdd977440dd61e7d91e328c64d` |
| Diagnostic commit | `71a6e60b4885e7ac5e8b7a3a34e4b8b541e337b3` |
| Installed library version | `0.3.20+bench.ef04b1b0274d` |
| Dataset | Wine Quality (white), eleven numerical features |
| Training / held-out query rows | 3000 / 1000 |
| Missingness | MCAR, nominal overall rate 0.10 |
| Always-observed feature | alcohol |
| Input and output dtype | float32 |
| Ground-truth dtype | float64 |
| Seed / reference repeat | 101 / 1 |
| Neighbors / weights | 5 / uniform |
| Faiss configuration | available donors, l2, Flat, mean |
| Original benchmark CPU | AMD EPYC 9V74 80-Core Processor |
| Diagnostic CPU | AMD EPYC 7763 64-Core Processor |

The dataset is [Wine Quality, Cortez et al. (2009)](https://doi.org/10.24432/C56S3T), distributed under CC BY 4.0. The quality column is excluded. Scaling uses observed training values only.

The diagnostic uses Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, faiss-cpu 1.15.1, SciPy 1.18.1, threadpoolctl 3.7.0 and joblib 1.6.0. Recorded native thread pools use one thread.

The preserved baseline JSON and source dataset archive match the original files. The three installed FaissImputer source files match the original benchmark wheel's source files. Prepared input fingerprints and both output fingerprints match the original records.

The reference records are zero-based indices 62 for KNNImputer and 64 for FaissImputer[available] in `reference-benchmark.json`.

This is an output diagnostic. It adds no timing measurements or speedup claims.

## Diagnostic method

All query and donor indices below are zero-based indices in the prepared arrays.

Models and tracing use the original float32 inputs. The archived inputs, outputs and captured KNN distances retain their original dtypes. Output differences and reconstruction errors are calculated after promoting the outputs to float64.

Each finite prepared float32 value is converted losslessly to a Python float and then exactly to a rational number using `Fraction.from_float`. For a query and donor sharing `m` observed features, the independent reference evaluates:

    squared_distance = (11 / m) * sum((query[j] - donor[j]) ** 2)

The sum includes only jointly observed coordinates, and the entire reference calculation uses rational arithmetic. These are exact distances between the represented, prepared binary32 values, not idealized distances before scaling or float32 conversion.

A donor is eligible for an imputed feature when its target value is observed and it shares at least one observed feature with the query.

An admissible exact top-5 selection includes every eligible donor strictly closer than the fifth-neighbor boundary and fills the remaining slots from donors exactly on that boundary. Floating-point equality in a captured distance array is evaluated separately from exact rational equality.

KNNImputer donor IDs are captured from the actual `argpartition` calls. Faiss donor IDs are reconstructed from the actual completed search results. Both traced transforms reproduce their corresponding untraced outputs.

Exact distances were independently recalculated for all 3000 donors of the affected query row. All seven masked entries in that row were examined. The query has no duplicate prepared query values and maps to one completed Faiss search. That search requested 16 candidates and applied precision refinement.

## Output differences

The comparison covers 1105 masked held-out entries. Absolute difference means:

    abs(float64(KNNImputer output) - float64(FaissImputer[available] output))

Two entries in query row 974 exceed 1e-5. The other 1103 entries have a maximum absolute difference of `2.384185791015625e-07`. Detailed row collection was not truncated.

The following values use unrounded round-trip decimal representations. They describe individual cells, not aggregated timing or quality statistics.

| Query row | Feature | KNNImputer output | FaissImputer[available] output | Absolute difference |
| ---: | --- | ---: | ---: | ---: |
| 974 | volatile acidity | -0.6151994466781616 | -0.8424705266952515 | 0.22727108001708984 |
| 974 | total sulfur dioxide | -1.044341802597046 | -0.7813240885734558 | 0.2630177140235901 |

### Near-equal distance ordering

Both affected features have a unique exact fifth neighbor: donor 138. KNN instead selects donor 1542, which is slightly farther under the exact prepared-input reference.

The exact squared distances are:

    donor 138:  792141977603119 / 13510798882111488
    donor 1542: 33005921127905 / 562949953421312

Their positive difference is:

    129466601 / 13510798882111488

Its float64 representation is `9.582453423343887e-09`.

The captured KNN distances reverse this ordering:

| Donor | Exact squared distance, float64 display | Captured KNN distance |
| ---: | ---: | ---: |
| 138 | 0.058630284153805855 | 0.24213826656341553 |
| 1542 | 0.058630293736259276 | 0.24213466048240662 |

The reference column contains squared distances. The captured KNN column contains unsquared distances.

Both donors share three observed coordinates with this query and have observed target values for the two affected features. Within each feature, the methods select the same other four donors:

| Feature | Shared selected donors | KNN fifth donor | Faiss fifth donor |
| --- | --- | ---: | ---: |
| volatile acidity | 2758, 896, 2478, 1574 | 1542 | 138 |
| total sulfur dioxide | 1158, 2758, 2478, 1574 | 1542 | 138 |

The different target values of the fifth donors produce the large differences in their imputed means.

All seven detailed features have a unique exact top-5 donor set. Faiss selections are exact-admissible in seven cases and KNN selections in five. All seven KNN selections are admissible under their own captured float32 distances.

### Aggregation residuals

For the fourteen examined method/cell selections, the maximum absolute difference between the stored output and the exact rational mean of its selected donor values is:

    13 / 167772160

Its float64 representation is `7.748603820800782e-08`.

These residuals include aggregation and final float32 output rounding. They are much smaller than the two large output differences, which are explained by donor selection.

The trace fields `mean_float32` and `mean_float64` are illustrative reductions of selected values. They do not capture the libraries' actual aggregation operations or reduction order.

The small differences in the other 1103 masked entries are reported as observations; their individual donor selections were not all traced.

## Reconstruction quality

RMSE and MAE are computed against held-out float64 ground truth over the same 1105 masked entries, in units standardized using observed training values.

These are single-case metrics computed from the archived output arrays. They are not medians across seeds or timing repeats.

| Method | RMSE | MAE |
| --- | ---: | ---: |
| KNNImputer | 0.6941226105750372 | 0.49915837710654554 |
| FaissImputer[available] | 0.6943968042774052 | 0.49919072773235573 |

KNNImputer has slightly lower RMSE and MAE in this case. Agreement with the exact neighbor-distance reference is distinct from reconstruction quality against ground truth.

Similar aggregate reconstruction errors do not establish equality of individual imputed values or algorithmic equivalence.

## Evidence and reproducibility

The evidence ZIP preserves:

- `wine_quality_output_diagnostic.json`: provenance, reproduction checks, actual neighbor traces, exact rational distances and means, and quality metrics.
- `wine_quality_output_diagnostic.npz`: prepared training/query arrays, ground truth, missing mask, both full outputs, traced query indices and captured KNN distance rows.
- `reference-benchmark.json`: the original benchmark records.
- `candidate-build.json`, dependency and installation records, and the rebuilt library wheel.
- `dataset.json` and the original `source-data/wine_quality_white.zip`.

Evidence ZIP SHA-256:

    1dec7440e7523b0b03352fec77ab0695444da86e0be20f0614f660e7d884d276

Output-array NPZ SHA-256:

    a54071b03908dd7688c87035dfccc2247c7cf405744c9bb1dd28ed8d6240305f

The report can be checked from these files without running either imputer again:

1. Load the NPZ with `allow_pickle=False`, preserving the stored array dtypes.
2. Promote both output arrays to float64 before subtraction. On `missing`, count absolute differences exceeding 1e-5 and their distinct query rows.
3. Recompute RMSE as `sqrt(mean((output - truth) ** 2))` and MAE as `mean(abs(output - truth))`, using promoted outputs and restricting both calculations to `missing`.
4. Recompute rational distances from the original prepared float32 values and apply feature-specific donor eligibility.
5. Compare exact boundaries and admissible selections with `rows[].features[].exact_reference`, `knn_observation` and `faiss_selections`.
6. Recompute exact selected-donor means and subtract them from the exact rational values of the stored outputs. Compare with `output_minus_exact_mean`.

The existing Real-data output diagnostic workflow supports `diagnostic = wine_quality_float32`. It fixes the library source to the original benchmark commit while using diagnostic code from the selected workflow branch. Commit `71a6e60b4885e7ac5e8b7a3a34e4b8b541e337b3` identifies the diagnostic code that produced this evidence.

These findings explain the selected Wine Quality float32 case. Other seeds, configurations and datasets require their own evidence before receiving the same explanation.