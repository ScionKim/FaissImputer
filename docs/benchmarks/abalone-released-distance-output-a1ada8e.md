# Abalone released 0.3.22: distance-weighted output diagnostic

[Benchmark index](README.md) · [Published distance-weighted Abalone comparison](released_abalone_distance_0.3.22.md)

The published Abalone comparison recorded differences between KNNImputer and
FaissImputer 0.3.22 with `weights="distance"`. The successful diagnostics
reproduce both methods at seed 101 and trace every masked entry differing
by more than `1e-5`: 21 entries for float32 and seven for float64.
The dtypes are separate cases, and their entry counts are not a count of
unique positions across both dtypes. No timing measurements were taken.

The observations distinguish exact boundary ties, distance-order reversals,
computed ties between unequal exact distances, and distance errors propagated
into inverse-distance weights. Every affected Faiss selection is admissible
under exact arithmetic on the represented inputs. This statement is limited
to the traced entries; it does not establish prediction equivalence,
all-input correctness, or general reconstruction-quality superiority.

## Configuration and provenance

- Published FaissImputer 0.3.22 and scikit-learn KNNImputer 1.9.1.
- Abalone numerical features; 3,000 training rows, 1,000 held-out query rows,
  seven features, seed 101, 10% target overall MCAR missingness.
- `Length` remains observed; `Sex` and `Rings` are excluded.
- Available donors, five neighbors, distance weights, mean aggregation,
  Flat index, one native thread, 256 MiB scikit-learn working memory.
- Python 3.12.14, NumPy 2.5.3, Faiss 1.15.1 and SciPy 1.18.1.
- Original benchmark source: `cd8117708f0bb43a2f79e15adf1e2e543ec6cda3`,
  [run 37241332456, attempt 1](https://github.com/ScionKim/FaissImputer/actions/runs/37241332456/attempts/1).
  This identifies the benchmark scripts, not the source revision of the installed wheel.

| Observation | Diagnostic source | Runner | Preserved artifact | GitHub run |
| --- | --- | --- | --- | --- |
| float32, reproduced | `a1ada8ee8c81240044b2a1b1d674e2abe464f72a` | AMD EPYC 7763 64-Core Processor | [ZIP](../../benchmarks/results/abalone-released-distance-output-a1ada8e-float32.zip) | [37274635203](https://github.com/ScionKim/FaissImputer/actions/runs/37274635203/attempts/1) |
| float64, reproduced | `198314643260fa7cd1ebccdcbd14ffe3954997ae` | AMD EPYC 7763 64-Core Processor | [ZIP](../../benchmarks/results/abalone-released-distance-output-1983146-float64.zip) | [37272999709](https://github.com/ScionKim/FaissImputer/actions/runs/37272999709/attempts/1) |
| Earlier float32, not reproduced | `198314643260fa7cd1ebccdcbd14ffe3954997ae` | AMD EPYC 9V74 80-Core Processor | [ZIP](../../benchmarks/results/abalone-released-distance-output-1983146-float32-mismatch.zip) | [37272212675](https://github.com/ScionKim/FaissImputer/actions/runs/37272212675/attempts/1) |

The successful diagnostics and original benchmark used the AMD EPYC 7763
CPU model, in separate workflow runs. The earlier float32 diagnostic used
AMD EPYC 9V74 and is retained below as a reproduction mismatch. These
observations do not isolate a hardware cause.

All three artifacts retain the published wheel, installation report,
dependency versions, original benchmark ZIP, reference JSON, source dataset,
prepared arrays and outputs. Successful artifacts also retain the detailed
selections, weights and captured KNN distance rows. The wheel SHA-256 is
`3343bd281e08a27869dc917f5e1479767ce5574be91a8d0f61247bdc728f62c3`.
Installed core files were checked against that wheel. The original benchmark
did not retain its wheel hash, so binary identity with that installation
is not established solely by the package version.

## Reproduction and scope

Both successful runs have `status="ok"`, `inputs_reproduced=true`,
`original_reproduced=true` and `traced_outputs_unchanged=true`. Each method
matches its archived full-output hash and all 711 saved masked-entry values.
Record indices 0 and 2 select `knn` and `current`, respectively; this
diagnostic does not trace the previous 0.3.21 release. Instrumented outputs
match the uninstrumented outputs exactly.

The following counts refer to one dtype and seed 101. Affected entries have
`abs(float64(Faiss output) - float64(KNN output)) > 1e-5` in standardized
units. Detailed rows include all missing features in those rows, so the
number of traced entries exceeds the number above the threshold. No medians
or pooling across seeds, repeats, dtypes or diagnostic attempts are used.

| Dtype | Scored entries | Entries above threshold | Affected rows | Traced entries | Maximum absolute output difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 711 | 21 | 13 | 31 | 0.07994997501373291 |
| float64 | 711 | 7 | 4 | 13 | 0.15553150590965467 |

Independent arithmetic on the saved arrays confirms the input and output
fingerprints, masked values, threshold counts and reconstruction metrics.
Exact rational distances, selection admissibility and weighted-reference
calculations were also checked against the stored observations. No imputer
was rerun for this review.

## Distance and weighting definitions

The exact squared-distance reference is
`(7 / shared_feature_count) * sum((query_j - donor_j)**2)` over shared
observed features. It uses exact rational arithmetic on the represented
binary32 or binary64 prepared values, not values before preprocessing.
An eligible donor contains the target feature. An admissible set includes
every eligible donor strictly closer than the exact fifth-neighbor distance
and fills the remaining slots with donors at that boundary.

KNN selections and weights are observed inside its actual imputation calls.
Faiss selections and weights are captured at the actual aggregation call,
using the batch row and feature to identify the output cell. The traces
preserve the original outputs.

Three arithmetic references separate the stages:

1. Exact rational squared distances, followed by Decimal square roots and
   inverse-distance weighting at 80 and 120 digits. These are rounded
   references, not certified exact weighted means.
2. The same high-precision weighting applied to captured unsquared distances,
   separating distance error from weight construction.
3. Exact rational weighted means of the captured floating-point target values
   and weights, separating aggregation and output assignment from the earlier stages.

When any selected distance is zero, only selected zero-distance donors
contribute, with equal normalized weights. A spurious positive distance
can therefore change which donors contribute, even if the selected IDs agree.
Agreement of the 80- and 120-digit results is a precision comparison, not
a proof or an equivalence test.

## Findings by mechanism

The counts below partition the entries above `1e-5`, separately by dtype.
Exact and computed ties are deliberately distinct.

| Mechanism | float32 | float64 |
| --- | ---: | ---: |
| Exact boundary tie | 1 | 1 |
| Strict order reversal | 1 | 3 |
| Computed distance tie | 2 | 0 |
| Zero-distance weighting | 6 | 3 |
| Positive-distance weighting | 11 | 0 |
| Total | 21 | 7 |

- **Exact boundary tie:** the methods choose different fifth donors with
  identical exact distances. Both selections are admissible. This occurs
  at float32 row 185 / `Shucked_weight` and float64 row 984 / `Shell_weight`.
- **Strict order reversal:** the captured KNN distances rank the farther
  boundary donor first. This affects float32 row 984 / `Shell_weight` and
  three float64 entries at rows 185 and 845.
- **Computed distance tie:** at float32 row 845, donors 1087 and 2312 have
  different exact distances, but KNN records both as `0.09992323070764542`.
  KNN selects donor 1087; Faiss selects the strictly closer donor 2312.
  This affects `Whole_weight` and `Shucked_weight`. It is not an exact tie
  and not a strict reversal of the captured distances.
- **Zero-distance weighting:** both methods select the same donor set, but
  KNN records a positive distance for an exact zero-distance donor. Six
  float32 entries and three float64 entries have different zero/nonzero
  weighting as a result.
- **Positive-distance weighting:** eleven float32 entries use the same
  admissible donors with positive exact distances. Errors in captured
  distances change inverse-distance weights enough to exceed the threshold.
  Weight construction and aggregation residuals are smaller in these entries.

Tie and ordering classifications can change with dtype because the exact
reference uses the actual prepared values of that dtype.

### Largest float64 difference

At row 536, donors 766 and 2952 have exact zero distance over their shared
observed features. KNN records donor 766 at `1.3938759963117321e-08` and
donor 2952 at zero, so only donor 2952 contributes. Faiss records both at
zero and gives each half of the normalized weight. For `Shell_weight`,
KNN returns `-0.13999762811991776` and Faiss returns `-0.2955291340295724`,
giving the absolute difference `0.15553150590965467`. The same mechanism
affects `Diameter` and `Whole_weight` in that row.

### Every entry above the threshold

Rows are zero-based indices in the prepared query array. Signed differences
are `Faiss output - KNN output`, with float32 values first promoted exactly
to float64. These display values use 12 significant digits; full unrounded
outputs and rational references remain in the diagnostic JSON.

| Dtype | Query row | Feature | Signed output difference | Mechanism |
| --- | ---: | --- | ---: | --- |
| float32 | 185 | Shucked_weight | +0.0799499750137 | Exact boundary tie |
| float32 | 266 | Shell_weight | +3.38554382324e-05 | Positive-distance weighting |
| float32 | 276 | Diameter | -0.000647008419037 | Zero-distance weighting |
| float32 | 276 | Shucked_weight | -0.00130695104599 | Zero-distance weighting |
| float32 | 285 | Diameter | -1.33514404297e-05 | Positive-distance weighting |
| float32 | 297 | Shell_weight | +3.07559967041e-05 | Positive-distance weighting |
| float32 | 303 | Diameter | +1.94907188416e-05 | Positive-distance weighting |
| float32 | 303 | Whole_weight | -1.43647193909e-05 | Positive-distance weighting |
| float32 | 352 | Diameter | -1.62720680237e-05 | Positive-distance weighting |
| float32 | 352 | Height | +0.000120759010315 | Positive-distance weighting |
| float32 | 492 | Viscera_weight | +1.31726264954e-05 | Positive-distance weighting |
| float32 | 604 | Height | -3.75509262085e-05 | Positive-distance weighting |
| float32 | 604 | Shucked_weight | -6.87837600708e-05 | Positive-distance weighting |
| float32 | 721 | Height | +0.000114500522614 | Positive-distance weighting |
| float32 | 845 | Whole_weight | -0.00760281085968 | Computed distance tie |
| float32 | 845 | Shucked_weight | -0.00320053100586 | Computed distance tie |
| float32 | 845 | Viscera_weight | -0.000870108604431 | Zero-distance weighting |
| float32 | 845 | Shell_weight | -0.003533244133 | Zero-distance weighting |
| float32 | 972 | Diameter | +0.00057265162468 | Zero-distance weighting |
| float32 | 972 | Whole_weight | -0.00217910297215 | Zero-distance weighting |
| float32 | 984 | Shell_weight | +0.0351462587714 | Strict order reversal |
| float64 | 185 | Shucked_weight | -0.0799492142787 | Strict order reversal |
| float64 | 536 | Diameter | -0.0504591665493 | Zero-distance weighting |
| float64 | 536 | Whole_weight | -0.140714357518 | Zero-distance weighting |
| float64 | 536 | Shell_weight | -0.15553150591 | Zero-distance weighting |
| float64 | 845 | Whole_weight | +0.00760347577059 | Strict order reversal |
| float64 | 845 | Shucked_weight | +0.00320188197897 | Strict order reversal |
| float64 | 984 | Shell_weight | -0.0351463748734 | Exact boundary tie |

For entries in this table, the largest absolute residual between the output
and the exact weighted mean of its captured floating-point values and weights
is `1.6777742704436618e-07` for float32 and
`3.215563768087608e-16` for float64. These residuals include output-dtype
assignment. Aggregation and assignment rounding alone do not explain the
differences above `1e-5`.

## Reconstruction error against ground truth

These are seed-101 errors across 711 masked held-out entries in standardized
units: `RMSE = sqrt(mean((output - truth)**2))` and
`MAE = mean(abs(output - truth))`. They are not medians across seeds or
repeats, and they do not measure agreement with an exact-neighbor reference.
The table uses the unrounded JSON values.

| Dtype | Method | RMSE | MAE |
| --- | --- | ---: | ---: |
| float32 | KNNImputer | 0.2754324098430359 | 0.17948065850707348 |
| float32 | FaissImputer 0.3.22 | 0.27550784304321424 | 0.1796286521158155 |
| float64 | KNNImputer | 0.2751931977365878 | 0.17953683898395867 |
| float64 | FaissImputer 0.3.22 | 0.2754407043565652 | 0.17948196716131387 |

KNN has lower RMSE and MAE for this float32 case. For float64, KNN has lower
RMSE while Faiss has lower MAE. Exact-neighbor admissibility does not imply
lower error against held-out truth. These observations establish neither
general quality superiority nor equality of individual imputed values.

## Earlier float32 reproduction mismatch

The earlier run 37272212675 reports `baseline_output_mismatch`. It reproduces
the inputs and the full Faiss output, but two of 711 KNN masked values differ
from the original. The changes are confined to row 845. The following values
are unrounded floats read from the original benchmark and saved diagnostic
arrays; the final reproduced run matches the original column.

| Feature | Original KNN value | Earlier mismatch-run KNN value | Absolute change |
| --- | ---: | ---: | ---: |
| Whole_weight | -1.192264437675476 | -1.199867606163025 | 0.007603168487548828 |
| Shucked_weight | -1.1745256185531616 | -1.1777273416519165 | 0.003201723098754883 |

The earlier run has 19 between-method entries above `1e-5`; the original
and final reproduced run have 21. The earlier diagnostic stopped before
donor tracing, so its selections and cause cannot be inferred from those
output values alone. It must not replace or be pooled with the original
benchmark results. CPU model and diagnostic revision differ between the
earlier and final float32 runs; this is not a controlled attribution of
the mismatch to hardware, BLAS dispatch or a specific arithmetic operation.

The diagnostic update permits distance-weight cases to retain current-run
observations after a mismatch, while preserving `original_reproduced=false`,
`status="baseline_output_mismatch"` and a nonzero exit. It also selects
changed-value rows and archived above-threshold rows for tracing. In the
final run both methods reproduce the original, so its traces apply to the
archived outputs. No mismatch check was relaxed to obtain that result.

## Reading and reproducing the evidence

Each artifact contains `abalone_released_0.3.22_distance_<dtype>_diagnostic.json`
and a matching NPZ. The JSON retains the full precision outputs, donor IDs,
exact squared distances, captured distances and weights, reference evaluations
and arithmetic residuals. The NPZ retains prepared arrays and outputs; the
successful runs also contain `traced_query_rows` and `knn_distance_rows`.

To recover the tables without new imputer measurements:

1. Verify the artifact checksums below and member hashes in `SHA256SUMS.txt`.
   Verify the NPZ against `arrays.sha256` in its diagnostic JSON.
2. Filter `rows[].features[]` by the stated difference threshold. Counts and
   signed differences come from `outputs.knn` and `outputs.faiss`.
3. Compare the actual donor sets in `knn_observation.training_row_indices`
   and `faiss_selections[0].training_row_indices`. Both admissible but
   different sets give an exact boundary-tie case. For an inadmissible KNN
   selection, compare `donor_details` exact fractions and captured KNN
   distances to distinguish a strict reversal from a computed tie.
4. For equal sets, compare `exact_squared_distances` with
   `weight_observation.weight_input_distances` to identify zero-distance
   changes. The remaining positive-distance cases are checked using
   `distance_weighted_reference`, `captured_distance_weighting_reference`
   and `captured_weight_arithmetic`. Residual maxima use
   `assigned_output_minus_exact_weighted_mean` only for the above-threshold entries.
5. Read the seed-level `quality` values, or recompute them from NPZ outputs
   and `truth` at `missing`. The earlier mismatch table compares its KNN
   output at those mask positions with `reference-benchmark.json` record 0
   `imputed_values`; the final run provides an independent matching copy.

The [diagnostic script](../../benchmarks/diagnose_abalone_output.py) runs through
the [Real-data output diagnostic workflow](../../.github/workflows/diagnose-real-data-float32.yml)
with `abalone_released_distance_float32` or `abalone_released_distance_float64`.
It verifies the archived source evidence and installed published wheel.
A future run can produce a different output; its reproduction status must
be checked before using its trace to explain the archived benchmark.

This report is separate from the [uniform-weight diagnostic](abalone-released-output-f5ccff9.md).
The conclusions cover seed 101, these prepared inputs, and the stated
threshold. They do not diagnose the other seeds, prove all-input correctness,
or identify the low-level operation responsible for every distance discrepancy.

## Artifact checksums

- `abalone-released-distance-output-a1ada8e-float32.zip`:
  `9b44217a95c1ae6f279fe3a8129070dc669c8d1c47930fbeef3f386eb7fca08e`.
- `abalone-released-distance-output-1983146-float64.zip`:
  `9ce6850590043a5e3960a9521a1d1794e8e8cef2f4a4dcd770f08a215739d962`.
- `abalone-released-distance-output-1983146-float32-mismatch.zip`:
  `63ea3c801877eb48f44602d4f98113949f514a1daeae299d1f528dc5773c9c00`.
