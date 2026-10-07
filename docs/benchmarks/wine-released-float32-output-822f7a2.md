# Wine Quality White released 0.3.22: float32 output diagnostics

[Benchmark index](README.md) · [Published uniform-weight comparison](released_wine_quality_float32_0.3.22.md) · [Published distance-weight comparison](released_wine_quality_float32_distance_0.3.22.md)

Four diagnostics examine output differences between published FaissImputer
0.3.22 and KNNImputer on held-out Wine Quality White data. Uniform seed 101
and distance-weight seeds 101 and 303 reproduce the archived inputs and both
methods' complete outputs. Their traces identify neighbor-order and weighting
differences. No timing measurements were taken.

Distance-weight seed 202 reproduces the inputs and Faiss output, but three
KNN values change by at most `1.7881393432617188e-07`. Its artifact retains
`status="baseline_output_mismatch"`, `original_reproduced=false` and a nonzero
workflow exit. Its traces describe the current execution only; they do not
establish the internal cause of the archived seed-202 outputs.

## Configuration and provenance

- Wine Quality White: 3,000 training rows, 1,000 held-out query rows and
  11 numerical features. The `quality` target is excluded.
- MCAR with 10% target overall missingness; `alcohol` remains observed.
- Prepared inputs are float32; ground truth is float64 and uses the scaler
  fitted to observed training values. The API is `fit_then_transform`.
- Available donors, five neighbors, `metric="l2"`, mean aggregation and
  the Flat index; uniform or distance weights as stated for each case.
- One native thread and 256 MiB scikit-learn working memory.
- Published FaissImputer 0.3.22, scikit-learn 1.9.1, NumPy 2.5.3,
  Faiss 1.15.1, SciPy 1.18.1 and Python 3.12.14.
- Benchmark source: `6d533d37c8173058ce2cfb314701377ee67b4b59`.
  [Uniform run 37403070692](https://github.com/ScionKim/FaissImputer/actions/runs/37403070692/attempts/1)
  used AMD EPYC 7763; [distance run 37403259216](https://github.com/ScionKim/FaissImputer/actions/runs/37403259216/attempts/1)
  used AMD EPYC 9V74.
- All diagnostic runs use source `822f7a251dcc43c8ef3d2ead9fb1b6a74b596d25`.
  These source commits identify scripts, not the source revision of the
  installed published wheel.

| Weights | Seed | Diagnostic runner | Reproduction | Preserved artifact | Run, attempt 1 |
| --- | ---: | --- | --- | --- | --- |
| uniform | 101 | AMD EPYC 7763 64-Core Processor | Both methods | [ZIP](../../benchmarks/results/wine-released-float32-output-822f7a2-uniform-seed101.zip) | [37412072617](https://github.com/ScionKim/FaissImputer/actions/runs/37412072617/attempts/1) |
| distance | 101 | AMD EPYC 9V74 80-Core Processor | Both methods | [ZIP](../../benchmarks/results/wine-released-float32-output-822f7a2-distance-seed101.zip) | [37412508984](https://github.com/ScionKim/FaissImputer/actions/runs/37412508984/attempts/1) |
| distance | 202 | AMD EPYC 7763 64-Core Processor | Faiss only; KNN mismatch | [ZIP](../../benchmarks/results/wine-released-float32-output-822f7a2-distance-seed202-mismatch.zip) | [37412766255](https://github.com/ScionKim/FaissImputer/actions/runs/37412766255/attempts/1) |
| distance | 303 | AMD EPYC 9V45 96-Core Processor | Both methods | [ZIP](../../benchmarks/results/wine-released-float32-output-822f7a2-distance-seed303.zip) | [37413270724](https://github.com/ScionKim/FaissImputer/actions/runs/37413270724/attempts/1) |

Each ZIP preserves its original benchmark archive, extracted reference JSON,
source dataset, dependency versions, published wheel and installation report,
diagnostic JSON and prepared arrays with outputs. Installed core-file hashes
match the retained wheel, whose SHA-256 is
`3343bd281e08a27869dc917f5e1479767ce5574be91a8d0f61247bdc728f62c3`.
The original benchmarks did not retain their wheel hashes; matching package
versions alone do not establish identity with those original binary artifacts.

These are separate workflow runs. Seed 303 reproduces the distance benchmark
despite a different CPU model, while seed 202 does not. This evidence does
not isolate a cause involving CPU model, BLAS dispatch or an individual
floating-point operation. The uniform and distance cases are also separate
runs and are not a controlled hardware comparison of weight settings.

## Reproduction and counting

All four diagnostics reproduce the prepared inputs and preserve outputs
through instrumentation (`traced_outputs_unchanged=true`). For the three
fully reproduced cases, each method matches its archived full-output hash
and every saved masked-entry value. Traces are observations of a diagnostic
execution reproducing those outputs, not recordings of the original run.
The previous 0.3.21 release is not traced.

An entry is above the threshold when
`abs(float64(Faiss output) - float64(KNN output)) > 1e-5` in standardized
units. Float32 values are promoted exactly before subtraction. Counts below
refer to one weight setting and seed, without pooling repeats or cases.

| Weights | Seed | Scored entries | Entries above threshold | Affected rows | Traced rows | Traced entries | Maximum absolute output difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 101 | 1105 | 2 | 1 | 1 | 7 | 0.2630177140235901 |
| distance | 101 | 1105 | 47 | 32 | 32 | 55 | 0.1321125626564026 |
| distance, current execution only | 202 | 1103 | 35 | 23 | 24 | 43 | 0.0041977763175964355 |
| distance | 303 | 1093 | 38 | 27 | 27 | 44 | 0.0031270980834960938 |

Detailed rows include all their missing features, not just entries above
the threshold. Seed 202 also includes row 256 because its KNN output changed
from the archive, even though its between-method differences are below the
threshold. No detailed rows were truncated. The archived seed-202 comparison
also has 35 above-threshold entries and the same maximum; those aggregate
matches do not make the complete outputs identical.

## Distance and arithmetic references

For represented prepared inputs, the exact squared-distance reference is
`(11 / shared_feature_count) * sum((query_j - donor_j)**2)` over shared
observed features, evaluated with rational arithmetic. A donor must contain
the target feature and share at least one observed feature with the query.
An admissible five-neighbor set includes all eligible donors strictly closer
than the exact fifth-neighbor boundary and fills its remaining slots at
that boundary. Exact arithmetic here concerns the stored binary32 inputs,
not ideal values before scaling.

KNN donor IDs and weights are observed during its actual imputation calls.
Faiss weights are captured at aggregation; distance-weight output cells are
mapped by the actual batch row and feature. Uniform selections come from
finished search results and are unambiguous in the traced case.

Uniform references use exact means of selected represented target values.
Distance-weight references separate three stages:

1. Exact rational squared distances followed by Decimal square roots and
   weighting at 80 and 120 digits. These are rounded references, not
   certified exact weighted means.
2. The same weighting applied to captured unsquared distances, distinguishing
   distance discrepancies from subsequent weight construction.
3. Exact rational arithmetic on captured floating-point targets and weights,
   distinguishing those operands from aggregation and output assignment.

If any selected distance is zero, only selected zero-distance donors
contribute, with equal normalized weights. A small positive computed distance
can change that behavior even when the selected donor IDs are the same.

## Findings

This table partitions the above-threshold entries in each diagnostic
execution. The seed-202 column is current-execution evidence only.

| Observed mechanism | Uniform 101 | Distance 101 | Distance 202, current only | Distance 303 |
| --- | ---: | ---: | ---: | ---: |
| Strict neighbor-order reversal | 2 | 1 | 0 | 0 |
| Exact zero recorded as positive by KNN, changing weights | 0 | 46 | 32 | 36 |
| Positive-distance differences propagated into weights | 0 | 0 | 3 | 2 |
| Total | 2 | 47 | 35 | 38 |

Every Faiss selection in these above-threshold cells is admissible under the
exact reference. KNN selections are also admissible except for the two
uniform-101 entries and one distance-101 entry with a strict order reversal.
These findings apply to the traced cells and represented inputs; they do
not prove all-input correctness or superiority in reconstruction error.

### Seed 101: neighbor-order reversal

At prepared query row 974, training donor 138 has exact squared distance
`0.058630284153805855` and donor 1542 has `0.058630293736259276`
(float64 displays of the stored rational values). Their exact squared gap
is approximately `9.582453423343887e-09`, with donor 138 strictly closer.
KNN captures unsquared distances `0.24213826656341553` for donor 138 and
`0.24213466048240662` for donor 1542, reversing their order. Faiss selects
donor 138 at the boundary; KNN selects donor 1542. This is not an exact tie.

| Weights | Feature | KNN output | Faiss output | Absolute difference |
| --- | --- | ---: | ---: | ---: |
| uniform | volatile acidity | -0.6151994466781616 | -0.8424705266952515 | 0.22727108001708984 |
| uniform | total sulfur dioxide | -1.044341802597046 | -0.7813240885734558 | 0.2630177140235901 |
| distance | volatile acidity | -0.7250814437866211 | -0.8571940064430237 | 0.1321125626564026 |

For distance weights, row 974's `total sulfur dioxide` uses a selected
zero-distance donor, so the boundary substitution does not produce an
above-threshold output difference there. Counts describe output effects,
not every differing donor ID.

### Seed 303: zero-distance weighting and an incidental tie

The largest seed-303 difference is at row 935 / `citric acid`. Both methods
select the same five donors. Donor 2090 has exact zero distance, but KNN
records `0.000661135942209512` and gives other selected donors positive
weight. Faiss records zero and uses only that zero-distance donor's target.
The outputs are `1.2789374589920044` for KNN and `1.2820645570755005` for
Faiss, an absolute difference of `0.0031270980834960938`.

One of the 36 zero-distance cases, row 925 / `fixed acidity`, also has a
different fifth donor: KNN chooses 800 and Faiss chooses 591. Those two
donors have identical represented feature values, exact distances and target
values; both choices are admissible. The shared zero-distance donor 2584
is recorded by KNN at `0.0008097228710539639` and by Faiss at zero.
The output difference is attributed to the zero-distance weighting change.
The boundary tie is incidental and is not counted as an additional cause.

### Positive-distance weighting

The three seed-202 current-execution cases are row 540 / `free sulfur dioxide`,
row 540 / `total sulfur dioxide` and row 831 / `chlorides`. Seed 303 has two:
rows 532 and 880 / `free sulfur dioxide`. Each pair selects the same
admissible donors, with positive distances. Captured-distance discrepancies
propagate through inverse-distance weights. Weight construction, aggregation
and output-assignment residuals are smaller than the observed output differences.

Across above-threshold entries, the largest absolute residual between an
assigned output and the exact mean of its captured operands is shown below.
Uniform operands are the selected targets; distance operands include the
captured weights. These residuals are not reconstruction errors.

| Weights | Seed | KNN maximum residual | Faiss maximum residual |
| --- | ---: | ---: | ---: |
| uniform | 101 | 4.76837158203125e-08 | 7.748603820800782e-08 |
| distance | 101 | 3.934592874332132e-07 | 1.617740868634111e-08 |
| distance, current execution only | 202 | 1.6396170390222112e-07 | 1.1606814601950922e-07 |
| distance | 303 | 2.346799823486246e-07 | 5.554867367535103e-08 |

Aggregation and assignment rounding alone do not explain the differences
above `1e-5` in these cells.

## Seed 202: preserved reproduction mismatch

The saved inputs and complete Faiss output match the archive. Three of
1,103 KNN masked values change, all in prepared query row 256:

| Feature | Archived KNN output | Diagnostic KNN output | Absolute change |
| --- | ---: | ---: | ---: |
| volatile acidity | -0.5069140195846558 | -0.5069138407707214 | 1.7881393432617188e-07 |
| chlorides | -0.6624231934547424 | -0.6624232530593872 | 5.960464477539063e-08 |
| density | -1.0092839002609253 | -1.0092837810516357 | 1.1920928955078125e-07 |

These changes are below `1e-5`, but that threshold selects detailed output
comparisons; it does not relax reproduction checks. The full KNN hash and
exact masked-value comparison correctly fail. The artifact is retained as
a mismatch, with `trace_explains_archived_outputs=false`.

Current instrumentation preserves current outputs and covers all 35 current
above-threshold entries plus the changed row. Their current observations
are 32 zero-distance weighting cases and three positive-distance weighting
cases. Although the above-threshold output values also match the archive,
matching values do not establish the original execution's internal choices.
The cause of the three cross-execution KNN changes remains unresolved.
Do not replace archived values with the diagnostic values or present this
run as a successful reproduction.

## Reconstruction error against held-out truth

RMSE and MAE below use every masked held-out entry for that case, in
standardized units: `sqrt(mean((output - truth)**2))` and
`mean(abs(output - truth))`. These are individual seed results, not timing
statistics or medians over repeats. Seed 202 reports current-execution
quality; use the original benchmark report for its archived quality.

| Weights | Seed | Method | RMSE | MAE |
| --- | ---: | --- | ---: | ---: |
| uniform | 101 | KNNImputer | 0.6941226105750372 | 0.49915837710654554 |
| uniform | 101 | FaissImputer 0.3.22 | 0.6943968042774052 | 0.49919072773235573 |
| distance | 101 | KNNImputer | 0.6232660825349146 | 0.3870385943709547 |
| distance | 101 | FaissImputer 0.3.22 | 0.6232332571138548 | 0.3868648175281769 |
| distance, current execution only | 202 | KNNImputer | 0.8067220051536561 | 0.4659985458585344 |
| distance, current execution only | 202 | FaissImputer 0.3.22 | 0.8067220329330516 | 0.46597707281682865 |
| distance | 303 | KNNImputer | 0.7460501777638537 | 0.40461214243602717 |
| distance | 303 | FaissImputer 0.3.22 | 0.7460501947648773 | 0.40459526431081144 |

The methods have similar aggregate reconstruction error under these tested
conditions. Their individual imputed values differ. Exact-neighbor fidelity
does not establish lower error against ground truth, algorithmic equivalence
or a general quality advantage.

## Reading and reproducing the evidence

Each ZIP contains
`wine_quality_released_0.3.22_float32_<weights>_seed<seed>_diagnostic.json`
and the matching NPZ. All displayed numbers come from these artifacts or
their retained original benchmark JSON. No imputer was rerun to prepare
this report; independent arithmetic checked saved arrays and trace data.

To recover the tables from saved evidence:

1. Verify the artifact hashes below and the member hashes in `SHA256SUMS.txt`.
   Verify the NPZ against `arrays.sha256` in its diagnostic JSON.
2. Check `original_reproduced`, `baseline_output_hashes_match`,
   `baseline_imputed_values_match`, `traced_outputs_unchanged` and
   `trace_explains_archived_outputs` before assigning an interpretation.
3. Count all NPZ `missing` entries and compare `knn_output` with `faiss_output`
   after exact promotion to float64. Match those counts with JSON
   `cells_above_threshold`, `affected_query_rows` and `max_abs_output_difference`.
   Detailed counts come from `rows` and `rows[].features`.
4. Filter `rows[].features[]` by the threshold. Compare actual IDs, exact
   rational distances, captured KNN distances and `admissible_exact_top_k`.
   For weighting, use `weight_observation.weight_input_distances` and
   `captured_weights`. A zero-rule change can coexist with a harmless tied
   donor substitution; inspect distance and target values before attributing
   a difference to donor IDs. The positive-distance cases use the three
   arithmetic references described above.
5. Uniform residuals use `output_minus_exact_mean`. Distance residuals use
   `captured_weight_arithmetic.assigned_output_minus_exact_weighted_mean`.
   Take maxima only over above-threshold entries for each method and case.
6. Recover the seed-202 change table from `baseline_value_comparison`, or
   compare its NPZ masked outputs with `reference-benchmark.json` using
   `baseline_record_indices` (KNN record 11, current Faiss record 10).
   Compute RMSE and MAE from NPZ outputs and `truth` at the `missing` mask,
   or read the validated diagnostic `quality` fields.

The [Wine diagnostic](../../benchmarks/diagnose_wine_quality_output.py) uses
the shared [output diagnostic](../../benchmarks/diagnose_abalone_output.py)
and [weight observations](../../benchmarks/diagnose_distance_weights.py).
The [Real-data output diagnostic workflow](../../.github/workflows/diagnose-real-data-float32.yml)
exposes these four choices:

```text
wine_released_uniform_float32_seed101
wine_released_distance_float32_seed101
wine_released_distance_float32_seed202
wine_released_distance_float32_seed303
```

A future execution can differ. Preserve its evidence and check reproduction
status before using its trace to explain archived outputs. These diagnostics
do not change the original benchmark data or add performance measurements.

## Artifact checksums

- `wine-released-float32-output-822f7a2-uniform-seed101.zip`:
  `108682c4acc6a21a4c07c9cbd33c869348cefe0fca216b51b67ba2c60bd3cee5`.
- `wine-released-float32-output-822f7a2-distance-seed101.zip`:
  `abd1a0e1471fbf3b073fdde50612a402bff10065793111a936eb8fcc0ffdba13`.
- `wine-released-float32-output-822f7a2-distance-seed202-mismatch.zip`:
  `9a74bc4fb261d5839d47db70de6071cae0e7156de582cfd0d6afaa5644b15a07`.
- `wine-released-float32-output-822f7a2-distance-seed303.zip`:
  `ad7a1635588981db5b41ed4707ea578efc232886e8f12d1cc6f4b2f9f995f85e`.
