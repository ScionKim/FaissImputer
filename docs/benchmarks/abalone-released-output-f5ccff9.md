# Abalone released 0.3.22: MCAR output diagnostic

[Benchmark index](README.md) · [Published Abalone comparison](released_abalone_0.3.22.md)

The published Abalone comparison recorded differences between KNNImputer
and FaissImputer 0.3.22 at seed 101. This diagnostic reproduces those
outputs and examines every masked entry differing by more than `1e-5`:
two entries for float32 and four for float64. These are dtype-specific
observations, with some positions appearing in both dtypes.

The differences are explained by two mechanisms: alternative selections
among exactly tied boundary donors, and candidate-order reversals in the
captured KNNImputer distances. For every affected entry, the Faiss donor
selection is admissible under exact arithmetic on the prepared input
values. This finding is limited to the examined entries; it does not
establish general neighbor-ordering or prediction equivalence.

No timing measurements were taken.

## Configuration and evidence

- Published FaissImputer 0.3.22 and scikit-learn KNNImputer 1.9.1.
- Abalone numerical features: 3,000 training rows, 1,000 held-out queries,
  seven features, seed 101, 10% target overall MCAR missingness.
- `Length` remains observed. `Sex` and `Rings` are excluded.
- Available donors, five neighbors, uniform weights, mean aggregation,
  Flat index, one native thread, and a 256 MiB scikit-learn working-memory setting.
- Python 3.12.14, NumPy 2.5.3, Faiss 1.15.1, SciPy 1.18.1.
- Diagnostic source: `f5ccff9968fd3f6f885b51d66ddb5d13a50cdd28`.
- Original benchmark source: `4acd09dfa1aca339f38d401bdb2a1076bf3abe8c`,
  [run 37099197073, attempt 1](https://github.com/ScionKim/FaissImputer/actions/runs/37099197073/attempts/1).
  This source identifies the benchmark scripts, not the source of the installed wheel.

| Dtype | Diagnostic runner | Preserved evidence | GitHub run |
| --- | --- | --- | --- |
| float32 | AMD EPYC 9V45 96-Core Processor | [Original artifact](../../benchmarks/results/abalone-released-output-f5ccff9-float32.zip) | [37163466543](https://github.com/ScionKim/FaissImputer/actions/runs/37163466543/attempts/1) |
| float64 | INTEL(R) XEON(R) PLATINUM 8573C | [Original artifact](../../benchmarks/results/abalone-released-output-f5ccff9-float64.zip) | [37163676248](https://github.com/ScionKim/FaissImputer/actions/runs/37163676248/attempts/1) |

The original benchmark used an Intel Xeon Platinum 8573C runner.
The float32 diagnostic used a different CPU, but reproduced the original
prepared-input fingerprints, full-output hashes and saved imputed values.
These runs provide output evidence, not a cross-hardware timing comparison.

Both artifacts retain the installed published wheel, its SHA-256, the
installation report, dependency versions, original benchmark ZIP, reference
JSON, source dataset ZIP, prepared arrays, outputs and captured distances.
The retained wheel has SHA-256
`3343bd281e08a27869dc917f5e1479767ce5574be91a8d0f61247bdc728f62c3`.
Installed core files match this wheel. The original benchmark did not
preserve a wheel hash, so binary identity with that earlier installation
is not established solely by the version number.

## Reproduction checks

Both diagnostics report `status="ok"`, `inputs_reproduced=true`,
`original_reproduced=true` and `traced_outputs_unchanged=true`.
The KNNImputer and current-release records each match their original
full-output hashes and all 711 saved masked-entry values. Record indices
0 and 2 select `knn` and `current`, respectively, from each dtype JSON.

Independent checks of the saved NPZ arrays confirm the input and output
fingerprints, masked-entry differences and reconstruction metrics.
Exact distances were independently recomputed from the saved training
and query arrays for all detailed rows. No imputer was rerun for this
saved-evidence review.

## Distance and selection definitions

For each query/donor pair, the reference computes the squared distance
as `(7 / shared_feature_count) * sum((query_j - donor_j)**2)` over their
shared observed features. Every operation uses exact rational arithmetic
on the represented binary32 or binary64 input values. A float32 value
can be promoted to binary64 without changing its represented value.
This reference does not use idealized values from before preprocessing.

Donors must contain the feature being imputed. An admissible set of five
donors includes every strictly closer eligible donor and fills the remaining
slots from donors at the exact fifth-neighbor distance. Exact ties can
therefore admit more than one output mean.

KNN donor IDs are observed inside its actual `argpartition` call.
Faiss donor selections are reconstructed from captured finished searches.
The traced transforms reproduce the uninstrumented outputs, and none of
the affected queries has an ambiguous duplicate-query mapping.

## All entries above the threshold

Row numbers and donor IDs are zero-based indices in the prepared arrays.
Differences are `Faiss output - KNN output` in standardized units.
Values below are rounded for readability; full precision, rational
numerators/denominators and complete donor sets are retained in the JSON.

| Dtype | Query row | Feature | Signed output difference | KNN-only donor | Faiss-only donor | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | 185 | Shucked_weight | +0.106646299362 | 1921 | 69 | Exact boundary tie; both choices admissible |
| float32 | 984 | Shell_weight | +0.0750841647387 | 113 | 346 | Captured KNN distance order reversed; Faiss selects the closer donor |
| float64 | 185 | Shucked_weight | -0.106646246792 | 69 | 1921 | Captured KNN distance order reversed; Faiss selects the closer donor |
| float64 | 984 | Shell_weight | -0.0750841752667 | 346 | 113 | Exact boundary tie; both choices admissible |
| float64 | 845 | Whole_weight | +0.00982442059762 | 2312 | 1087 | Captured KNN distance order reversed; Faiss selects the closer donor |
| float64 | 845 | Shucked_weight | +0.00413713888417 | 2312 | 1087 | Captured KNN distance order reversed; Faiss selects the closer donor |

For float32, row 185 has an exact boundary tie between donors 69 and
1921; both selections are admissible. At row 984, donor 346 is strictly
closer than donor 113 in exact arithmetic, while the captured KNN distances
place donor 113 first.

For float64, row 984 has an exact boundary tie between donors 113 and
346. At row 185, donor 1921 is strictly closer than donor 69. At row 845,
donor 1087 is strictly closer than donor 2312; that exchange affects two
features. The captured KNN distances reverse these strict orderings.

The tie status differs by dtype because the reference uses the actual
prepared values of each dtype. A floating-point equality in a computed
distance array is not itself evidence of an exact mathematical tie.

### Boundary distance evidence

`Exact squared gap` is `D²(KNN-only donor) - D²(Faiss-only donor)`.
A positive gap makes the Faiss-only donor strictly closer. The final two
columns are the actual captured KNN distances (not squared distances).
For strict reversals, KNN reports a smaller distance for the farther donor.

| Dtype | Query row / feature | Exact squared gap | KNN distance to KNN-only donor | KNN distance to Faiss-only donor |
| --- | --- | --- | --- | --- |
| float32 | 185 / Shucked_weight | 0 | 0.18786023557186127 | 0.18786099553108215 |
| float32 | 984 / Shell_weight | 1.73670331449e-08 | 0.11012771725654602 | 0.11012866348028183 |
| float64 | 185 / Shucked_weight | 2.15657473115e-17 | 0.18786014876752272 | 0.1878601487675296 |
| float64 | 984 / Shell_weight | 0 | 0.11012768790828696 | 0.11012768790829049 |
| float64 | 845 / Whole_weight | 1.56858599818e-16 | 0.099918300068913252 | 0.099918300068944366 |
| float64 | 845 / Shucked_weight | 1.56858599818e-16 | 0.099918300068913252 | 0.099918300068944366 |

The selected-donor means explain the large output differences. The largest
absolute residual between an affected output and the float64-rounded exact
mean of its selected donor values is 3.57627869541e-08 for float32 and
2.22044604925e-16 for float64. Aggregation rounding does not explain the large differences.

## Reconstruction quality is a separate question

These metrics describe seed 101 only, across its 711 masked query entries.
They are errors against held-out ground truth in standardized units, not
errors against an exact-neighbor reference. They are not medians across
the three benchmark seeds or across repeated timing trials.

| Dtype | Method | RMSE | MAE |
| --- | --- | --- | --- |
| float32 | KNNImputer | 0.274559765876 | 0.177144064717 |
| float32 | FaissImputer 0.3.22 | 0.274643705434 | 0.177399663669 |
| float64 | KNNImputer | 0.274643706073 | 0.177399662529 |
| float64 | FaissImputer 0.3.22 | 0.274563045129 | 0.177163700697 |

KNNImputer has lower RMSE and MAE for this float32 case; FaissImputer has
lower RMSE and MAE for this float64 case. Exact-neighbor agreement does
not guarantee lower reconstruction error. Neither result establishes
general quality superiority.

## Reading or reproducing the evidence

Each archive contains `abalone_released_0.3.22_<dtype>_diagnostic.json`
and its corresponding `.npz` file. In the JSON, `rows[].features[]` holds
`exact_reference`, `knn_observation`, `faiss_selections`, `outputs` and
`donor_details`. Filter features by `abs(outputs.faiss - outputs.knn) > 1e-5`
to recover the entries shown above. The output-difference column is the
signed subtraction of those outputs. The squared-gap column subtracts
the two stored exact rational distances before converting for display.
`quality` contains the seed-level reconstruction errors.

The [diagnostic script](../../benchmarks/diagnose_abalone_output.py) is run
through the [Real-data output diagnostic workflow](../../.github/workflows/diagnose-real-data-float32.yml).
Choose `abalone_released_float32` or `abalone_released_float64`.
The workflow installs the published 0.3.22 wheel with the recorded dependency
versions and verifies the preserved source evidence. A baseline-output
mismatch is reported as a failure and prevents tracing that different output
as if it reproduced the original case.

## Scope

This diagnostic concerns the published MCAR, seed 101 cases. It is separate
from the [earlier Abalone MAR, seed 303 diagnostic](abalone-output-1969f4b.md).
The reference covers represented standardized values, and the detailed
selection findings cover the entries above the stated threshold. They do
not prove all-input correctness, general prediction equivalence, or the
precise low-level arithmetic operation responsible for a captured reversal.

## Artifact checksums

- `abalone-released-output-f5ccff9-float32.zip`:
  `40d3555f510cda3231db329088af53b61b07f1b8275960f1837ade2336cfff91`.
- `abalone-released-output-f5ccff9-float64.zip`:
  `1b1778cae309710e8ca91c4168ecc4e17bb9068badaccdb913fe48bc6b4e4e4a`.
