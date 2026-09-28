# Abalone float64 output diagnostic - 1969f4b

This diagnostic reproduces the recorded Abalone float64 outputs and explains all nine masked entries whose absolute difference between KNNImputer and FaissImputer[available] exceeds 1e-5 in the selected case.

Six differences involve different admissible selections among exactly tied neighbors. Three involve a reversal of near-equal neighbor distances in the captured KNNImputer floating-point calculation. These conclusions apply to the case examined here.

[Evidence archive](../../benchmarks/results/abalone-output-1969f4b.zip) | [Original benchmark report](real-data-datasets-ef04b1b.md) | [Diagnostic script](https://github.com/ScionKim/FaissImputer/blob/1969f4b9e4b0b4c5a037c1db76baba5378419e55/benchmarks/diagnose_abalone_output.py) | [Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/36368000261)

## Scope and provenance

| Field | Value |
| --- | --- |
| Library source commit | `ef04b1b0274d3abdd977440dd61e7d91e328c64d` |
| Diagnostic commit | `1969f4b9e4b0b4c5a037c1db76baba5378419e55` |
| Installed library version | `0.3.20+bench.ef04b1b0274d` |
| Dataset | Abalone, seven numerical features |
| Training / held-out query rows | 3000 / 1000 |
| Missingness | MAR, nominal overall rate 0.10 |
| Always-observed MAR driver | Length |
| Input dtype | float64 |
| Seed / reference repeat | 303 / 1 |
| Neighbors / weights | 5 / uniform |
| Faiss configuration | available donors, l2, Flat, mean |
| Original benchmark CPU | AMD EPYC 9V74 80-Core Processor |
| Diagnostic CPU | AMD EPYC 7763 64-Core Processor |

The dataset is [Abalone, Nash et al. (1994)](https://doi.org/10.24432/C55C7W), distributed under CC BY 4.0. Sex and Rings are excluded. Scaling uses observed training values only.

The diagnostic uses Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, faiss-cpu 1.15.1, SciPy 1.18.1, threadpoolctl 3.7.0 and joblib 1.6.0. Recorded native thread pools use one thread.

The preserved baseline JSON and dataset archive match their original SHA-256 digests. The three installed FaissImputer source files match the original benchmark wheel's source files. Prepared input fingerprints and both output fingerprints match the original records.

The reference records are zero-based indices 346 for KNNImputer and 348 for FaissImputer[available] in `reference-benchmark.json`.

This is an output diagnostic. It adds no timing measurements or speedup claims.

## Diagnostic method

All row and donor indices below are zero-based indices in the prepared arrays.

Each finite prepared float64 value is converted exactly using `Fraction.from_float`. For a query and donor sharing `m` observed features, the independent reference computes:

    squared_distance = (7 / m) * sum((query[j] - donor[j]) ** 2)

The sum includes only jointly observed coordinates, and all arithmetic in this reference is rational. These are exact distances between the represented, prepared binary64 values, not idealized distances before scaling.

A donor is eligible for an imputed feature when its target value is observed and it shares at least one observed feature with the query.

An admissible exact top-5 selection includes every eligible donor strictly closer than the fifth-neighbor boundary and fills the remaining slots from donors exactly on that boundary. Multiple selections can therefore be valid. Training-row order defines one representative selection, not a uniquely required answer.

KNNImputer donor IDs are captured from the actual `argpartition` calls. Faiss donor IDs are reconstructed from the actual completed search results. Both traced transforms reproduce their corresponding untraced outputs.

Exact distances were independently recalculated for all 3000 donors of each of the four affected query rows. All 13 masked entries in those rows were examined. Each query maps uniquely to one completed Faiss search, and all four searches applied precision refinement.

## Output differences

The comparison covers 671 masked held-out entries. Absolute difference means:

    abs(KNNImputer output - FaissImputer[available] output)

Nine entries in four rows exceed 1e-5. The other 662 entries have a maximum absolute difference of `4.440892098500626e-16`. Detailed row collection was not truncated.

The following values are unrounded float64 results represented as round-trip decimal strings. They are individual cell differences, not aggregated timing or quality statistics.

| Query row | Feature | Absolute output difference | Classification |
| ---: | --- | ---: | --- |
| 80 | Shucked_weight | 0.04247741079337092 | Exact boundary tie |
| 318 | Height | 0.20175530092661365 | Exact boundary tie |
| 318 | Whole_weight | 0.03355469714163031 | Exact boundary tie |
| 318 | Shucked_weight | 0.37280706281415954 | Exact boundary tie |
| 318 | Viscera_weight | 0.06466760179197831 | Exact boundary tie |
| 318 | Shell_weight | 0.12178092840348256 | Exact boundary tie |
| 412 | Height | 0.15131647569496035 | Captured KNN distance order reversed |
| 412 | Whole_weight | 0.0010168090042918243 | Captured KNN distance order reversed |
| 660 | Shell_weight | 0.2550235912449398 | Captured KNN distance order reversed |

### Exact boundary ties

For the six entries classified as exact boundary ties, both methods select admissible exact top-5 donor sets.

The largest difference occurs at query row 318, Shucked_weight. There are 16 eligible donors at exact distance zero:

| Method | Selected training rows | Output |
| --- | --- | ---: |
| KNNImputer | 1555, 2101, 2160, 1332, 1389 | 0.29905433569582085 |
| FaissImputer[available] | 56, 168, 202, 688, 814 | -0.0737527271183387 |

Both selections are valid under the exact distance reference. Their donor target values differ, so their means differ.

Some exact ties are numerically split by the captured KNN distance calculation. These observations should therefore not be reduced to a claim that only the tie-breaking policy differs.

### Near-equal distance order reversals

Three entries have a unique exact fifth neighbor, but the captured KNN calculation ranks another donor ahead of it. Faiss selects the exact fifth neighbor in these entries.

The gap below is the farther donor's exact squared distance minus the nearer donor's exact squared distance. Its float64 display is shown; the exact numerator and denominator are preserved in the diagnostic JSON.

| Query row | Feature(s) | Exact fifth donor | Fifth donor selected by KNN | Exact squared-distance gap, float64 display |
| ---: | --- | ---: | ---: | ---: |
| 412 | Height, Whole_weight | 2093 | 359 | 3.8663560038479463e-16 |
| 660 | Shell_weight | 1943 | 2763 | 6.433312756463483e-17 |

The actual captured KNN distances show the reversed ordering. These are distances, not squared distances:

| Query row | Exact nearer donor: captured distance | Exact farther donor: captured distance |
| ---: | --- | --- |
| 412 | 2093: 0.1861476630090333 | 359: 0.18614766300903193 |
| 660 | 1943: 0.15486754631881605 | 2763: 0.15486754631881353 |

KNNImputer selects valid neighbors according to its own computed distances. The disagreement arises between that computed ordering and the exact prepared-input reference.

Across all 13 detailed masked entries, Faiss selections are exact-admissible in 13 cases and KNN selections in 10. All 13 KNN selections are admissible under their captured KNN distances.

For the 26 examined method/cell selections, the maximum absolute residual between the actual output and the correctly rounded exact rational mean of the selected donor values is `2.220446049250313e-16`. The large output differences are therefore explained by donor selection rather than mean aggregation.

## Reconstruction quality

RMSE and MAE are computed against held-out ground truth over the same 671 masked entries, in units standardized using observed training values.

These are single-case metrics computed from the archived output arrays. They are not medians across seeds or timing repeats.

| Method | RMSE | MAE |
| --- | ---: | ---: |
| KNNImputer | 0.3518318476795419 | 0.2221907184304144 |
| FaissImputer[available] | 0.35343837650499127 | 0.22191983222001 |

Agreement with the exact neighbor-distance reference is distinct from reconstruction quality against ground truth. Here Faiss has a slightly higher RMSE and a slightly lower MAE.

Similar aggregate reconstruction errors do not establish equality of individual imputed values or algorithmic equivalence.

## Evidence and reproducibility

The evidence ZIP preserves:

- `abalone_output_diagnostic.json`: provenance, reproduction checks, actual neighbor traces, exact rational distances and means, and quality metrics.
- `abalone_output_diagnostic.npz`: prepared training/query arrays, ground truth, missing mask, both full outputs, traced query indices and captured KNN distance rows.
- `reference-benchmark.json`: the original benchmark records.
- `candidate-build.json`, dependency and installation records, and the rebuilt library wheel.
- `dataset.json` and the original `source-data/abalone.zip`.

Evidence ZIP SHA-256:

    c8d67645aab03925146f731ca1e10fa71c14c8d3970f37591fa3e69107cef68d

Output-array NPZ SHA-256:

    cbea45fefed6027ee717282fee94706f88b90ce223b01de39427a608ad3734d6

The report can be checked from these files without running either imputer again:

1. Load the NPZ with `allow_pickle=False`.
2. Compute absolute output differences on `missing`, then count entries exceeding 1e-5 and their distinct query rows.
3. Recompute RMSE as `sqrt(mean((output - truth) ** 2))` and MAE as `mean(abs(output - truth))`, restricted to `missing`.
4. For each detailed query, recompute rational distances from the prepared arrays and apply feature-specific donor eligibility.
5. Compare the exact boundary and admissible selections with `rows[].features[].exact_reference`, `knn_observation` and `faiss_selections`.
6. Recompute each selected donor mean exactly and compare it with the archived output.

The existing Real-data output diagnostic workflow supports `diagnostic = abalone_float64`. It fixes the library source to the original benchmark commit while using diagnostic code from the selected workflow branch. Commit `1969f4b9e4b0b4c5a037c1db76baba5378419e55` identifies the diagnostic code that produced this evidence.

These findings explain the selected Abalone float64 case. Other seeds, configurations, float32 cases and Wine Quality discrepancies require their own evidence before receiving the same explanation.