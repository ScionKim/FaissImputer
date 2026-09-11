# FaissImputer

> **Warning:** FaissImputer 0.1.x has a known neighbor-mapping bug that can
> produce incorrect imputations, and it is incompatible with scikit-learn 1.8+.
> Use version 0.2.0 or newer.

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/v0.3.12/LICENSE)

A scikit-learn-compatible missing-value imputer with [Faiss](https://github.com/facebookresearch/faiss)-backed neighbor search.

Current release: [0.3.12](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.12).
See [Releases](https://github.com/ScionKim/FaissImputer/releases) for version history.

## Comparison with KNNImputer

FaissImputer supports scikit-learn pipelines, but its defaults and options
differ from [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).
The comparison below describes the current source version, including the
unreleased `keep_empty_features` option. PyPI FaissImputer 0.3.12 does not
include this option.

| Behavior | FaissImputer | KNNImputer |
| --- | --- | --- |
| Donors | Training rows observed in every non-empty feature by default; `donor_policy="available"` permits partially observed donors selected per missing feature. | Donors selected per missing feature; other donor features may be missing. |
| Neighbors | `n_neighbors=3`; complete mode requires at least that many complete donors. Available mode permits fewer eligible donors. | `n_neighbors=5`; fewer usable neighbors are allowed. |
| Aggregation | Mean (default) or median via `strategy`, with uniform weights by default. Distance and callable weights require `strategy="mean"` and `metric="l2"`. | Mean with uniform (default), distance, or callable weights; no median strategy. |
| Numeric precision | Converts inputs and produces imputed values as `float32`. | Supports floating inputs including `float64`, without forcing conversion to `float32`. |
| All-missing training columns | Dropped by default; `keep_empty_features=True` retains them with zero values under either donor policy. | Dropped by default; `keep_empty_features=True` retains them with zero values. |
| Missing-value marker | `NaN`; no configurable `missing_values` parameter. | Configurable `missing_values`, default `np.nan`. |
| Missing indicators | `add_indicator=True` appends 0/1 columns for features missing during fit; disabled by default. | `add_indicator=True` appends indicators for features missing during fit. |

For a closer comparison, use `donor_policy="available"` and `strategy="mean"`,
and match `n_neighbors` and `weights`. Available mode requires `metric="l2"`
and `index_factory="Flat"`; compare it with KNNImputer's default
`metric="nan_euclidean"`.

These settings do not guarantee identical donor choices or imputed values:
float32 conversion, distance calculations, and ties can affect results.
See the [real-data comparison](https://github.com/ScionKim/FaissImputer/blob/v0.3.10/docs/benchmarks/real-data-a3bd1ce3.md)
for measured per-cell differences, and the
[scikit-learn imputation guide](https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation)
for KNNImputer behavior.

## Performance at a glance

FaissImputer can accelerate nearest-neighbor imputation, especially when
queries reuse the same missing-feature patterns.

### Which donor policy should I use?

A **donor** is a training row used to supply a missing value.

- **`complete` (default):** Uses training rows observed in every non-empty
  feature as donors. Rows missing any of those features are excluded from
  neighbor search. Columns entirely missing during fit do not disqualify rows.
  Choose this when you have enough complete training rows.
- **`available`:** Also allows partially observed training rows as donors.
  Donors are selected separately for each missing feature: they must contain
  that feature's value and share at least one originally observed feature
  with the query row. No fully observed training row is required.

The benchmarks below use fully observed training data for `complete`
and partially missing training data for `available`. They compare each
policy against KNNImputer, not the two policies against each other.

### Released 0.3.10 benchmarks

Both runs compared PyPI FaissImputer 0.3.10 with scikit-learn 1.9.0's
KNNImputer on synthetic float32 data, using one thread, 300 queries,
20 features, five neighbors, and uniform-weight mean aggregation.

Each query had four missing features. Complete training data was fully
observed; available training data had approximately 10% MCAR missingness.
Inputs were identical within each policy comparison.

Times measure the first transform after fitting, excluding fit time.
Medians cover three seeds and three fresh runs per seed. Speedups are
KNNImputer/FaissImputer ratios of unrounded median times.

#### 20,000-row comparison — AMD runner

| Donor policy | Query missingness | KNNImputer | FaissImputer | Speedup |
|---|---|---:|---:|---:|
| complete | One shared pattern | 422.4 ms | 22.0 ms | **19.24×** |
| complete | Random patterns | 409.8 ms | 107.2 ms | **3.82×** |
| available | One shared pattern | 391.4 ms | 166.2 ms | **2.35×** |
| available | Random patterns | 385.6 ms | 167.5 ms | **2.30×** |

[Raw results and environment](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/scaling-threads-34297607304.json)
· [Workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/34297607304)

#### Training-size comparison — Intel runner

Values above 1× mean FaissImputer is faster.

| Training rows | complete / fixed | complete / random | available / fixed | available / random |
|---:|---:|---:|---:|---:|
| 1,000 | 6.75× | **0.71× — slower** | 1.24× | 1.19× |
| 5,000 | 9.64× | 1.46× | 1.34× | 1.34× |
| 20,000 | 11.00× | 1.63× | 1.32× | 1.33× |
| 100,000 | 15.56× | 2.30× | 1.73× | 1.64× |

[Raw results and environment](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/scaling-threads-34310124369.json)
· [Workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/34310124369)

In both runs, complete outputs matched KNNImputer exactly; available
outputs differed by at most `4.77e-7` on the tested inputs.
Repeated outputs were unchanged.

Memory advantages varied with size. In the scaling run, available's
median whole-worker peak RSS was 6–21% higher at 5,000 and 20,000 rows,
but 15–27% lower at 100,000 rows, compared with KNNImputer.

These runs used different CPUs. Differences between their results do
not measure a change between software versions. Speed and memory
advantages are not guaranteed on other workloads.

## Installation

FaissImputer requires Python 3.10 or newer.

```bash
python -m pip install --upgrade "faiss-imputer>=0.3.12"
```

## Usage

### Complete donors (default)

When at least one training column is non-empty, training data must contain
enough rows observed in all non-empty columns to supply `n_neighbors` donors.

```python
import numpy as np

from faiss_imputer import FaissImputer

X_train = np.array(
    [
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
    ],
    dtype=np.float32,
)
X_missing = np.array(
    [
        [1.5, np.nan, 150.0],
        [2.5, 25.0, np.nan],
    ],
    dtype=np.float32,
)

imputer = FaissImputer(n_neighbors=2)
X_imputed = imputer.fit(X_train).transform(X_missing)
print(X_imputed)
```

Expected output:

```text
[[  1.5  15.  150. ]
 [  2.5  25.  250. ]]
```

### Partially observed donors

Set `donor_policy="available"` to choose donors separately for each missing feature. No fully observed training row is required.

```python
import numpy as np

from faiss_imputer import FaissImputer

# No training row is completely observed.
X_train = np.array(
    [
        [0.0, 10.0, np.nan],
        [2.0, 30.0, np.nan],
        [1.0, np.nan, 20.0],
        [3.0, np.nan, 40.0],
    ],
    dtype=np.float32,
)
X_missing = np.array(
    [[0.1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    dtype=np.float32,
)

imputer = FaissImputer(n_neighbors=1, donor_policy="available")
X_imputed = imputer.fit(X_train).transform(X_missing)
print(X_imputed)
```

Expected output:

```text
[[ 0.1 10.  20. ]
 [ 1.5 20.  30. ]]
```

By default, both policies return a new NumPy `float32` array. The training and query inputs are not modified.

### Feature names and pandas output

The following example requires FaissImputer 0.3.3 or newer.
It also requires pandas, which is optional:
`python -m pip install pandas`.

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faiss_imputer import FaissImputer

X = pd.DataFrame({
    "age": [20.0, None, 40.0],
    "income": [100.0, 190.0, 300.0],
})

pipe = make_pipeline(
    FaissImputer(n_neighbors=1),
    StandardScaler(),
).set_output(transform="pandas")

result = pipe.fit_transform(X)

print(pipe.get_feature_names_out())  # ['age' 'income']
print(result.columns.tolist())      # ['age', 'income']
```

Use `pipe.set_output(transform="default")` to return NumPy arrays instead.
For unnamed array inputs, feature names are generated as `x0`, `x1`, and so on.

## Parameters

- `n_neighbors` (default: `3`): Positive integer specifying the maximum number of donors used for each missing feature. When non-empty training columns exist, the complete-donor policy requires at least this many rows observed in all of them.
- `metric` (default: `"l2"`): Supports `"l2"` and `"ip"`. Raw inner product is not cosine similarity. The available-donor policy requires `"l2"`.
- `strategy` (default: `"mean"`): Supports `"mean"` and `"median"` for aggregating donor values and calculating fallback column statistics.
- `index_factory` (default: `"Flat"`): Faiss index description for the complete-donor policy. The available-donor policy accepts only `"Flat"` and uses the distance backend described below.
- `donor_policy` (default: `"complete"`): Use training rows observed in all non-empty features with `"complete"`, or allow partially observed training rows with `"available"`.
- `weights` (default: `"uniform"`): `"uniform"` or `None` preserves the existing unweighted aggregation. `"distance"` uses inverse Euclidean distance; a callable supplies custom weights. Distance and callable weights require `strategy="mean"` and `metric="l2"` under either donor policy.
- `add_indicator` (default: `False`): Append missingness indicators to the imputed output. Indicator columns are selected during `fit()` and remain fixed until refitting.
- `keep_empty_features` (default: `False`, unreleased): Drop columns that were entirely missing during `fit()`. Set to `True` to retain those columns with zero values. The selection remains fixed until refitting.

With distance weighting, if any selected donor has distance zero, only
the selected zero-distance donors contribute to that missing feature.

Callables receive a two-dimensional array of NaN-aware Euclidean distances
and must return real weights of the same shape. They should operate
independently on each row, without relying on batch size or neighbor order.
Unavailable neighbors have `NaN` distances and are excluded regardless of
the returned weight. Returned `NaN` weights contribute zero.

Invalid shapes, non-real weights, infinite weights for usable donors,
zero weight sums, or results outside the finite `float32` range raise
`ValueError`. Fitted fallback column statistics remain unweighted.

## Important behavior

### Shared behavior

- Inputs must be two-dimensional numeric array-like data, with `NaN` marking missing values. Values are converted to `float32`; infinity is not accepted.
- `transform()` requires the same number of features as `fit()`.
- An entirely missing query row uses column means or medians learned during `fit()` for non-empty training columns.
- A failed `fit()`, including a failed refit, clears the fitted state.

With `add_indicator=True`, appended values are `1` for originally missing
query entries and `0` otherwise. Indicator features are selected from all
training rows before donor filtering, in original column order. A feature
first missing only at transform time does not receive a new indicator column.

Indicator names use `missingindicator_<feature_name>`, such as
`missingindicator_age` or `missingindicator_x0`. Both
`get_feature_names_out()` and pandas output include the added columns.

Columns entirely missing during `fit()` are excluded from donor selection
and neighbor search. With `keep_empty_features=False`, they are omitted
from the imputed output. With `True`, they remain in their original positions
and always contain zero, even if a later query supplies an observed value
in that column. `transform()` still requires the full original input schema.

If every training column is empty, fitting succeeds without a neighbor
index. Transform returns no imputed columns by default, or all-zero columns
with `keep_empty_features=True`; no donor-count minimum applies in this case.

With `add_indicator=True`, indicators still refer to the original input
columns and query missingness, including dropped or zero-filled columns.
Output names and pandas columns follow the retained imputed columns first,
then the indicators in their original feature order. NaN-aware distance
normalization continues to use the original input feature count, including
for callable weights.

### Complete-donor policy

- Donors must be observed in all non-empty training columns. When any such columns exist, at least one complete donor is required.
- When non-empty training columns exist, `n_neighbors` cannot exceed the number of complete donors.
- Neighbor search uses only the originally observed columns of each query row.
- The default `index_factory="Flat"` performs exact neighbor search.

### Available-donor policy

- Only `metric="l2"` and `index_factory="Flat"` are supported. Mean supports all weight options; median requires uniform weighting.
- A donor must observe the feature being imputed and share at least one originally observed feature with the query row.
- Donors are ranked by squared L2 distance over shared observed features, scaled by the total feature count divided by the shared feature count.
- Each missing feature uses up to `n_neighbors` eligible donors. Fewer eligible donors are allowed.
- If no eligible donor exists for a feature, its fitted column mean or median is used.
- Rows missing all non-empty training features are ignored. Entirely missing training columns follow `keep_empty_features`.

The available-donor policy uses batched NaN-aware distances and Faiss neighbor selection, with float64 safeguards for detected numerical risks. This backend is also used when all training donors are complete.

Batching trades memory for throughput. Its internal batch-sizing budget is
not a total process-memory limit: fitted donor data, distance calculations,
and temporary arrays also consume memory. Larger batches can improve speed
but increase peak memory usage. FaissImputer does not set the number of
threads; configure thread limits in the application when needed.

Donor-side arrays are prepared during `fit()` and reused across query batches.
This reduces repeated work but increases retained memory after fitting;
lower peak memory is not guaranteed.

## Benchmarks

Performance depends on donor policy, data size, and missingness patterns. Neither faster execution nor lower memory use than `KNNImputer` is guaranteed. Similar average errors do not imply identical imputed values; ties and numerical precision can affect donor selection.

- [Complete-aggregation recovery pilot](https://github.com/ScionKim/FaissImputer/blob/v0.3.10/docs/benchmarks/complete-aggregation-e5ef482.md): compares v0.3.8 source with the implementation prepared for 0.3.10 on one runner, across three seeds and five query/feature shapes. Complete-policy first transform was 16.69-60.96% faster; available-policy changes ranged from -3.59% to +3.84%. These are source-checkout measurements with ordinary-scale inputs, not published-wheel measurements or a KNNImputer comparison.
- [Available-donor batching and threads](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/available-batching-90c8cfb8.md): compares 16/64/128 MiB batch budgets and 1/2/4 threads, including repeated 500,000-row measurements and a single-run million-row pilot. Documents both speed gains and memory tradeoffs.
- [Real-data pilot](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/real-data-a3bd1ce3.md): compares SimpleImputer, KNNImputer, and both Faiss donor policies under MCAR and selected MAR missingness. This small dataset is not a scalability test.
- [Complete-donor patterns](https://github.com/ScionKim/FaissImputer/blob/v0.3.1/docs/benchmarks/complete-patterns-9d179b2b.md): measures the impact of training size and query missingness patterns.
- [Partial-donor development snapshot](https://github.com/ScionKim/FaissImputer/blob/bc1929f58608033bdca565260878e5c8f2a7571f/docs/benchmarks/partial-donors-0a3cc077.md): historical results for measured commit `0a3cc077`, not the final 0.3.0 implementation.
- [Historical 0.2.0 benchmark](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/docs/benchmarks/v0.2.0.md): results for the earlier complete-donor-only implementation.

Reports provide measurement conditions, results, and links to reproduction code and raw data.

## Example notebook

See [Imputing Missing Values with Faiss Imputer](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/notebooks/Impute_Missing_Values_with_Faiss_Imputer.ipynb) for a complete-donor example.

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/ScionKim/FaissImputer/issues) or create a pull request. Further work is tracked in the [roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md).

## Author

- **GitHub:** [@ScionKim](https://github.com/ScionKim/)

## License

This project is licensed under the [MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.3.12/LICENSE).

### Third-Party Licenses

FaissImputer depends on Meta's [Faiss](https://github.com/facebookresearch/faiss), which is distributed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.
