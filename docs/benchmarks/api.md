# API reference

This reference describes FaissImputer 0.3.16.

See the [README](../README.md) for installation and a quick start, or
[usage examples](usage.md) for complete examples.

## Constructor

```python
import numpy as np
from faiss_imputer import FaissImputer

imputer = FaissImputer(
    n_neighbors=3,
    metric="l2",
    strategy="mean",
    index_factory="Flat",
    donor_policy="complete",
    weights="uniform",
    add_indicator=False,
    keep_empty_features=False,
    missing_values=np.nan,
    copy=True,
)
```

| Parameter | Default | Accepted values and behavior |
| --- | --- | --- |
| `n_neighbors` | `3` | Positive Python or NumPy integer. Boolean values are rejected. |
| `metric` | `"l2"` | `"l2"`, `"nan_euclidean"`, or `"ip"`. `"nan_euclidean"` is an alias for `"l2"`. |
| `strategy` | `"mean"` | `"mean"` or `"median"`. |
| `index_factory` | `"Flat"` | Faiss index-factory string. Available-donor mode requires `"Flat"`. |
| `donor_policy` | `"complete"` | `"complete"` or `"available"`. |
| `weights` | `"uniform"` | `"uniform"`, `None`, `"distance"`, or a callable. Non-uniform weights require mean aggregation and an L2 metric. |
| `add_indicator` | `False` | Append indicators for features that contained missing values during fit. |
| `keep_empty_features` | `False` | Retain entirely missing training columns as zero-valued output columns. |
| `missing_values` | `np.nan` | NaN or a finite real numeric marker. |
| `copy` | `True` | Preserve transform input by default; `False` permits input reuse when possible. |

`add_indicator`, `keep_empty_features`, and `copy` accept Python and NumPy
boolean values.

## Methods

- `fit(X, y=None)`: Learn donors, column statistics, feature names, and
  optional indicator features. Returns the estimator; `y` is ignored.
- `transform(X)`: Impute using fitted state. Requires the original number
  of input features, including columns that were empty during fit.
- `fit_transform(X, y=None)`: Fit and then transform the training input.
- `get_feature_names_out(input_features=None)`: Return output feature names,
  accounting for dropped columns and appended indicators.
- `set_output(transform="pandas")`: Return pandas DataFrames.
  Use `transform="default"` to return NumPy arrays.

Failed fits and refits clear fitted state. Call `fit()` successfully before
using `transform()` or `get_feature_names_out()`.

## Input and output precision

Inputs must be two-dimensional numeric data.

After array validation:

- `float32` arrays retain `float32`.
- `float64` arrays retain `float64`.
- Integer arrays and other supported numeric dtypes are converted to
  `float32`.
- Lists and pandas inputs use the dtype inferred during validation.
  Floating-point lists commonly become `float64`.

Donor values and fitted statistics retain the normalized training dtype.
The output dtype follows the normalized query dtype, which may differ
from the training dtype.

For example, a float64-trained estimator returns float32 output for a
float32 query. Imputed values may therefore be rounded. A float64 query
cannot recover precision already lost in float32 training data.

Observed values must be finite and representable in the normalized input
dtype. Results that exceed the output dtype's finite range are rejected.
Empty-feature restoration and appended indicators retain the output dtype.

### Search precision and numeric limits

Complete-donor mode converts search vectors to float32. Stored donor
values and imputation output can still be float64.

Coordinates required for a complete-donor search must be representable
as finite float32 values. Non-Flat factories also require this conversion
when fitting their full donor index. A fully observed query does not
require a neighbor search.

For non-uniform weights, distances to selected complete donors are
calculated from the original normalized donor and query values in float64.

Available-donor mode calculates distances in float64 and uses a float32
selection cache with precision refinement. With float64 training data or
queries, selected distances are recomputed in float64 before weighting.

When intermediate squared norms overflow, available mode can recompute
distances using shared observed features. Required squared distances that
overflow float64, or positive squared distances that underflow to zero,
raise `ValueError`.

Dtype preservation does not guarantee identical neighbor choices or
results to `KNNImputer`. Search precision, distance calculations, and ties
can affect results.

## Missing-value markers

The default marker is `np.nan`. A finite Python or NumPy integer or
floating-point marker is also accepted.

Boolean, string, complex, array-valued, infinite, and `None` markers are
rejected.

Numeric markers are matched before dtype conversion. Marker matching
preserves the original numeric values of mixed lists and pandas columns
so that conversion does not turn a distinct observed value into a missing
entry.

A numeric marker may exceed the float32 range because matched entries are
replaced with NaN before observed values are converted.

When a numeric marker is configured, actual NaN values are rejected.
Use one missing-value representation consistently for fit and transform.
Observed infinities are always rejected.

## Donor policies and metrics

### Complete donors

The default policy uses training rows observed in every non-empty
training feature.

When non-empty features exist, fitting requires at least one complete
donor and at least `n_neighbors` complete donors.

Search uses only the query's observed, non-empty training features.
Missing query coordinates do not participate in neighbor selection.

`metric="l2"` and `metric="nan_euclidean"` use the same search path.
`metric="ip"` uses raw inner-product similarity, without automatic
normalization, and supports only uniform weights.

Non-Flat factories retain their Faiss training requirements. They must
also support the projected feature dimensions encountered during
transform.

### Available donors

`donor_policy="available"` requires `index_factory="Flat"` and either
`metric="l2"` or `metric="nan_euclidean"`.

Partially observed training rows can donate values. A donor must observe
the target feature and share at least one observed, non-empty training
feature with the query.

Neighbors are selected separately for each missing feature. Fewer than
`n_neighbors` usable donors are allowed. Fully missing training rows do
not supply donor values.

### Fallback statistics

Both policies learn column means or medians from all observed training
values before donor filtering.

Rows missing all usable features receive fitted column statistics.
Available mode also uses these statistics when no usable donor exists
for a missing feature.

Fallback statistics are unweighted, including when distance or callable
weights are configured.

## Aggregation and weights

`strategy="mean"` is the default. `strategy="median"` is supported with
uniform weights.

`weights="uniform"` and `weights=None` give every selected donor equal
weight.

`weights="distance"` uses inverse Euclidean distance. If selected donors
include exact zero-distance matches, only those matches contribute.

Non-uniform weights require:

- `strategy="mean"`;
- `metric="l2"` or `metric="nan_euclidean"`.

Distances supplied to weighting are unsquared Euclidean distances.
Missing-feature normalization uses the original input feature count,
including columns that were empty during fit.

### Callable weights

A callable receives a two-dimensional distance array. Each row describes
the selected neighbors for one imputed value.

```python
def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)
```

The callable must return real numeric weights with exactly the same shape.
Its behavior should be independent across rows because batching may change
which rows are passed together.

Unavailable distances are represented by NaN. Weights for unavailable
neighbors and NaN weights are ignored. Contributing weights must be finite,
and each imputed value must have a nonzero weight sum.

Invalid shapes, complex weights, infinite contributing weights, zero
weight sums, and unrepresentable weighted results raise `ValueError`.

## Empty training features

A feature entirely missing during fit is excluded from neighbor selection.

With `keep_empty_features=False`, its imputed output column is dropped.
With `keep_empty_features=True`, its output column is retained and filled
with zero, including when a later query supplies an observed value there.

Transform input must always include the original feature count.

If every training feature is empty, fitting succeeds without donors or
a distance index. Ordinary parameter validation still applies.
Transform returns either zero-valued retained columns or no imputed
columns, followed by any requested indicators.

## Missing indicators

With `add_indicator=True`, indicator features are selected from all
training rows before donor filtering, in original column order.

Each appended value is:

- `1` when the original query entry was missing;
- `0` otherwise.

Indicator features remain fixed until refitting. A feature first missing
only during transform does not receive a new indicator column.

Dropped empty features can still contribute indicators. Indicator values
are captured before imputation or in-place changes.

Names use `missingindicator_<feature_name>`, such as
`missingindicator_age` or `missingindicator_x0`.

Output order is the retained imputed features followed by indicators.
Both `get_feature_names_out()` and pandas output include the added columns.

## Copy behavior

`copy=True` preserves transform input.

With `copy=False`, writable contiguous float32 or float64 arrays can be
reused. Both C-contiguous and Fortran-contiguous arrays are eligible.
Read-only and non-contiguous arrays are copied.

Input conversion, numeric-marker normalization, empty-feature handling,
and output formatting may require additional allocation. Disabling copying
does not guarantee allocation-free operation.

Input may be modified even when appended indicators or output formatting
produce a new returned object.

`fit()` preserves training input regardless of `copy`.
`fit_transform()` includes a transform step and can therefore modify
eligible training input when `copy=False`.

## Feature names and output containers

String column names from pandas training input are retained.
NumPy input uses generated names such as `x0`, `x1`, and `x2`.

Explicit names passed to `get_feature_names_out()` must have the original
input feature count and agree with fitted names when those names exist.

Pandas output uses the transformed feature names and preserves the query
index. Input feature order must agree with the fitted schema.

## Memory and threading

Float64 value arrays require more storage than float32 arrays.

Available mode retains prepared donor data and processes queries in
batches. Batch limits do not bound total process memory.

FaissImputer does not set global native-library thread limits.
Configure threading externally when comparing performance.