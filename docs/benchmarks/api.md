# API reference

This document describes FaissImputer 0.3.15.

[README](../README.md) · [Usage examples](usage.md)

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

| Parameter | Accepted values and behavior |
| --- | --- |
| `n_neighbors` | Positive Python or NumPy integer. Booleans are rejected. Complete mode requires enough complete donors when non-empty training columns exist. |
| `metric` | `"l2"`, `"nan_euclidean"` as an alias for `"l2"`, or `"ip"` in complete mode. Callable metrics are not supported. |
| `strategy` | `"mean"` or `"median"` for neighbor aggregation and fallback statistics. |
| `index_factory` | A Faiss index-description string. Default: `"Flat"`. Available mode accepts only `"Flat"`. |
| `donor_policy` | `"complete"` or `"available"`. See donor selection below. |
| `weights` | `"uniform"`, `None`, `"distance"`, or a callable. Non-uniform weights require mean aggregation and an L2 metric. |
| `add_indicator` | Boolean. Append missingness indicators selected during fit. |
| `keep_empty_features` | Boolean. Retain entirely missing training columns with zero values when enabled. |
| `missing_values` | `NaN` or a finite Python/NumPy integer or floating-point scalar. Boolean, string, complex, `None`, and infinite markers are rejected. |
| `copy` | Boolean. Preserve transform inputs by default; permit input reuse when disabled. |

## Methods

### fit(X, y=None)

Learn donors, fallback statistics, feature selection, and optional indicators.
Returns the fitted estimator. `y` is ignored.

Input must be two-dimensional numeric data with at least one row and column.
Training columns may be entirely missing.

`fit()` does not modify training data, including with `copy=False`.
A failed fit or refit clears the fitted state; subsequent transformation
requires a successful fit.

### transform(X)

Impute queries using the fitted donors and statistics.
Queries must contain the original number of input features, including columns
that were entirely missing during fit. When fitted feature names are available,
named inputs must match those names and their order.

Output is a NumPy `float32` array by default. Output width can differ from input
width because of empty-column removal and missing indicators.

### fit_transform(X, y=None)

Fit and then transform the same data. With `copy=False`, the transform step
can modify the supplied training array after fitting.

### get_feature_names_out(input_features=None)

Return output names in column order. Named training inputs retain their feature
names; unnamed inputs use `x0`, `x1`, and so on. Explicit names must match the
fitted schema.

Dropped empty columns are excluded. Indicator names follow the imputed columns.
This method requires a successful fit.

### set_output(transform="pandas")

Return pandas DataFrames, preserving the query index and using
`get_feature_names_out()` for column names. Pandas is an optional dependency.

Use `set_output(transform="default")` to return NumPy arrays.
Output selection also works through scikit-learn pipelines.

## Missing values and precision

With `missing_values=np.nan`, NaN entries are missing. With a numeric marker,
exact equality determines missingness before float32 conversion. Distinct
observed values remain observed even if they later round to the same float32
value as the marker.

Use the same marker when preparing training and query data. Actual NaN entries
are rejected when a numeric marker is configured. Refit after changing the
marker.

A numeric marker can exceed the float32 range because matching entries are
removed before conversion. Observed values must remain finite after conversion;
infinity and overflowing observations are rejected.

Imputation outputs remain float32. Internal float64 calculations and numerical
safeguards do not provide float64 input preservation or guarantee identical
neighbor ordering to KNNImputer.

## Donor selection and fallback

### Complete donors

- Use training rows observed in every non-empty training column.
- Require at least `n_neighbors` such rows when non-empty columns exist.
- Search only the originally observed, non-empty features of each query.
- Use fitted column statistics when all usable query features are missing.

`"Flat"` performs exact search on the computed vectors. Other factories may
require training and must support the dimensions used during fit and projected
query searches. Faiss configuration or training errors can propagate.

`metric="ip"` selects by raw inner product, not cosine similarity, and supports
only uniform weighting.

### Available donors

- Require `index_factory="Flat"` and an L2 metric.
- Permit partially observed training rows.
- For each target feature, donors must observe that feature and share at least
  one usable observed feature with the query.
- Use up to `n_neighbors` eligible donors; fewer are allowed.
- Fall back to the fitted column statistic if no usable neighbor exists.
- Ignore training rows missing every non-empty feature.

Distances use squared differences over shared observed features, scaled by the
original input feature count divided by the shared feature count.

Fallback statistics use all observed training values in each non-empty column.
They remain unweighted and follow `strategy`, including when neighbor
aggregation uses distance or callable weights.

## Weights

`"uniform"` and `None` use unweighted mean or median aggregation.

`"distance"` uses inverse Euclidean distance with mean aggregation.
If selected donors include distance-zero matches, only those selected matches
contribute to the imputed value.

Callable weights receive a two-dimensional array of unsquared, NaN-aware
Euclidean distances and must return real weights with the same shape.
Each row describes neighbors for an imputed value. Functions should operate
independently on each row, without depending on batch size or neighbor order.

Unavailable neighbors have NaN distances and are excluded regardless of the
returned weight. Returned NaN weights contribute zero.

Invalid shapes, complex weights, infinite weights for usable donors, zero
weight sums, or weighted results outside the finite float32 range raise
`ValueError`.

Distance and callable weights require `strategy="mean"` and either
`metric="l2"` or `metric="nan_euclidean"`.

## Empty training columns

Columns entirely missing during fit do not participate in donor selection
or neighbor search.

- `keep_empty_features=False`: omit those columns from imputed output.
- `keep_empty_features=True`: retain them in their original positions and
  always fill them with zero, even if queries provide observed values.

The input schema remains fixed until refitting. Transform still accepts the
full original feature set.

If every training column is empty, fitting succeeds without a neighbor index.
The imputed portion of the output has zero columns by default, or contains
zeros when retention is enabled. The donor-count minimum does not apply,
but ordinary parameter and input validation still applies.

Distance normalization retains the original input feature count, including
for callable weights.

## Missing indicators

`add_indicator=True` appends 1 for originally missing query entries and 0
otherwise. Features are selected from all training rows before donor filtering,
in original column order.

A feature first missing only during transform does not gain an indicator.
Selection remains fixed until refitting.

Names follow `missingindicator_<feature_name>`, such as
`missingindicator_age` or `missingindicator_x0`. Both output feature names and
pandas columns include these additions.

Indicators describe original query missingness, including for columns dropped
or zero-filled by empty-feature handling. Retained imputed columns come first,
followed by indicator columns.

## Input copying

`copy=True` preserves inputs during transform.

With `copy=False`, writable C-contiguous or F-contiguous float32 arrays can be
imputed in place. Read-only or non-contiguous arrays are copied. Data type
conversion, numeric missing-marker normalization, and empty-column handling
can also require new arrays.

Indicators are computed before any input values are modified. Appending them
creates a new output array, but the input may already have been imputed in
place. A newly allocated output therefore does not guarantee input preservation.

`fit()` itself preserves training data. `fit_transform()` can modify it during
the transform step.

## Memory and threading

Available mode reuses donor-side arrays prepared during fit and processes
queries in batches. This reduces repeated preparation work while increasing
retained fitted memory.

Internal batch-sizing budgets are not total process-memory limits. Donor
storage, distance calculations, and temporary arrays also consume memory.
Neither lower peak memory nor faster execution is guaranteed for every workload.

FaissImputer does not set the number of threads. Configure thread limits in
the application when needed.