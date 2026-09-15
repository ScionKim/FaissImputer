# Usage examples

These examples describe FaissImputer 0.3.20.

[README](../README.md) · [API reference](api.md)

The examples below use these imports:

```python
import numpy as np
from faiss_imputer import FaissImputer
```

## Complete donors

The default policy uses training rows observed in every non-empty column.
Provide at least `n_neighbors` such rows.

```python
train = np.array(
    [[0, 10], [2, 20], [4, 40]],
    dtype=np.float32,
)
query = np.array([[1.8, np.nan]], dtype=np.float32)

imputer = FaissImputer(n_neighbors=1).fit(train)
print(imputer.transform(query))
# [[ 1.8 20. ]]
```

By default, the query is preserved and the result is a new float32 array.

## Partially observed donors

Available mode selects donors separately for each missing feature.
No fully observed training row is required.

```python
train = np.array(
    [
        [0, 10, np.nan],
        [2, 30, np.nan],
        [1, np.nan, 20],
        [3, np.nan, 40],
    ],
    dtype=np.float32,
)
query = np.array(
    [[0.1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    dtype=np.float32,
)

imputer = FaissImputer(
    n_neighbors=1,
    donor_policy="available",
).fit(train)

print(imputer.transform(query))
# [[ 0.1 10.  20. ]
#  [ 1.5 20.  30. ]]
```

The entirely missing query uses fitted column means.
Set `strategy="median"` to use medians instead.

## Numeric markers, empty columns, and indicators

This example uses `-1` as the missing marker. The second training column
is entirely missing and is retained with zero values.
Indicators record the original query missingness.

```python
train = np.array(
    [[0, -1, 10], [2, -1, 20], [4, -1, -1]],
    dtype=np.float32,
)
query = np.array(
    [[0.2, -1, -1], [3, 99, 30]],
    dtype=np.float32,
)

imputer = FaissImputer(
    n_neighbors=1,
    donor_policy="available",
    missing_values=-1,
    keep_empty_features=True,
    add_indicator=True,
).fit(train)

print(imputer.transform(query))
# [[ 0.2  0.  10.   1.   1. ]
#  [ 3.   0.  30.   0.   0. ]]

print(imputer.get_feature_names_out().tolist())
# ['x0', 'x1', 'x2', 'missingindicator_x1', 'missingindicator_x2']
```

The query value `99` becomes zero because its column was empty during fit.
Its indicator remains zero because `99` was originally observed.

With `keep_empty_features=False`, that imputed column is dropped while
its indicator remains. Actual NaN entries are rejected when a numeric
missing marker is configured.

## Distance and callable weights

Distance weighting requires mean aggregation and either an L2 metric or a
callable metric. `"nan_euclidean"` is an alias for the existing `"l2"` path.

```python
train = np.array([[0, 10], [2, 20]], dtype=np.float32)
query = np.array([[0.5, np.nan]], dtype=np.float32)

imputer = FaissImputer(
    n_neighbors=2,
    metric="nan_euclidean",
    weights="distance",
).fit(train)

print(imputer.transform(query))
# [[ 0.5 12.5 ]]
```

A callable can supply custom weights with the same shape as its distance input:

```python
def shifted_inverse(distances):
    return 1.0 / (1.0 + distances)

imputer.set_params(weights=shifted_inverse).fit(train)
result = imputer.transform(query)
```

Callable weights should operate independently on each row.
See the [weight rules](api.md#callable-weights) for unavailable neighbors and validation.

## Callable metrics

A custom metric receives the query row first and donor row second.
Missing entries are normalized to NaN before the callback runs.

This example uses Manhattan distance over shared observed features:

```python
def nan_manhattan(x, y, *, missing_values=np.nan):
    shared = ~np.isnan(x) & ~np.isnan(y)
    if not shared.any():
        return np.nan
    return np.abs(
        x[shared].astype(np.float64)
        - y[shared].astype(np.float64)
    ).sum()


train = np.array(
    [[0, 3, 10], [2, 2, 20]],
    dtype=np.float64,
)
query = np.array([[0, 0, np.nan]], dtype=np.float64)

imputer = FaissImputer(
    n_neighbors=1,
    metric=nan_manhattan,
).fit(train)

print(imputer.transform(query))
# [[ 0.  0. 10.]]
```

The first donor has Manhattan distance 3; the second has distance 4.

Callable metrics support both donor policies and require
`index_factory="Flat"`. Return a nonnegative finite distance, or `np.nan`
to exclude a donor whose distance is undefined.

Distance and callable weights use the returned distances directly,
without squaring or missing-feature normalization.

Callbacks receive the original feature count and column order.
Fit-time empty columns are represented by NaN in both rows.
Direct evaluation of Python callbacks can be slower than built-in metrics.

See the [callable metric rules](api.md#callable-metrics) for validation,
tie handling, and fallback behavior.

## Reusing query storage

`copy=False` permits in-place imputation when the input can be reused.

```python
train = np.array([[0, 10], [2, 20]], dtype=np.float32)
query = np.array([[0.25, np.nan]], dtype=np.float32)

imputer = FaissImputer(n_neighbors=1, copy=False).fit(train)
result = imputer.transform(query)

print(result is query)
# True

print(query)
# [[ 0.25 10.  ]]
```

This example uses a writable contiguous float32 array. Other inputs may
require copying or conversion.

With `add_indicator=True`, the returned array can be newly allocated even
though the input has already been modified. Use `copy=True` when the original
query must be preserved.

`fit()` itself preserves training data. With `copy=False`, `fit_transform()`
can modify it during the transform step.

## Pipeline and pandas output

Pandas is optional:

```bash
python -m pip install pandas
```

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "age": [20.0, None, 40.0],
    "income": [100.0, 190.0, 300.0],
})

pipeline = make_pipeline(
    FaissImputer(n_neighbors=1),
    StandardScaler(),
).set_output(transform="pandas")

result = pipeline.fit_transform(data)

print(pipeline.get_feature_names_out().tolist())
# ['age', 'income']

print(result.columns.tolist())
# ['age', 'income']
```

Use `pipeline.set_output(transform="default")` to return NumPy arrays.
For unnamed array inputs, generated feature names are `x0`, `x1`, and so on.

When indicators are enabled, their names follow the retained imputed columns.
See the [API reference](api.md) for output shapes and empty-feature handling.

## Float64 precision

Float64 training arrays retain their precision in donor values and fitted
statistics. Output dtype follows the query dtype after input conversion.

```python
import numpy as np
from faiss_imputer import FaissImputer

train = np.array(
    [[0, 16777217], [2, 16777219]],
    dtype=np.float64,
)
query = np.array([[0.25, np.nan]], dtype=np.float64)

imputer = FaissImputer(n_neighbors=1)
result = imputer.fit(train).transform(query)

print(result.dtype)
# float64

print(result[0, 1])
# 16777217.0
```

Writable contiguous float64 arrays are also eligible for reuse with
`copy=False`. Input conversion and output formatting may still require
allocation.

With built-in metrics, complete-donor search uses float32 vectors even
when donor values and output are float64. Callable metrics receive rows
in their normalized input dtypes and do not use float32 search vectors.

Dtype preservation does not guarantee identical neighbor choices to
`KNNImputer`.

See [search precision and numeric limits](api.md#search-precision-and-numeric-limits)
for the supported distance ranges and error behavior.