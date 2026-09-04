# FaissImputer

> **Warning:** FaissImputer 0.1.x has a known neighbor-mapping bug that can
> produce incorrect imputations, and it is incompatible with scikit-learn 1.8+.
> Use version 0.2.0 or newer.

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/LICENSE)

A scikit-learn-compatible missing-value imputer with [Faiss](https://github.com/facebookresearch/faiss)-backed neighbor search.

Current release: [0.3.2](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.2).
See [Releases](https://github.com/ScionKim/FaissImputer/releases) for version history.

## Performance at a glance

FaissImputer can accelerate nearest-neighbor imputation, especially when
queries reuse the same missing-feature patterns.

Selected benchmark results:

| Donor policy | Training rows | Query missingness | KNNImputer | FaissImputer | Speedup |
|---|---:|---|---:|---:|---:|
| complete | 20,000 | One shared pattern | 314.3 ms | 34.8 ms | **9.04×** |
| complete | 20,000 | Random patterns | 316.8 ms | 133.0 ms | **2.38×** |
| complete | 1,000 | Random patterns | 22.6 ms | 30.9 ms | 0.73× — slower |
| available | 500,000 | Random patterns | 9.027 s | 6.179 s | **1.46×** |
| available | 1,000,000 | Random patterns | 21.724 s | 19.681 s | **1.10×** |

**What was measured:** fit + transform on synthetic data, with 300 query
rows, 20 features, 5 neighbors, mean aggregation, and one native thread.
Each query has four missing features. Training data is fully observed
for `complete`, and has 10% MCAR missingness for `available`.

Complete-policy times are medians across five seeds of three-run medians,
measured at commit `9d179b2b`. Available-policy results measure the
128 MiB batching candidate adopted in 0.3.2: three-run medians at
500,000 rows and a **single-run pilot** at 1,000,000 rows.
These are measured development snapshots, not a fresh benchmark of the
published 0.3.2 package. CPU models differ between experiments;
compare methods within each row. Speedups use unrounded times.

**Output agreement:** the complete-policy benchmark matched KNNImputer
exactly on the tested inputs. The available-policy experiments had a
maximum absolute difference of approximately `2.38e-7` from KNNImputer.
This measures agreement, not accuracy against ground truth or a guarantee
for other datasets.

**Tradeoffs:** small datasets with varied missingness can be slower.
Memory use is not always lower: in the million-row pilot, whole-worker
peak RSS was about 9.1% higher than KNNImputer.

[Complete-policy results and reproduction](https://github.com/ScionKim/FaissImputer/blob/v0.3.1/docs/benchmarks/complete-patterns-9d179b2b.md)
· [Available-policy results, memory and thread comparisons](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/available-batching-90c8cfb8.md)

## Installation

FaissImputer requires Python 3.10 or newer.

```bash
python -m pip install --upgrade "faiss-imputer>=0.3.2"
```

## Usage

### Complete donors (default)

Training data must contain enough fully observed rows to supply `n_neighbors` donors.

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

Both policies return a new NumPy `float32` array. The training and query inputs are not modified.

## Parameters

- `n_neighbors` (default: `3`): Positive integer specifying the maximum number of donors used for each missing feature. The complete-donor policy requires at least this many complete training rows.
- `metric` (default: `"l2"`): Supports `"l2"` and `"ip"`. Raw inner product is not cosine similarity. The available-donor policy requires `"l2"`.
- `strategy` (default: `"mean"`): Supports `"mean"` and `"median"` for aggregating donor values and calculating fallback column statistics.
- `index_factory` (default: `"Flat"`): Faiss index description for the complete-donor policy. The available-donor policy accepts only `"Flat"` and uses the distance backend described below.
- `donor_policy` (default: `"complete"`): Use fully observed training rows with `"complete"`, or allow partially observed training rows with `"available"`.

## Important behavior

### Shared behavior

- Inputs must be two-dimensional numeric array-like data, with `NaN` marking missing values. Values are converted to `float32`; infinity is not accepted.
- `transform()` requires the same number of features as `fit()`.
- An entirely missing query row uses column means or medians learned during `fit()`.
- A failed `fit()`, including a failed refit, clears the fitted state.

### Complete-donor policy

- Only fully observed training rows are used as donors. At least one complete row is required.
- `n_neighbors` cannot exceed the number of complete donors.
- Neighbor search uses only the originally observed columns of each query row.
- The default `index_factory="Flat"` performs exact neighbor search.

### Available-donor policy

- Only `metric="l2"` and `index_factory="Flat"` are supported. Both `"mean"` and `"median"` strategies are supported.
- A donor must observe the feature being imputed and share at least one originally observed feature with the query row.
- Donors are ranked by squared L2 distance over shared observed features, scaled by the total feature count divided by the shared feature count.
- Each missing feature uses up to `n_neighbors` eligible donors. Fewer eligible donors are allowed.
- If no eligible donor exists for a feature, its fitted column mean or median is used.
- Entirely missing training rows are ignored; entirely missing training columns are rejected.

The available-donor policy uses batched NaN-aware distances and Faiss neighbor selection, with float64 safeguards for detected numerical risks. This backend is also used when all training donors are complete.

Batching trades memory for throughput. Its internal batch-sizing budget is
not a total process-memory limit: fitted donor data, distance calculations,
and temporary arrays also consume memory. Larger batches can improve speed
but increase peak memory usage. FaissImputer does not set the number of
threads; configure thread limits in the application when needed.

## Benchmarks

Performance depends on donor policy, data size, and missingness patterns. Neither faster execution nor lower memory use than `KNNImputer` is guaranteed. Similar average errors do not imply identical imputed values; ties and numerical precision can affect donor selection.

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

This project is licensed under the [MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.3.2/LICENSE).

### Third-Party Licenses

FaissImputer depends on Meta's [Faiss](https://github.com/facebookresearch/faiss), which is distributed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.
