# FaissImputer

> **Warning:** FaissImputer 0.1.x has a known neighbor-mapping bug that can
> produce incorrect imputations, and it is incompatible with scikit-learn 1.8+.
> Use version 0.2.0 or newer.

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/LICENSE)

A scikit-learn-compatible missing-value imputer with [Faiss](https://github.com/facebookresearch/faiss)-backed neighbor search.

## What's new in 0.3.1

- Optimizes the donor-array layout for `donor_policy="complete"` to reduce repeated column-projection overhead.
- Adds a reproducible missingness-pattern benchmark, raw results, and a report.
- Leaves the public API, imputation algorithm, and available-donor policy unchanged.

Performance depends on data size and missingness patterns. Small training sets with many different patterns can still be slower than `KNNImputer`.

See the [0.3.1 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.1) and the [complete-donor benchmark report](https://github.com/ScionKim/FaissImputer/blob/v0.3.1/docs/benchmarks/complete-patterns-9d179b2b.md).

## What's new in 0.3.0

- Adds `donor_policy="available"` to use partially observed training rows as donors.
- Keeps `donor_policy="complete"` as the default, preserving the existing behavior.
- Adds batched NaN-aware distances and numerical safeguards for the available-donor policy.
- Expands regression tests, dependency compatibility checks, and installed-package smoke tests.

See the [0.3.0 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.0).

## Installation

FaissImputer requires Python 3.10 or newer.

```bash
python -m pip install --upgrade "faiss-imputer>=0.3.1"
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

## Benchmark

### Partial donors: development snapshot

The [partial-donor benchmark report](https://github.com/ScionKim/FaissImputer/blob/bc1929f58608033bdca565260878e5c8f2a7571f/docs/benchmarks/partial-donors-0a3cc077.md) includes timings, tolerance-based agreement checks against `KNNImputer`, raw results, and limitations.

It measures development commit `0a3cc077`, not the final 0.3.0 implementation. Subsequent changes include the backend used for `"available"` with fully observed donors.

Performance depends on the donor policy, data size, and missingness pattern. Neither faster execution nor lower memory use than `KNNImputer` is guaranteed. Agreement on the tested cases is not a guarantee of identical results on every dataset.

### Version 0.2.0: historical results

In a controlled synthetic benchmark with complete training donors, FaissImputer 0.2.0 and scikit-learn's KNNImputer produced exactly matching imputed values. With 20,000 training rows, FaissImputer transformed repeated missingness patterns up to 19.62 times faster, while performance was effectively tied when almost every test row had a different pattern.

When the training data itself contained missing values, FaissImputer was less accurate because version 0.2.0 uses only fully observed rows as donors. These single-thread results are specific to the tested synthetic data and hardware, not a general performance guarantee.

See the [full 0.2.0 benchmark report](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/docs/benchmarks/v0.2.0.md), [reproduction script](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/benchmarks/benchmark_imputers.py), and [raw results](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/benchmarks/results/v0.2.0-windows-py3.12.json).

## What's new in 0.2.2

- Clears fitted state after a failed `fit()`, including failed refits.
- Corrects scikit-learn transformer, NaN-support, and `float32` dtype-preservation tags.
- Adds regression tests for failed fits, estimator tags, and pipeline use.

See the [0.2.2 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.2).

## What's new in 0.2.1

Version 0.2.1 is a packaging and documentation hotfix. The imputation behavior is unchanged from 0.2.0.

- Corrects the PyPI description so it no longer presents 0.2.0 as unreleased.
- Uses portable warning markup and absolute documentation links that work on PyPI.
- Verifies built package metadata against the release tag before publication.

See the [0.2.1 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.1).

## What's new in 0.2.0

Version 0.2.0 corrected the imputation behavior of 0.1.x.

- Corrects the neighbor-to-donor mapping that could return values from the wrong training row.
- Uses the donor rows learned during `fit()` when new data is passed to `transform()`.
- Searches for neighbors using only the columns observed in each query row.
- Preserves the input array and reuses fallback statistics learned during `fit()`.
- Supports modern scikit-learn validation APIs and adds regression tests and automatic CI.

See the [0.2.0 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.0).

## Example notebook

See [Imputing Missing Values with Faiss Imputer](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/notebooks/Impute_Missing_Values_with_Faiss_Imputer.ipynb) for a complete-donor example.

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/ScionKim/FaissImputer/issues) or create a pull request. Further work is tracked in the [roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md).

## Author

- **GitHub:** [@ScionKim](https://github.com/ScionKim/)

## License

This project is licensed under the [MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/LICENSE).

### Third-Party Licenses

FaissImputer depends on Meta's [Faiss](https://github.com/facebookresearch/faiss), which is distributed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.
