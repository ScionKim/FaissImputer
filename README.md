# FaissImputer

> **Warning:** FaissImputer 0.1.x has a known neighbor-mapping bug that can
> produce incorrect imputations, and it is incompatible with scikit-learn 1.8+.
> Use version 0.2.0 or newer.

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/main/LICENSE)

Impute missing values using Meta's [faiss](https://github.com/facebookresearch/faiss) - A Python library for missing data imputation with k nearest neighbors.

## Unreleased: partial donors

> **Unreleased source-only feature:** The following option and example are
> implemented in this source tree but have not been released on PyPI.
> PyPI 0.2.2 does not include them.

The development version adds `donor_policy`:

- `"complete"` (default): use fully observed training rows, preserving the existing behavior.
- `"available"`: allow partially observed training rows, choosing donors separately for each missing feature.

For `donor_policy="available"`:

- Only `metric="l2"` and `index_factory="Flat"` are supported. Both `strategy="mean"` and `strategy="median"` are available.
- A donor must observe the feature being imputed and share at least one originally observed feature with the query row.
- Donors are ranked by squared L2 distance over shared observed features, scaled by the total feature count divided by the shared feature count.
- Each missing feature uses up to `n_neighbors` eligible donors. Fewer eligible donors are allowed.
- If no eligible donor exists, the fitted column mean or median is used. Entirely missing query rows use these fitted statistics.
- Entirely missing training rows are ignored; entirely missing training columns are rejected.

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

The output is a new NumPy `float32` array; the training and query inputs are not modified.
Performance depends on data size and missingness. This mode is not guaranteed to be faster than `KNNImputer`.

See the [partial-donor benchmark report](https://github.com/ScionKim/FaissImputer/blob/bc1929f58608033bdca565260878e5c8f2a7571f/docs/benchmarks/partial-donors-0a3cc077.md) for measured timings, quality checks, and limitations.

The Installation, Usage, Parameters, and Important behavior sections below describe the released 0.2.2 package.
Earlier release notes and the 0.2.0 benchmark remain historical.

## What's new in 0.2.2

- Clears fitted state after a failed `fit()`, including failed refits.
- Corrects scikit-learn transformer, NaN-support, and `float32` dtype-preservation tags.
- Adds regression tests for failed fits, estimator tags, and pipeline use.

See the [0.2.2 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.2) for details.

## What's new in 0.2.1

Version 0.2.1 is a packaging and documentation hotfix. The imputation behavior is unchanged from 0.2.0.

- Corrects the PyPI description so it no longer presents 0.2.0 as unreleased.
- Uses portable warning markup and absolute documentation links that work on PyPI.
- Verifies built package metadata against the release tag before publication.

See the [0.2.1 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.1) for the packaged changes.

## What's new in 0.2.0

Version 0.2.0 is the corrected release recommended for users upgrading from 0.1.x.

- Corrects the neighbor-to-donor mapping that could return values from the wrong training row.
- Uses the donor rows learned during `fit()` when new data is passed to `transform()`.
- Searches for neighbors using only the columns observed in each query row.
- Preserves the input array and reuses fallback statistics learned during `fit()`.
- Supports modern scikit-learn validation APIs and adds regression tests and automatic CI.

Upgrade an existing installation with:

```bash
python -m pip install --upgrade "faiss-imputer>=0.2.2"
```

See the [0.2.0 release notes](https://github.com/ScionKim/FaissImputer/releases/tag/v0.2.0) and the [full benchmark report](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/docs/benchmarks/v0.2.0.md) for details and current limitations.

## Installation

You can install `faiss-imputer` using `pip`:

FaissImputer requires Python 3.10 or newer.

```bash
pip install faiss-imputer
```

## Usage

```python
import numpy as np

from faiss_imputer import FaissImputer

# Complete rows used as reference donors
X_train = np.array(
    [
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
    ],
    dtype=np.float32,
)

# New rows containing missing values
X_missing = np.array(
    [
        [1.5, np.nan, 150.0],
        [2.5, 25.0, np.nan],
    ],
    dtype=np.float32,
)

imputer = FaissImputer(n_neighbors=2)
imputer.fit(X_train)

X_imputed = imputer.transform(X_missing)
print(X_imputed)
```

`transform()` returns a NumPy `float32` array. The input array is not modified.

## Parameters

- `n_neighbors`: Number of reference rows used to calculate each missing value.
- `metric`: Neighbor-search metric. Supported values are `"l2"` and `"ip"`. Raw inner product is not cosine similarity.
- `strategy`: Aggregation method for neighbor values. Supported values are `"mean"` and `"median"`.
- `index_factory`: Faiss index description. The default `"Flat"` performs exact search.

## Important behavior

- In released 0.2.2, `fit()` uses only rows without any missing values as reference donors.
- Neighbor search compares only the columns observed in each query row.
- `n_neighbors` cannot exceed the number of complete reference rows.
- An entirely missing row is filled using statistics learned during `fit()`.
- Input data must be a two-dimensional numeric array with the same number of columns used during `fit()`.

## Benchmark

In a controlled synthetic benchmark with complete training donors, FaissImputer 0.2.0 and scikit-learn's KNNImputer produced exactly matching imputed values. With 20,000 training rows, FaissImputer transformed repeated missingness patterns up to 19.62 times faster, while performance was effectively tied when almost every test row had a different pattern.

When the training data itself contained missing values, FaissImputer was less accurate because version 0.2.0 uses only fully observed rows as donors. These single-thread results are specific to the tested synthetic data and hardware, not a general performance guarantee.

See the [full 0.2.0 benchmark report](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/docs/benchmarks/v0.2.0.md), [reproduction script](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/benchmarks/benchmark_imputers.py), and [raw results](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/benchmarks/results/v0.2.0-windows-py3.12.json). Planned improvements are tracked in the [roadmap](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/ROADMAP.md).

## Example notebook

See [Imputing Missing Values with Faiss Imputer](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/notebooks/Impute_Missing_Values_with_Faiss_Imputer.ipynb) for a notebook example.

## Contributing

Contributions are welcome! If you find a bug or have an enhancement suggestion, please open an issue or create a pull request.

## Author

- **GitHub:** [@ScionKim](https://github.com/ScionKim/)

## License

This project is licensed under the [MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.2.2/LICENSE).

### Third-Party Licenses

FaissImputer depends on Meta's [Faiss](https://github.com/facebookresearch/faiss), which is distributed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).

FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.

For detailed licensing information of the Faiss library, please refer to the [Faiss repository](https://github.com/facebookresearch/faiss).

