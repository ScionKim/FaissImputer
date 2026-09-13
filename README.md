# FaissImputer

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/LICENSE)

Nearest-neighbor imputation with Faiss-backed search, scikit-learn pipelines,
feature names, and optional pandas output.

Current release: [0.3.15](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.15).

> FaissImputer 0.1.x has a known neighbor-mapping bug and is incompatible
> with scikit-learn 1.8+. Use version 0.2.0 or newer.

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install --upgrade "faiss-imputer>=0.3.15"
```

## Quick start

```python
import numpy as np
from faiss_imputer import FaissImputer

train = np.array([[0, 10], [2, 20], [4, 40]], dtype=np.float32)
query = np.array([[1.8, np.nan]], dtype=np.float32)

imputer = FaissImputer(n_neighbors=1)
print(imputer.fit(train).transform(query))
# [[ 1.8 20. ]]
```

By default, inputs are preserved and output is a NumPy `float32` array.
For partially observed training data, use `donor_policy="available"`.

See [usage examples](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/docs/usage.md)
for partial donors, missing indicators, custom markers, and pandas pipelines.

## Choosing a donor policy

| Policy | Training data and donor selection |
| --- | --- |
| `"complete"` (default) | Uses rows observed in every non-empty training column. Requires at least `n_neighbors` such rows when non-empty columns exist. |
| `"available"` | Allows partially observed rows. Selects eligible donors separately for each missing feature and permits fewer than `n_neighbors` usable donors. |

Available mode requires `index_factory="Flat"` and either `metric="l2"`
or `metric="nan_euclidean"`. A donor must observe the target feature and
share an observed feature with the query. If no usable neighbor exists,
the fitted column statistic supplies the value.

## Comparison with KNNImputer

The comparison below describes FaissImputer 0.3.15.

| Behavior | FaissImputer | KNNImputer |
| --- | --- | --- |
| Defaults | 3 neighbors, complete donors | 5 neighbors, donors selected per feature |
| Aggregation | Mean or median; weights supported with mean and L2 metrics | Weighted mean |
| Precision | Converts inputs and outputs to `float32` | Supports floating inputs including `float64` |
| Metrics | `"l2"`, its `"nan_euclidean"` alias, and `"ip"` in complete mode | `"nan_euclidean"` or a callable |
| Missing markers | `NaN` or a finite numeric marker | Configurable `missing_values` |

Both provide missing indicators, empty-feature retention, and a `copy` option.
Their detailed input and copying rules can differ.

For a closer comparison, use `donor_policy="available"`,
`metric="nan_euclidean"`, and matching neighbor counts, weights, and missing
markers. Matching settings does not guarantee identical donor choices or
imputed values; float32 conversion, distance calculations, and ties matter.

## Parameters

<details>
<summary>Parameter names and defaults</summary>

| Parameter | Default |
| --- | --- |
| `n_neighbors` | `3` |
| `metric` | `"l2"` |
| `strategy` | `"mean"` |
| `index_factory` | `"Flat"` |
| `donor_policy` | `"complete"` |
| `weights` | `"uniform"` |
| `add_indicator` | `False` |
| `keep_empty_features` | `False` |
| `missing_values` | `np.nan` |
| `copy` | `True` |

Distance and callable weights require `strategy="mean"` and either
`metric="l2"` or `metric="nan_euclidean"`.

</details>

See the [API reference](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/docs/api.md)
for accepted values, output rules, and edge cases.

## Important behavior

- Inputs must be two-dimensional numeric data. Observed values are converted
  to `float32`; infinity and values outside its finite range are rejected.
- Columns entirely missing during fit are dropped by default.
  `keep_empty_features=True` retains them with zero values, including when
  later queries contain observed values in those columns.
- `transform()` requires the original input feature count. Indicator columns
  are selected during fit and remain fixed until refitting.
- `copy=True` preserves inputs. With `copy=False`, eligible inputs may be
  modified even when added indicators cause a new output array to be returned.
- Entirely missing query rows use fitted column means or medians.
  A failed fit or refit clears the fitted state.

## Benchmarks

The following historical measurements compare **PyPI FaissImputer 0.3.10**
with **KNNImputer from scikit-learn 1.9.0**.

The AMD runner used one thread, 20,000 training rows, 300 queries,
20 features, five neighbors, and uniform-weight mean aggregation.
Each query had four missing features. Complete-mode training data was
fully observed; available-mode training data had approximately 10% MCAR
missingness.

Times are first-transform medians across three seeds and three fresh runs
per seed, excluding fit time.

| Donor policy | Query missingness | KNNImputer | FaissImputer | Speedup |
| --- | --- | ---: | ---: | ---: |
| complete | One shared pattern | 422.4 ms | 22.0 ms | 19.24× |
| complete | Random patterns | 409.8 ms | 107.2 ms | 3.82× |
| available | One shared pattern | 391.4 ms | 166.2 ms | 2.35× |
| available | Random patterns | 385.6 ms | 167.5 ms | 2.30× |

[AMD results and environment](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/scaling-threads-34297607304.json)

Performance varies with workload. A separate Intel training-size sweep
included a slower case: complete mode with random query patterns at
1,000 training rows achieved **0.71×** KNNImputer's speed. Memory advantages
also varied with data size.

[Intel results and environment](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/scaling-threads-34310124369.json)

These measurements apply to 0.3.10 and the recorded hardware and workloads.
See the [benchmark reports](https://github.com/ScionKim/FaissImputer/tree/main/docs/benchmarks)
for detailed conditions, quality comparisons, and historical experiments.

## Documentation

- [API reference](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/docs/api.md)
- [Usage examples](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/docs/usage.md)
- [Real-data comparison](https://github.com/ScionKim/FaissImputer/blob/v0.3.10/docs/benchmarks/real-data-a3bd1ce3.md)
- [Example notebook](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/notebooks/Impute_Missing_Values_with_Faiss_Imputer.ipynb)
- [Roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md)
- [Release history](https://github.com/ScionKim/FaissImputer/releases)

## Contributing

Bug reports and pull requests are welcome.
Please use the [issue tracker](https://github.com/ScionKim/FaissImputer/issues).

Author: [Hakkil Kim / ScionKim](https://github.com/ScionKim/).

## License

This project is licensed under the
[MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.3.15/LICENSE).

Faiss is developed by Meta and distributed under the
[MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.