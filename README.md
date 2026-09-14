# FaissImputer

[![PyPI Version](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/LICENSE)

Nearest-neighbor imputation with Faiss-backed search, scikit-learn pipelines,
feature names, and optional pandas output.

Current release: [0.3.19](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.19).

> FaissImputer 0.1.x has a known neighbor-mapping bug and is incompatible
> with scikit-learn 1.8+. Use version 0.2.0 or newer.

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install --upgrade "faiss-imputer>=0.3.19"
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

By default, inputs are preserved and output is a NumPy array.
Query arrays retain their `float32` or `float64` dtype.
For partially observed training data, use `donor_policy="available"`.

See [usage examples](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/docs/usage.md)
for partial donors, missing indicators, custom markers, callable metrics,
and pandas pipelines.

## Choosing a donor policy

| Policy | Training data and donor selection |
| --- | --- |
| `"complete"` (default) | Uses rows observed in every non-empty training column. Requires at least `n_neighbors` such rows when non-empty columns exist. |
| `"available"` | Allows partially observed rows. Selects eligible donors separately for each missing feature and permits fewer than `n_neighbors` usable donors. |

Available mode requires `index_factory="Flat"` and `metric="l2"`,
`metric="nan_euclidean"`, or a callable. Donors must observe the target
feature and have a defined distance to the query. Built-in L2 metrics
require a shared observed feature. If no usable donor exists, the fitted
column statistic supplies the value.

## Comparison with KNNImputer

The comparison below describes FaissImputer 0.3.15.

| Behavior | FaissImputer | KNNImputer |
| --- | --- | --- |
| Defaults | 3 neighbors, complete donors | 5 neighbors, donors selected per feature |
| Aggregation | Mean or median; non-uniform weights with mean and L2 or callable metrics | Weighted mean |
| Precision | Preserves `float32` and `float64` values; complete-donor search with built-in metrics uses `float32` vectors | Supports floating inputs including `float64` |
| Metrics | `"l2"`, its `"nan_euclidean"` alias, `"ip"` in complete mode, and callables under both donor policies | `"nan_euclidean"` or a callable |
| Missing markers | `NaN` or a finite numeric marker, matched before dtype conversion | Configurable `missing_values` |

Both provide missing indicators, empty-feature retention, and a `copy` option.
Their detailed input and copying rules can differ.

For a closer comparison, use `donor_policy="available"`,
`metric="nan_euclidean"`, and matching neighbor counts, weights, and missing
markers. Matching settings does not guarantee identical donor choices or
imputed values; search precision, distance calculations, and ties matter.

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

Distance and callable weights require `strategy="mean"` and
`metric="l2"`, `metric="nan_euclidean"`, or a callable metric.
Callable metrics require `index_factory="Flat"` and evaluate donors
directly in Python, which can be slower than built-in metrics.

</details>

See the [API reference](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/docs/benchmarks/api.md)
for accepted values, output rules, and edge cases.

## Important behavior

- Inputs must be two-dimensional numeric data. `float32` and `float64`
  arrays retain their dtype; integer arrays are converted to `float32`.
  Observed values must be finite. Output dtype follows the query after
  input conversion.
- Columns entirely missing during fit are dropped by default.
  `keep_empty_features=True` retains them with zero values, including when
  later queries contain observed values in those columns.
- `transform()` requires the original input feature count. Indicator columns
  are selected during fit and remain fixed until refitting.
- `copy=True` preserves inputs. With `copy=False`, writable contiguous
  `float32` or `float64` input may be modified, even when indicators or
  output formatting create a new returned object.
- Entirely missing query rows use fitted column means or medians.
  A failed fit or refit clears the fitted state.

## Benchmarks
### Released-version comparison: 0.3.19

In a single-threaded synthetic benchmark with 20,000 training rows,
300 queries, and 20 features, FaissImputer 0.3.19 achieved 1.17–1.82×
KNNImputer's speed for fit plus the first transform across complete and
available donor policies with float32 and float64 inputs.

Outputs matched FaissImputer 0.3.16 exactly, and median combined times
differed by less than 0.5%. Compared with KNNImputer, peak process RSS
was 36–39% lower in complete mode and 7–23% higher in available mode.

[Full report and conditions](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/released_versions_0.3.19.md)
· [Raw results](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/released_versions_0.3.19.json)

### Historical measurements: 0.3.10

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

- [API reference](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/docs/benchmarks/api.md)
- [Usage examples](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/docs/benchmarks/usage.md)
- [Real-data comparison](https://github.com/ScionKim/FaissImputer/blob/v0.3.10/docs/benchmarks/real-data-a3bd1ce3.md)
- [Example notebook](https://github.com/ScionKim/FaissImputer/blob/v0.3.0/notebooks/Impute_Missing_Values_with_Faiss_Imputer.ipynb)
- [Roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md)
- [Release history](https://github.com/ScionKim/FaissImputer/releases)

## Contributing

Bug reports and pull requests are welcome.
Please use the [issue tracker](https://github.com/ScionKim/FaissImputer/issues).

Author: [ScionKim](https://github.com/ScionKim).

## License

This project is licensed under the
[MIT License](https://github.com/ScionKim/FaissImputer/blob/v0.3.19/LICENSE).

Faiss is developed by Meta and distributed under the
[MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.
