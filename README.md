# FaissImputer

**Fast KNN imputation with control over which rows can help.**

A nearby row is useful only if it contains the value you need.
FaissImputer lets you choose between complete donors and partially
observed donors, with a familiar scikit-learn interface.

[![PyPI](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![Python](https://img.shields.io/pypi/pyversions/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![Tests](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml)

[Examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md)
· [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)
· [Benchmarks](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/README.md)

## Why FaissImputer?

- **Choose donors to suit your data.** Use fully observed training rows,
  or draw from partially observed rows separately for each missing
  feature. Complete-donor filtering is an additional policy that
  KNNImputer does not expose.
- **Get more search options.** Complete-donor mode supports native
  inner-product search and trainable or approximate Faiss indexes,
  alongside the default Flat index.
- **Account for numerical edge cases.** Distance search and aggregation
  include targeted precision and overflow safeguards, backed by
  regression tests. Their scope and limits are documented in the
  [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md).

![Published FaissImputer 0.3.21 available-donor first-transform benchmark: float32 100.39 ms versus KNNImputer 274.60 ms, float64 119.33 ms versus 332.68 ms; median paired speedups 2.77x and 2.76x, excluding fit.](https://raw.githubusercontent.com/ScionKim/FaissImputer/main/docs/assets/available-transform-0.3.21.png)

[See the benchmark conditions, results, and trade-offs](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/released_versions_0.3.21.md).

## Installation

Install with Python 3.10 or newer:

```bash
python -m pip install faiss-imputer
```

## Quick start

Fit on your training data, then fill the gaps in new rows:

```python
import numpy as np
from faiss_imputer import FaissImputer

train = np.array([[0, 10], [2, 20], [4, 40]], dtype=np.float32)
query = np.array([[1.8, np.nan]], dtype=np.float32)

imputer = FaissImputer(n_neighbors=1)
result = imputer.fit(train).transform(query)

print(result)
# [[ 1.8 20. ]]
```

Work with NumPy arrays or pandas DataFrames, use scikit-learn pipelines,
and retain feature names and optional missing-value indicators.
[See more examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md).

## Choose which rows can help

A nearby row can fill a missing value only if it contains that value.
FaissImputer gives you two ways to choose those *donors*:

| Policy | Which rows contribute? |
| --- | --- |
| `complete` — default | Training rows observed in every non-empty feature. At least `n_neighbors` eligible complete rows are required when non-empty features exist. |
| `available` | Partially observed rows can contribute to the features they contain. With built-in L2 metrics, they must also share an observed feature with the query. |

For incomplete training data, start with available donors:

```python
imputer = FaissImputer(n_neighbors=5, donor_policy="available")
```

Available mode requires `index_factory="Flat"`, permits fewer than
`n_neighbors` usable donors, and falls back to a fitted column statistic
when no donor is usable. Changing the policy changes the donor pool and
can change reconstruction quality.

You can also choose mean or median aggregation, configure weights, or
provide a custom distance function. The
[API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)
explains the supported combinations, precision safeguards, and edge cases.

## Performance

### Published 0.3.21: a separate-query comparison

**2.76–2.77× the first-transform speed of KNNImputer on one
available-donor workload.** These measurements are from the published
FaissImputer **0.3.21** package.

The workload used 20,000 training rows, 300 held-out queries, 20 features,
five neighbors, and uniform-weight mean aggregation. Training data had
10% MCAR missingness; each query had four missing features. Measurements
used one native thread on an AMD EPYC 7763 runner.

First-transform times in milliseconds, **excluding fit**:

| Input | KNNImputer 1.9.1 | FaissImputer 0.3.21 | Paired speedup |
| --- | ---: | ---: | ---: |
| float32 | 274.60 [272.58–282.91] | 100.39 [97.96–104.07] | 2.77× |
| float64 | 332.68 [320.05–341.91] | 119.33 [117.30–126.13] | 2.76× |

Times are median [min–max] across three seeds and three fresh workers per
seed, with a small untimed warmup. Speedups are medians of the nine matched
KNNImputer/FaissImputer timing ratios, not ratios of the displayed medians.

Fitting took longer than KNNImputer. Fit plus first transform was **2.49×**
as fast for both dtypes, while whole-worker peak RSS was slightly higher.
Performance depends on the workload: available mode was slower than
KNNImputer on Wine Quality for both tested dtypes and on Abalone for
float64 in the real-data study linked below.

[Full comparison and methodology](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/released_versions_0.3.21.md)
· [Raw measurements](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/released_versions_0.3.21.zip)
· [Unrounded summary](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/released_versions_0.3.21-summary.json)

### Explore other workloads

Each report identifies the version or source commit actually measured
and links to its raw evidence and reproduction instructions.

| Study | What it covers |
| --- | --- |
| [Scaling and missingness](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/fit-transform-ofat-9683d03.md) · `9683d03` | A 12-run same-data sweep across rows, features, missingness, and neighbors. APIs and dtypes are reported separately. |
| [Wine Quality and Abalone](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/real-data-datasets-ef04b1b.md) · `ef04b1b` | Held-out real-data comparisons covering speed, memory, reconstruction error, and donor counts. |
| [Float64 refinement](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/available-selected-distances-c02b71d.md) · `c02b71d` | A source-build optimization comparison with matched timings and a separate same-code control. |

Similar RMSE or MAE values describe similar aggregate reconstruction
error on the tested data. They do not establish identical predictions
or algorithmic equivalence.

## Is it a fit for your project?

Use FaissImputer when nearest-neighbor imputation suits your numerical
data and you want control over donor eligibility, aggregation, or Faiss
search options. Scale features appropriately before using distances.

FaissImputer is not a drop-in replacement for KNNImputer. Defaults,
search precision, ties, and some edge cases differ. For KNNImputer
comparisons, use `donor_policy="available"` and match the neighbor count,
weights, and missing markers.
KNNImputer remains a good choice when its behavior meets your needs and
FaissImputer offers no measured advantage for your workload.

## Learn more

- [Usage examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md) — partial donors, indicators, custom metrics, and pandas pipelines.
- [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md) — parameters, numerical behavior, and memory considerations.
- [Benchmark reports](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/README.md) — measurements, limitations, and reproducible evidence.
- [Migration notes](https://github.com/ScionKim/FaissImputer/blob/main/docs/migration.md) — compatibility changes and historical correctness notes.
- [Roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md) · [Release history](https://github.com/ScionKim/FaissImputer/releases).

## Contributing

Bug reports and focused pull requests are welcome. A small, reproducible
example helps us understand unexpected behavior.
[Open an issue](https://github.com/ScionKim/FaissImputer/issues).

Created by [ScionKim](https://github.com/ScionKim).

## License

FaissImputer is released under the
[MIT License](https://github.com/ScionKim/FaissImputer/blob/main/LICENSE).
Faiss is developed by Meta and distributed under the
[MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.