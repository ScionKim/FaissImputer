# FaissImputer

**KNN imputation for incomplete numerical data with Faiss-backed neighbor search.**

FaissImputer fills missing values using nearby training rows, with a
familiar scikit-learn interface. A *donor* is a training row used to
supply a value for a missing feature.

[![PyPI](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![Python](https://img.shields.io/pypi/pyversions/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![Tests](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml)

[Examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md)
· [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)
· [Benchmarks](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/README.md)

## Why FaissImputer?

As the donor pool grows, finding neighbors can become a major cost of
KNN imputation. FaissImputer brings Faiss-backed search to this workload
and provides options for choosing both the donors and the search method.
Consider it when neighbor search is a bottleneck or when you need
options beyond those exposed by scikit-learn's `KNNImputer`.

- **Choose which rows can contribute.** Use fully observed training rows,
  or draw from partially observed rows separately for each missing
  feature. Complete-donor filtering is an additional policy that
  KNNImputer does not expose.
- **Explore search options for larger donor pools.** Complete-donor mode
  supports trainable or approximate Faiss indexes alongside the default
  Flat index. Available-donor mode requires Flat.

The benefit depends on your workload and configuration; a larger donor
pool does not guarantee a speedup. See the
[performance measurements](https://github.com/ScionKim/FaissImputer#performance)
for measured examples and trade-offs.

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

Choose a donor policy to match your training data:

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

### Transform performance — published 0.3.21

This benchmark compares the published **FaissImputer 0.3.21** release
with **scikit-learn KNNImputer 1.9.1** on one available-donor workload.

![Published FaissImputer 0.3.21 available-donor first-transform benchmark: float32 100.39 ms versus KNNImputer 274.60 ms, float64 119.33 ms versus 332.68 ms; median paired speedups 2.77x and 2.76x, excluding fit.](https://raw.githubusercontent.com/ScionKim/FaissImputer/main/docs/assets/available-transform-0.3.21.png)

The workload used 20,000 training rows, 300 held-out queries, 20 features,
five neighbors, and uniform-weight mean aggregation. Training data had
10% MCAR missingness; each query had four missing features. Measurements
used one native thread on an AMD EPYC 7763 runner.

The chart reports first-transform latency in **milliseconds, excluding
fit**. Times summarize three seeds and three fresh workers per seed,
with a small untimed warmup. Bars show medians and whiskers show the
observed minimum and maximum. Speedups are medians of nine matched
KNNImputer/FaissImputer timing ratios, not ratios of the displayed medians.

Fitting took longer than KNNImputer. Including fit, the median paired
speedup was **2.49×** for both dtypes. Whole-worker peak RSS was slightly
higher than KNNImputer.

Performance varies with donor count, workload size, dtype, missingness,
and configuration. In the archived real-data studies linked below,
available mode was slower than KNNImputer on Wine Quality for both
tested dtypes and on Abalone float64, but faster on Abalone float32.
These studies identify the versions or source commits actually measured.

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

Scale features appropriately before using distance-based imputation.

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