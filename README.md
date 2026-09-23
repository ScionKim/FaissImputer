# FaissImputer

Fast KNN imputation for incomplete numerical data, built for scikit-learn pipelines.

A nearby row is useful for imputation only if it contains the value you
need to fill. Queries can observe different features, and each missing
feature can require different donors. FaissImputer handles these
constraints explicitly, including when the training data is itself incomplete.

[![Tests](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ScionKim/FaissImputer/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![Python](https://img.shields.io/pypi/pyversions/faiss-imputer.svg)](https://pypi.org/project/faiss-imputer/)
[![License](https://img.shields.io/pypi/l/faiss-imputer.svg)](https://github.com/ScionKim/FaissImputer/blob/main/LICENSE)

## Why FaissImputer?

- **Choose your donor policy.** Like sklearn's `KNNImputer`,
  `donor_policy="available"` can use partially observed training rows,
  selecting donors that observe each target feature and computing
  distances from co-observed features. FaissImputer additionally provides
  `donor_policy="complete"`, which restricts donors to fully observed
  training rows — a policy with no equivalent `KNNImputer` option. The
  main difference is computational: FaissImputer uses
  missingness-pattern-grouped Faiss search for complete donors and
  optimized batched matrix distances for available donors, rather than
  sklearn's chunked all-pairs distance computation.
- **Search with Faiss.** Native inner-product search and
  trainable/approximate Faiss index factories are capabilities that
  sklearn's `KNNImputer` does not expose.
- **Prioritize numerical reliability.** FaissImputer includes targeted
  precision and overflow safeguards for distance search and aggregation,
  with regression tests covering numerical edge cases.
- **Keep familiar workflows.** Use fit/transform, scikit-learn pipelines,
  feature names, missing indicators, and optional pandas output.

## Performance

### Separate-query benchmark — published 0.3.20

**A 2.7× first-transform speedup over scikit-learn KNNImputer on one
available-donor workload.**

This comparison measured the published **FaissImputer 0.3.20** package.
It used 20,000 training rows, 300 queries, 20 features, five neighbors,
and uniform-weight mean aggregation. Training data had 10% MCAR
missingness; each query had four missing features.

First-transform medians, **excluding fit**, on an AMD EPYC 7763 runner
with one native thread:

| Input dtype | KNNImputer 1.9.1 | FaissImputer 0.3.20 | Speedup |
| --- | ---: | ---: | ---: |
| float32 | 279.93 ms | 103.40 ms | 2.71× |
| float64 | 334.84 ms | 122.48 ms | 2.73× |

Each method and dtype used three seeds and three fresh workers per seed,
with a small untimed warmup. Fit took longer than KNNImputer in these
cases; fit plus first transform was still 2.44–2.49× as fast.

The full run covered both donor policies and both dtypes.
All benchmark runs passed output and input-preservation checks.
Performance depends on workload, configuration, and hardware.

[Benchmark run and artifacts](https://github.com/ScionKim/FaissImputer/actions/runs/35015576984)
· [Reports, methodology, and historical results](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/README.md)

### Same-data OFAT benchmark — source commit 9683d03

A separate 12-run benchmark sweep measures imputation of the same
incomplete data used for fitting. It covers row counts, feature counts,
missing rates, MCAR/MAR missingness, and neighbor counts using a source
build at commit `9683d03`.

Results are reported separately for `fit_transform` and
`fit_then_transform`, and for float32 and float64. Both API measurements
include fitting. Timings are median [min–max], and speedups are medians
of matched record-level KNNImputer/FaissImputer timing ratios.

FaissImputer's available-donor mode showed similar aggregate
reconstruction RMSE and MAE to KNNImputer under the tested conditions.
This does not establish identical predictions or algorithmic equivalence.
Complete-donor results are reported alongside donor counts and
reconstruction error to make the quality trade-off explicit.

The Intel k=30 observation and Intel 50,000-row stress result are
presented separately from the AMD sweeps. The report also documents
changes in guaranteed complete rows across neighbor settings and the
stress run's single repeat per seed.

[Full report and methodology](docs/benchmarks/fit-transform-ofat-9683d03.md)
· [Raw benchmark results](benchmarks/results/ofat-2026-09-22/)
· [Full-precision analysis summary](benchmarks/results/ofat-2026-09-22-summary.json)

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install --upgrade faiss-imputer
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

For partially observed training data, use
`FaissImputer(donor_policy="available")`.

[More examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md)
cover partial donors, indicators, numeric missing markers, custom metrics,
and pandas pipelines.

## When should I use it?

Use FaissImputer when distance-based imputation suits your numerical data
and you need control over donor eligibility or reuse of fitted donors
across query batches. Scale features appropriately before using distances.

Prefer KNNImputer when its behavior meets your needs and FaissImputer
offers no measured advantage for your workload. FaissImputer is not a
drop-in replacement: defaults, search precision, ties, and some edge
cases differ.

## Donor policies

- **`"complete"` — default:** Use training rows observed in every non-empty
  feature. When non-empty features exist, at least `n_neighbors` complete
  donors are required.
- **`"available"`:** Use partially observed training rows and select donors
  separately for each missing feature. Fewer than `n_neighbors` usable
  donors are allowed.

In available-donor mode with built-in L2 metrics, a donor must observe
the target feature and share an observed feature with the query.
Squared L2 distances use those shared coordinates, scaled by the original
feature count divided by the shared count. Eligibility therefore varies by target
feature, unlike ordinary retrieval from a fixed set of complete vectors.

If no usable donor exists, the fitted column statistic supplies the value.
Available mode requires `index_factory="Flat"`.

## Comparison with KNNImputer

| Behavior | FaissImputer | KNNImputer |
| --- | --- | --- |
| Donors | Complete rows by default; feature-specific partial donors in available mode | Feature-specific donors; other donor features may be missing |
| Built-in search | Faiss-backed search for complete donors; missing-aware search for available donors | Missing-aware pairwise distance computation |
| Default neighbors | 3 | 5 |
| Aggregation | Mean or median; non-uniform weights with mean | Mean with uniform, distance, or callable weights |

Both support fit/transform workflows, pipelines, missing indicators,
feature names, and pandas output through `set_output`.

For comparable configurations, use available donors and matching neighbor
counts, weights, and missing markers. Matching settings does not guarantee
identical output. In particular, built-in complete-donor search uses float32
vectors even when stored values and output are float64.

See the [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)
for precision, copy behavior, callbacks, and edge cases.

## Documentation

- [API reference](https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)
- [Usage examples](https://github.com/ScionKim/FaissImputer/blob/main/docs/usage.md)
- [Benchmark reports and reproduction details](https://github.com/ScionKim/FaissImputer/blob/main/docs/benchmarks/README.md)
- [Migration notes and historical correctness warnings](https://github.com/ScionKim/FaissImputer/blob/main/docs/migration.md)
- [Roadmap](https://github.com/ScionKim/FaissImputer/blob/main/ROADMAP.md)
- [Release history](https://github.com/ScionKim/FaissImputer/releases)

## Contributing

Bug reports and focused pull requests are welcome.
Please include a reproducible example when reporting unexpected behavior
through the [issue tracker](https://github.com/ScionKim/FaissImputer/issues).

Author: [ScionKim](https://github.com/ScionKim).

## License

[MIT License](https://github.com/ScionKim/FaissImputer/blob/main/LICENSE).

Faiss is developed by Meta and distributed under the
[MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
FaissImputer is not affiliated with or endorsed by Meta or the Faiss maintainers.