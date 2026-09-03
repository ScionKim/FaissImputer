# Real-data imputation pilot

## Scope and reproducibility

This pilot compares four imputers on real feature values with artificial missingness. It is not a clinical evaluation or a scalability benchmark.

- Measured commit: `a3bd1ce3cda6cb3fe4f8d83d7c03f857c51231da`
- Run date: 2026-09-03
- [Raw results](../../benchmarks/results/real-data-a3bd1ce3.json)
- [Benchmark script](../../benchmarks/benchmark_real_data.py)
- Environment: Linux x86_64, Python 3.12.14, faiss-imputer 0.3.1, NumPy 2.5.2, scikit-learn 1.9.0, Faiss 1.15.0.
- Native libraries limited to one thread.

From a checkout of the measured commit with these dependencies installed:

```bash
python -u -m benchmarks.benchmark_real_data --seeds 101 202 303 404 505 --repeats 3
```

The JSON records dependency versions, the dataset fingerprint, mask statistics, quality metrics, and individual timing samples.

## Experimental design

The benchmark uses scikit-learn's bundled [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), loaded with `scaled=False`: 442 rows and 10 features. The target is unused.

Each seed splits the data into 331 training rows and 111 query rows before masking. Age and sex remain observed; the other eight features are eligible for masking.

- **MCAR:** each eligible cell is masked independently with probability 0.25.
- **Selected MAR:** masking probability is 0.125 at or below the training age median and 0.375 above it.

The nominal overall missing rate is 20%; realized rates vary. Both mechanisms reuse the same split and seeded uniform draws.

Scaling is fitted only on observed training values and shared across methods. Inputs and scaled scoring truth use `float32`. RMSE and MAE score only hidden query cells, in standardized units. Different mechanisms have different scored cells and fitted scales, so their scores are not directly controlled comparisons of mechanism difficulty.

The methods are column-mean `SimpleImputer`, uniform-weight `KNNImputer`, and both Faiss donor policies. Neighbor-based methods use five neighbors; Faiss uses L2, Flat, and mean aggregation.

## Accuracy

Values are arithmetic means of the five per-seed scores, not pooled errors across cells. Lower is better.

| Method | MCAR RMSE | MCAR MAE | Selected MAR RMSE | Selected MAR MAE |
|---|---:|---:|---:|---:|
| SimpleImputer | 1.0090 | 0.8060 | 1.0050 | 0.8019 |
| KNNImputer | 0.8299 | 0.6416 | 0.8752 | 0.6836 |
| Faiss complete | 0.8451 | 0.6661 | 0.8319 | 0.6504 |
| Faiss available | 0.8295 | 0.6408 | 0.8740 | 0.6820 |

Available-policy Faiss and KNN have similar average errors here. The small differences do not establish superior accuracy or statistical equivalence.

## Runtime

Each seed uses three repetitions with a fresh estimator. Libraries are warmed beforehand, and method order rotates across repetitions.

The table averages the five per-seed median **fit + transform** times. Preprocessing, scoring, and validation are excluded. Separate fit/transform times, quartiles, and raw samples are retained in the JSON.

| Method | MCAR total (ms) | Selected MAR total (ms) |
|---|---:|---:|
| SimpleImputer | 1.533 | 1.535 |
| KNNImputer | 5.124 | 4.826 |
| Faiss complete | 6.073 | 5.779 |
| Faiss available | 3.328 | 3.627 |

Available-policy Faiss had lower median total time than KNN in all ten scenarios. Ratios of the table's KNN and available times are approximately 1.54 and 1.33, respectively. These are measurements on this small workload, not general speed guarantees.

## Output differences and validation

Available-policy output is **not identical to KNN output**. Maximum absolute differences in standardized units were:

- MCAR, seed 505: 0.767783.
- Selected MAR, seed 202: 0.492505.
- Selected MAR, seed 404: 0.568845.
- Selected MAR, seed 505: 0.745808.

The other six scenarios differed by at most approximately `2.4e-7`.

A separate local donor-level reproduction linked the substantive differences to tied or near-tied neighbors and numerical precision at selection boundaries. Different selected donors can produce substantial imputed-value differences; these are not merely rounding differences in the final output. The JSON records output-difference magnitudes, not donor-level diagnostic traces.

All 40 method validation records passed, and 120 timing samples were collected. Checks cover output shape, finiteness, observed-value preservation, unchanged inputs, separate output storage, Faiss output dtype, and repeatability. No method was inapplicable.

Passing these checks does not require equality to KNN, an accuracy ranking, or a speed threshold. Timing repetitions are not independent accuracy experiments.

## Limitations

This is one small dataset with synthetic masks, one neighbor count, and selected missingness settings. Peak memory was not measured. Results do not establish behavior on naturally missing data, larger datasets, other hardware, or other missingness mechanisms.
