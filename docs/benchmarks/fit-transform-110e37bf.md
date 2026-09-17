# Same-data fit_transform benchmark

[Benchmark index](README.md) ·
[Raw results](../../benchmarks/results/fit-transform-110e37bf.json)

This report compares `fit_transform(X)` with `fit(X)` followed by
`transform(X)` on the same incomplete training data.

The primary workloads contain **10,000 and 20,000 rows**. The 3,000-row
case shows the crossover region; 1,000 rows are retained only as a
small-data regression case.

Results describe the **unreleased source candidate**
`0.3.20+bench.110e37bf9123`, not the published 0.3.20 package.

## Main findings

On the 10,000- and 20,000-row workloads:

- Available-donor FaissImputer completed `fit_transform()` **1.72–2.18×
  as fast as KNNImputer**, with closely matching aggregate reconstruction
  errors.
- Available-donor worker peak RSS medians were **55.4–71.8% lower**.
  These are process measurements, not retained-model memory.
- Complete-donor FaissImputer was **6.57–13.78× as fast**, but used only
  about 12% of training rows as donors and had **38.2–41.7% higher median
  RMSE**. Its speed and quality must be considered together.
- Within each method, both APIs and repeated runs produced identical
  outputs for matching inputs. Their total-time medians differed by
  less than 1% in the primary workloads.

All **432 benchmark runs** passed output and input-preservation checks.
No worker failed, timed out, or remained unrun.

## Conditions

| Item | Configuration |
| --- | --- |
| CPU | AMD EPYC 7763 64-Core Processor |
| Runner | Linux x86-64; 4 logical CPUs visible |
| Native threads | 1 for Faiss and reported numerical thread pools |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Faiss | 1.15.1 |
| Training and query rows | The same 1,000, 3,000, 10,000, or 20,000 rows |
| Features | 20 correlated synthetic numerical features |
| Input dtypes | float32 and float64 |
| Missingness | 10% target probability; the first five rows are kept complete |
| Neighbors and aggregation | 5 neighbors; uniform-weight mean |
| Faiss search | `metric="l2"`, `index_factory="Flat"` |
| Donor policies | Complete and available |
| scikit-learn working memory | 256 MiB |
| Seeds | 101, 202, 303 |
| Repetitions | 3 per seed |
| Samples per size/dtype/method/API | 9 |

All methods receive identical incomplete data for each size, dtype, and
seed. Float64 data is generated without a float32 round trip.

Each measurement uses a fresh sequential worker and a fresh estimator
after a separate small warmup. Method/API order rotates across seeds
and repetitions.

The split API measures fit and transform consecutively, without explicit
garbage collection or memory sampling between them. Reported total times
include both operations.

## Primary timing results

Times are seconds: **median [minimum–maximum]** across nine measurements.
Speed ratios divide the KNNImputer median by the corresponding method
median. Values above 1 indicate a shorter elapsed time.

| Rows | dtype | Method | fit_transform, seconds | KNN / method |
| ---: | --- | --- | ---: | ---: |
| 10,000 | float32 | KNNImputer | 3.195 [3.158–3.287] | 1.00× |
| 10,000 | float32 | FaissImputer complete | 0.486 [0.474–0.493] | 6.57× |
| 10,000 | float32 | FaissImputer available | 1.601 [1.553–1.690] | 2.00× |
| 10,000 | float64 | KNNImputer | 3.670 [3.632–3.702] | 1.00× |
| 10,000 | float64 | FaissImputer complete | 0.523 [0.512–0.530] | 7.01× |
| 10,000 | float64 | FaissImputer available | 2.138 [2.096–2.219] | 1.72× |
| 20,000 | float32 | KNNImputer | 12.785 [12.616–12.968] | 1.00× |
| 20,000 | float32 | FaissImputer complete | 0.982 [0.971–0.993] | 13.02× |
| 20,000 | float32 | FaissImputer available | 5.868 [5.769–6.035] | 2.18× |
| 20,000 | float64 | KNNImputer | 14.652 [14.446–14.930] | 1.00× |
| 20,000 | float64 | FaissImputer complete | 1.064 [1.044–1.075] | 13.78× |
| 20,000 | float64 | FaissImputer available | 7.037 [6.930–7.248] | 2.08× |

The ranges are observed variation, not confidence intervals.

Fit alone was slower for both FaissImputer policies than for KNNImputer
in these primary cases. The advantage came from the overall imputation
operation.

## API comparison

Median total times in seconds:

| Rows | dtype | Method | fit_transform(X) | fit(X), then transform(X) |
| ---: | --- | --- | ---: | ---: |
| 10,000 | float32 | KNNImputer | 3.195 | 3.188 |
| 10,000 | float32 | FaissImputer complete | 0.486 | 0.487 |
| 10,000 | float32 | FaissImputer available | 1.601 | 1.586 |
| 10,000 | float64 | KNNImputer | 3.670 | 3.653 |
| 10,000 | float64 | FaissImputer complete | 0.523 | 0.518 |
| 10,000 | float64 | FaissImputer available | 2.138 | 2.144 |
| 20,000 | float32 | KNNImputer | 12.785 | 12.794 |
| 20,000 | float32 | FaissImputer complete | 0.982 | 0.986 |
| 20,000 | float32 | FaissImputer available | 5.868 | 5.820 |
| 20,000 | float64 | KNNImputer | 14.652 | 14.669 |
| 20,000 | float64 | FaissImputer complete | 1.064 | 1.060 |
| 20,000 | float64 | FaissImputer available | 7.037 | 7.044 |

These measurements show similar performance between the two invocation
styles. They do not establish a separate speed advantage from calling
`fit_transform()`.

## Reconstruction quality

Quality is measured against the hidden values of masked **training
entries**. This is training-data reconstruction, not held-out prediction.

Each cell shows **median RMSE / median MAE**. Lower values indicate
smaller reconstruction errors.

| Rows | dtype | KNNImputer | FaissImputer complete | FaissImputer available |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | float32 | 0.210520 / 0.157676 | 0.298408 / 0.221596 | 0.210520 / 0.157676 |
| 10,000 | float64 | 0.210520 / 0.157676 | 0.298408 / 0.221596 | 0.210520 / 0.157676 |
| 20,000 | float32 | 0.188205 / 0.140668 | 0.260171 / 0.193025 | 0.188204 / 0.140666 |
| 20,000 | float64 | 0.188204 / 0.140666 | 0.260171 / 0.193025 | 0.188204 / 0.140666 |

Complete-donor fitting retained 1,199–1,232 donors at 10,000 rows and
2,417–2,451 donors at 20,000 rows. Its median RMSE was approximately
41.75% and 38.24% higher than KNNImputer, respectively.

Available-donor aggregate errors were close to KNNImputer, but their
full-output hashes differed. Rounded metric agreement does not establish
identical outputs. The JSON does not retain individual imputed values,
so it cannot provide a cross-method maximum absolute output difference.

Within each method, output hashes and compared imputed values matched
across both APIs and repetitions. Checks also covered output shape,
dtype, finite values, preserved observed entries, and unchanged input.

## Process memory

Median worker peak RSS for `fit_transform()`, in MiB:

| Rows | dtype | KNNImputer | FaissImputer complete | FaissImputer available |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | float32 | 561.62 | 145.05 | 185.05 |
| 10,000 | float64 | 695.88 | 149.85 | 196.23 |
| 20,000 | float32 | 570.01 | 151.98 | 253.94 |
| 20,000 | float64 | 719.13 | 163.42 | 261.35 |

Peak RSS includes process setup, data preparation, warmup, fitting,
transformation, and validation. It is sampled before JSON serialization.

These values are neither retained fitted memory nor transform-only
memory peaks. The 256 MiB scikit-learn working-memory setting controls
distance chunking; it does not cap total process memory.

## Crossover case: 3,000 rows

Median `fit_transform()` times in milliseconds:

| dtype | KNNImputer | FaissImputer complete | FaissImputer available |
| --- | ---: | ---: | ---: |
| float32 | 299.40 | 174.12 | 207.01 |
| float64 | 347.61 | 180.42 | 389.11 |

Available mode was faster for float32 at this size, while float64 still
took longer than KNNImputer. At 10,000 and 20,000 rows, available mode was
faster for both dtypes.

This identifies an observed crossover region for these workloads, not
a universal row-count threshold.

## Small-data regression case: 1,000 rows

This case is retained for regression tracking and is excluded from
the primary performance conclusions.

Median `fit_transform()` times in milliseconds:

| dtype | KNNImputer | FaissImputer complete | FaissImputer available |
| --- | ---: | ---: | ---: |
| float32 | 46.09 | 75.89 | 50.41 |
| float64 | 46.44 | 78.88 | 107.91 |

The earlier [small-data pilot](../../benchmarks/results/fit-transform-ed7f92fc.json)
used an AMD EPYC 9V74 runner. Its timings are not pooled with this run.

## Provenance and reproduction

- Source commit: `110e37bf9123577786934f4cacb316000a18fa9e`
- Candidate version: `0.3.20+bench.110e37bf9123`
- Wheel SHA256: `ef1f11ea65d1f7c9c2ae03daa7c137b01db2d81ad39525c1920250e8d39a5176`
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/35205569911), attempt 1
- [Recorded workflow](https://github.com/ScionKim/FaissImputer/blob/110e37bf9123577786934f4cacb316000a18fa9e/.github/workflows/benchmark-fit-transform.yml)
- [Raw results](../../benchmarks/results/fit-transform-110e37bf.json)

The workflow builds a candidate wheel from the recorded source, installs
it in an isolated environment, and runs benchmark code outside the source
checkout. Its artifact contains the wheel, provenance, installation
report, dependency versions, and result JSON.

Using that candidate environment and its `candidate-build.json`, the
measurement command is:

```bash
python -u -m benchmarks.benchmark_fit_transform \
  --expected-version "0.3.20+bench.110e37bf9123" \
  --provenance /path/to/candidate-build.json \
  --sizes 1000 3000 10000 20000 \
  --seeds 101 202 303 \
  --repeats 3 \
  --timeout-seconds 300 \
  --budget-seconds 5400 \
  --output fit_transform.json
```

The complete run took approximately 25.4 minutes, including worker
startup and validation.

## Scope

These results cover one synthetic data family, 20 features, roughly
10% missingness, five neighbors, uniform weights, and one native thread.

They do not establish performance for real-world datasets, other
missingness mechanisms, callable metrics, other index factories,
multi-threaded execution, or larger datasets.

The first five rows are deliberately kept complete, so the missingness
pattern is not an unrestricted MCAR sample.

All performance comparisons above use matching inputs and the same
runner within this execution. Published-package performance claims
remain separate from this development benchmark.