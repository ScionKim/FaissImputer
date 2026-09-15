# Available-donor distance buffer benchmark

This benchmark compares released FaissImputer 0.3.19 with an unpublished
candidate that reuses distance work buffers and uses floating-point matrix
multiplication to count shared observed features.

Available-donor transform time decreased by 38–41% in this run.
Measured imputation outputs were unchanged from 0.3.19.

## Versions and conditions

- Baseline: PyPI FaissImputer 0.3.19.
- Candidate: `0.3.19+bench.94adbf660f59`.
- Candidate source commit: `94adbf660f59f5dd7a03691af8a6f71d99a7d467`.
- CPU: AMD EPYC 9V74; one native thread.
- Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, Faiss 1.15.0.
- Training data: 20,000 rows and 20 features.
- Queries: 300 rows, each missing four randomly selected features.
- Complete policy: fully observed training data.
- Available policy: training data with 10% MCAR missingness.
- Both float32 and float64; float64 data generated without a float32 round trip.
- Five neighbors, mean aggregation, uniform weights, Flat L2 search.
- Seeds: 101, 202, and 303; three fresh workers per method and seed.
- Workers ran sequentially, with method order rotated across repetitions.
- Each worker measured the first transform and two additional transforms.

Each table entry is the median of nine workers. Total time is fit plus the
first transform, calculated per worker before taking the median.
Timing excludes process startup, data generation, warmup, and validation.

## Timing

| Policy | dtype | 0.3.19 transform | Candidate transform | 0.3.19 total | Candidate total |
| --- | --- | ---: | ---: | ---: | ---: |
| complete | float32 | 108.4 ms | 108.6 ms | 112.3 ms | 112.5 ms |
| complete | float64 | 135.8 ms | 136.1 ms | 141.4 ms | 141.4 ms |
| available | float32 | 168.0 ms | 99.4 ms | 178.2 ms | 108.8 ms |
| available | float64 | 180.1 ms | 110.9 ms | 191.2 ms | 121.3 ms |

For available donors:

- First-transform time decreased by 40.8% for float32 and 38.4% for float64.
- Total time decreased by 39.0% for float32 and 36.6% for float64.
- Repeated-transform time decreased by 42.6% for float32 and 39.9% for float64.

Complete-policy total time differed by less than 0.2%.

## Memory

| Available-policy dtype | 0.3.19 peak RSS | Candidate peak RSS |
| --- | ---: | ---: |
| float32 | 292.9 MiB | 287.9 MiB |
| float64 | 291.3 MiB | 291.4 MiB |

Peak RSS covers the entire worker process, including validation and repeated
transforms. It is not an isolated measurement of the distance workspace.

Float32 peak RSS decreased by approximately 5 MiB; float64 peak RSS was
essentially unchanged. The main measured improvement was execution time.

## Output agreement

All 108 workers completed successfully and passed the benchmark checks.

- Candidate versus 0.3.19: maximum absolute imputed-value difference was zero
  in every measured case.
- Repeated runs produced identical outputs for matching cases.
- Complete-policy outputs matched KNNImputer exactly.
- Available-policy maximum absolute differences from KNNImputer were
  `4.76837158203125e-7` for float32 and `8.881784197001252e-16` for float64,
  unchanged from the baseline.

These results describe this synthetic workload on this runner. Baseline and
candidate timings were measured within the same workflow run.

## Reproduction and raw results

Run `benchmark-released.yml` in `candidate` mode using the measured source
commit. This compares the candidate wheel with PyPI 0.3.19 and KNNImputer.

- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/34910709635)
- [Raw results](../../benchmarks/results/available-distance-buffers-94adbf66.json)