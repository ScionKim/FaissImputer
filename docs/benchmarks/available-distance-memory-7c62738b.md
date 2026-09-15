# Available-donor search memory benchmark

This benchmark compares the previous distance-buffer optimization with
an unpublished candidate that prepares search matrices in groups of query
rows and releases temporary arrays before precise distance refinement.

Available-policy peak process RSS decreased by approximately 13%, while
first-transform time decreased by a further 13–15%. Measured imputation
outputs were unchanged.

## Compared versions

- Baseline: `0.3.19+bench.94adbf660f59`, including the earlier speed improvement.
- Candidate: `0.3.19+bench.7c62738b4235`.
- Baseline commit: `94adbf660f59f5dd7a03691af8a6f71d99a7d467`.
- Candidate commit: `7c62738b4235d08ca3a3ba0664d4c1eefeb5db0f`.

Both versions were built from source commits. The baseline is not the
published PyPI 0.3.19 package.

## Conditions

- CPU: AMD EPYC 7763; one native thread.
- Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, Faiss 1.15.0.
- Training data: 20,000 rows and 20 features.
- Queries: 300 rows, each missing four randomly selected features.
- Complete policy: fully observed training data.
- Available policy: training data with 10% MCAR missingness.
- Both float32 and float64; float64 generated without a float32 round trip.
- Five neighbors, mean aggregation, uniform weights, Flat L2 search.
- Three seeds: 101, 202, and 303; three fresh workers per method and seed.
- Sequential workers with method order rotated across repetitions.
- Each worker measured fit, the first transform, and two additional transforms.

Results below are medians of nine workers per method and condition.
Timings exclude process startup, data generation, warmup, and validation.

## Available-policy results

| dtype | Baseline peak RSS | Candidate peak RSS | Baseline transform | Candidate transform |
| --- | ---: | ---: | ---: | ---: |
| float32 | 285.61 MiB | 248.32 MiB | 115.56 ms | 98.68 ms |
| float64 | 289.03 MiB | 251.46 MiB | 135.18 ms | 117.57 ms |

- Peak RSS decreased by 13.1% for float32 and 13.0% for float64,
  approximately 37 MiB in both cases.
- First-transform time decreased by 14.6% for float32 and 13.0% for float64.
- Fit plus first-transform time decreased from 127.76 to 110.66 ms for
  float32 and from 146.94 to 130.10 ms for float64.
- Repeated-transform time decreased by 15.0% for float32 and 12.7% for float64.

Peak RSS covers the entire worker process, including validation and repeated
transforms. It is not an isolated measurement of a single array or fitted model.

Complete-policy total time differed by less than 0.2%, with essentially
unchanged peak RSS.

## Output agreement

All 108 workers completed successfully and passed the benchmark checks.

- Candidate versus baseline: maximum absolute imputed-value difference
  was zero in every measured case.
- Repeated runs produced identical outputs for matching cases.
- Complete-policy outputs matched KNNImputer exactly.
- Available-policy maximum absolute differences from KNNImputer were
  `4.76837158203125e-7` for float32 and `8.881784197001252e-16` for float64.

These results describe this workload and runner. The baseline and candidate
were measured within the same run. The earlier speed benchmark used a
different CPU, so its percentages should not be combined with these results
to claim a cumulative improvement over PyPI 0.3.19.

## Reproduction and supporting results

Run `benchmark-released.yml` in `candidate` mode on the candidate commit,
setting `baseline_commit` to
`94adbf660f59f5dd7a03691af8a6f71d99a7d467`.

- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/34917029070)
- [Raw results](../../benchmarks/results/available-distance-memory-7c62738b.json)
- [Earlier distance-buffer speed benchmark](available-distance-buffers-94adbf66.md)