# Released-version benchmark: FaissImputer 0.3.10

Historical measurements comparing **PyPI FaissImputer 0.3.10** with
**KNNImputer from scikit-learn 1.9.0** on AMD and Intel runners.

For the newer release comparison, see the
[0.3.19 benchmark report](released_versions_0.3.19.md).

## Measurement setup

Both runs used:

- Synthetic float32 data.
- One thread, 300 query rows, and 20 features.
- Five neighbors and uniform-weight mean aggregation.
- Four missing features per query.
- Fully observed training data for complete-policy cases.
- Approximately 10% MCAR training missingness for available-policy cases.
- Identical inputs within each policy comparison.
- Three seeds and three fresh runs per seed.

Times measure the **first transform after fitting**, excluding fit time.
Reported times are medians across those runs.

Speedup is KNNImputer time divided by FaissImputer time, calculated from
unrounded medians. Values above 1× mean FaissImputer is faster; values
below 1× mean it is slower.

## AMD runner

The AMD run used **20,000 training rows**.

A shared pattern means every query has the same missing feature positions.
Random patterns allow those positions to differ between queries.

| Donor policy | Query missingness | KNNImputer | FaissImputer 0.3.10 | Speedup |
| --- | --- | ---: | ---: | ---: |
| complete | One shared pattern | 422.4 ms | 22.0 ms | 19.24× |
| complete | Random patterns | 409.8 ms | 107.2 ms | 3.82× |
| available | One shared pattern | 391.4 ms | 166.2 ms | 2.35× |
| available | Random patterns | 385.6 ms | 167.5 ms | 2.30× |

FaissImputer was faster in all four measured cases. The largest advantage
occurred with complete donors and one shared query missingness pattern.

[Raw results and environment](../../benchmarks/results/scaling-threads-34297607304.json)
· [Workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/34297607304)

## Intel runner

The Intel run varied training size from **1,000 to 100,000 rows**.
The table shows speedup over KNNImputer.

Here, `fixed` means one shared query missingness pattern.

| Training rows | complete / fixed | complete / random | available / fixed | available / random |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 6.75× | **0.71× — slower** | 1.24× | 1.19× |
| 5,000 | 9.64× | 1.46× | 1.34× | 1.34× |
| 20,000 | 11.00× | 1.63× | 1.32× | 1.33× |
| 100,000 | 15.56× | 2.30× | 1.73× | 1.64× |

The complete-policy case with random query patterns and 1,000 training
rows was slower than KNNImputer. The other listed cases were faster,
with advantages depending on training size and missingness pattern.

[Raw results and environment](../../benchmarks/results/scaling-threads-34310124369.json)
· [Workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/34310124369)

## Output agreement

In both runs:

- Complete-policy outputs matched KNNImputer exactly.
- Available-policy outputs differed from KNNImputer by at most `4.77e-7`
  on the tested inputs.
- Repeated transforms produced unchanged outputs.

These comparisons describe agreement between imputers, rather than
imputation error against hidden ground truth.

## Memory

Memory advantages varied with training size.

In the Intel scaling run, available mode's median whole-worker peak RSS
was **6–21% higher** than KNNImputer's at 5,000 and 20,000 training rows,
and **15–27% lower** at 100,000 rows.

Whole-worker peak RSS covers the worker process and is not an exact
measurement of fitted-model storage.

## Interpretation

These measurements show that training size, donor policy, and query
missingness patterns materially affect the performance advantage.

The AMD and Intel runs used different CPUs. Differences between their
results do not measure a change between software versions.

Likewise, comparing these historical values with a newer release's
results from another environment does not establish a performance
regression or improvement. A version comparison requires matched
hardware, dependencies, inputs, and measurement definitions.