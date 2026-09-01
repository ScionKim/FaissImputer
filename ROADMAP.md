# Roadmap

This roadmap records the next improvements after the 0.2.0 correctness release. Priorities are based on the reproducible [0.2.0 benchmark](docs/benchmarks/v0.2.0.md), not on a claim that one imputer is best for every dataset.

## 0.2.0 baseline

- Correct neighbor-to-donor mapping.
- Search using only each query row's observed columns.
- Preserve fitted donors and fitted fallback statistics during `transform()`.
- Support current scikit-learn validation APIs.
- Add regression tests, CI, modern packaging, and automated PyPI publishing.
- Publish a controlled benchmark against `SimpleImputer` and `KNNImputer`.

## 0.3.0 priorities

### P0: Use partially observed training rows as donors

**Why:** With 10% independently missing training cells, only about 598 of 5,000 rows were fully observed. In the 0.2.0 benchmark, FaissImputer's RMSE was 0.3495 versus 0.2440 for KNNImputer because FaissImputer discarded every incomplete training row.

Define the behavior before implementation:

- For each target column, consider only donors that contain a value for that column.
- Calculate distance from features observed in both the query and donor.
- Exclude donors with no commonly observed distance feature.
- If fewer than `n_neighbors` donors are available, use all valid donors.
- If no donor is available, use the statistic learned during `fit()`.
- Document whether exact KNNImputer compatibility is a goal or whether FaissImputer intentionally uses a different rule.

Implementation work:

- Preserve training values and their observation mask during `fit()`.
- Build or select valid donor pools per target column.
- Add distance calculation and scaling for shared observed features.
- Preserve the fast complete-donor path.
- Add small hand-calculated examples and regression tests for every fallback.

Done when:

- Fitting succeeds even when there are no fully observed rows, provided valid donors exist for the requested columns.
- A donor missing the target value is never used for that target.
- Donors with no shared observed distance feature are excluded.
- `transform()` leaves no unexpected `NaN` values and does not modify its input.
- Complete-training results remain compatible with 0.2.0 within `atol=1e-6`.
- On the fixed 0.2.0 incomplete-training benchmark, RMSE improves from 0.3495, with `<=0.27` used as a regression target rather than a universal accuracy claim.

### P1: Reduce the cost of many unique missingness patterns

**Why:** With 20,000 training rows and 300 test rows, one shared missingness pattern was 19.62 times faster than KNNImputer and eight patterns were 10.78 times faster. With 294 patterns, the 0.99 times ratio was effectively tied because index preparation was repeated for almost every pattern.

Investigation and implementation work:

- Measure index construction and search time separately.
- Prototype a bounded pattern-index cache and record its memory cost.
- Investigate a single-index exact-search representation for mixed masks.
- Preserve exact `Flat` L2 results before optimizing approximate indexes.
- Define safe fallback behavior for other metrics and index factories.
- Clear cached state correctly after a new `fit()` call.

Done when:

- Imputed values remain compatible with 0.2.0 within `atol=1e-6` on complete-donor tests.
- The 20,000-row, many-pattern benchmark reaches at least a twofold transform improvement over the 0.2.0 FaissImputer baseline, using `<=170 ms` as the current machine-specific regression target.
- The one-pattern path does not regress by more than 20%.
- Memory use cannot grow without a documented bound as new patterns appear.

### P2: Broaden and automate benchmark coverage

- Add public real-world numeric datasets alongside synthetic data.
- Test MCAR and selected MAR scenarios separately.
- Vary row count, feature count, missing rate, neighbor count, and repeated-pattern count.
- Keep accuracy and runtime measurements separate.
- Record package versions, seeds, thread count, medians, and interquartile ranges.
- Add a short correctness smoke benchmark to CI; keep longer performance runs manual or scheduled to avoid noisy per-commit timing gates.

Done when the benchmark can be recreated from a clean environment with one documented command and every published result contains its environment and scope limitations.

## Later work

Approximate indexes, GPU-specific optimization, and very large datasets should be prioritized only after real benchmark evidence identifies a useful target and an acceptable accuracy/performance tradeoff.

