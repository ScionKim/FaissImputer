# Roadmap

Updated after [0.3.4](https://github.com/ScionKim/FaissImputer/releases/tag/v0.3.4), based on a source and regression review of commit [`bc592934`](https://github.com/ScionKim/FaissImputer/tree/bc592934e83d5435672a1be31801613ef7b6c06d).

The next patch focuses on consistent imputation results and interoperability. Performance work follows those fixes, with measurements tied to the version actually tested. These priorities are not release-date commitments or promises of universal speedups or numerical identity with `KNNImputer`.

## Completed through 0.3.4

- Correct neighbor-to-donor mapping, observed-feature search, and fitted fallback statistics.
- Opt-in `donor_policy="available"` support for partially observed training rows, with per-target donor eligibility and fallbacks.
- Reduced complete-donor column-projection overhead and batched available-donor distance calculations.
- Prepared donor-side arrays reused across available-donor query batches, with the retained-memory tradeoff documented.
- Input-preservation checks, failed-refit state cleanup, and available-donor numerical safeguards. Remaining numerical issues are listed below.
- Pipeline and ColumnTransformer integration, feature names, and optional pandas output.
- Minimum/latest dependency CI, package metadata checks, installed-wheel smoke tests, and automated PyPI publishing.
- Reproducible synthetic benchmarks, a real-data MCAR/MAR pilot, and historical batching/thread experiments. Coverage and current-release measurements still need expansion.

## Next patch: correctness and interoperability

### 1. Make available-donor results independent of query batching

**Problem:** A numerical safeguard can switch the entire query batch from float32 to float64 neighbor selection. Adding an unrelated query can therefore change the imputation of an existing query.

Reproduced on 0.3.4 with NumPy 2.5.2, scikit-learn 1.9.0, and Faiss 1.15.0:

```python
import numpy as np
from faiss_imputer import FaissImputer

imputer = FaissImputer(n_neighbors=1, donor_policy="available").fit(
    [[1, 0.0001, 10], [1, 0, 20]]
)
imputer.transform([[0, 0, np.nan]])                  # Missing value: 10
imputer.transform([[0, 0, np.nan], [1, 0, np.nan]])  # First missing value: 20
```

In this example, the distances are distinct before float32 rounding; the closer donor supplies 20.

Work and completion criteria:

- Define a consistent precision and tie-handling rule for each query, independent of other queries in its batch.
- Add regression checks for single versus grouped calls, query reordering, internal batch boundaries, and repeated transforms.
- Check small cases against an independent direct-distance reference, including near ties and rows that trigger numerical safeguards.
- Preserve per-target donor eligibility, originally observed values, mean/median aggregation, and fitted fallbacks.
- Resolve the example above consistently to the closer donor. Document any intentional changes from 0.3.4; preserving its erroneous donor choices is not a compatibility requirement.
- Measure the cost of the fix. Do not make equality to `KNNImputer` on every input a release condition: its precision and tie choices can differ.

### 2. Accept NumPy integer neighbor counts

**Problem:** The Python-`int`-only validation rejects NumPy integers. A GridSearchCV parameter grid such as `{"faissimputer__n_neighbors": np.arange(1, 3)}` fails for both donor policies.

Done when:

- Positive integral scalar values, including NumPy integer types, are accepted; invalid values are rejected and boolean handling is explicit.
- Values are converted to native integers where required by Faiss, while preserving scikit-learn estimator cloning behavior.
- A real Pipeline/GridSearchCV regression using a NumPy-generated grid succeeds for both donor policies.
- Existing donor-count constraints, failed-fit cleanup, and input-preservation checks continue to pass.

## Follow-up: targeted performance improvements

### Avoid unused complete-donor index storage

The complete policy currently builds a full-dimensional index during `fit()`, but `transform()` builds projected indexes for its observed-feature patterns instead of searching that stored index.

- Remove or avoid the unused allocation while retaining appropriate fit-time factory validation and fitted-state checks.
- Verify supported metrics/factories, failed refits, feature-name handling, and output behavior.
- Measure fit time, retained fitted memory, and transform time against the corrected baseline.

### Avoid unnecessary candidate expansion for sparse targets

The available policy widens candidate selection for the whole batch until every missing target has enough donors or all donors have been considered. A target with fewer than `n_neighbors` observed donors can force exhaustive selection, including repeated work for already-resolved queries.

- Investigate per-target donor counts and candidate pools, and expansion restricted to unresolved queries.
- Preserve exact eligible-neighbor selection and the documented behavior when fewer than `n_neighbors` donors exist or none share observed features.
- Add a benchmark with highly missing target columns and mixed easy/difficult queries; uniform low-rate MCAR alone does not cover this case.
- Compare outputs with an independent reference and the corrected baseline, including ties and insufficient-donor cases.

## Follow-up: current-release benchmarks and documentation

### Measure the workloads users actually run

The [historical million-row pilot](docs/benchmarks/available-batching-90c8cfb8.md) used one million training rows but only 300 query rows and one timing run. It predates 0.3.4 donor preparation. It does not establish the cost of imputing one million query rows or running `fit_transform()` on one million rows.

- Benchmark the released package being documented, against KNNImputer and an appropriate prior FaissImputer baseline on matching hardware and inputs.
- Vary training rows, query rows, and feature count independently; extend missingness, neighbor-count, and pattern coverage where informative.
- Separate fit, first transform, repeated transforms, and same-data `fit_transform()` at feasible sizes. Exact available-donor pairwise work grows with both donor and query counts.
- Report quality against hidden ground truth separately from output agreement with another imputer.
- Measure retained fitted memory and phase-specific peak memory, distinguishing these from whole-worker peak RSS and internal batch-sizing budgets.
- Use multiple seeds and repetitions. Record versions, hardware, thread limits, timing dispersion, scope limitations, reproduction commands, and linked raw results.
- Extend the current single-dataset real-data pilot with datasets representing useful application workloads. Retain SimpleImputer as a low-cost baseline where appropriate.
- Keep long performance runs manual or scheduled; use small correctness checks in CI rather than noisy per-commit speed gates.

Publish each new report with its measured revision and environment. Keep historical reports labeled as historical rather than relabeling their measurements as current-release results.

### Clarify compatibility and help users choose

- Add a concise KNNImputer comparison covering donor defaults, neighbor defaults, float32 conversion, all-missing columns, and unsupported options such as weights and missing indicators.
- Explain that scikit-learn integration does not imply identical constructor options or identical donor choices. The [real-data pilot](docs/benchmarks/real-data-a3bd1ce3.md) already documents substantive per-cell differences despite similar average errors.
- Keep completed items and remaining evidence gaps in this roadmap current as changes ship.

## Later work, driven by evidence and user needs

- **Extreme numerical scales:** protect mean/median aggregation from intermediate float32 overflow and review complete-donor distance underflow/overflow. Use explicit finite-input reproductions and an independent reference. Reproduced errors on ordinary-scale inputs belong in the next correctness patch.
- **Factory support:** define which index factories remain valid when queries have different observed-feature counts. For example, a factory can accept the fitted dimension and reject a projected dimension. Provide clear validation or a documented fallback.
- **Broader interoperability checks:** add standard scikit-learn estimator checks and address remaining error-message requirements; expand installation/basic-execution coverage to Windows and macOS.
- **Memory controls:** use current measurements to evaluate a public batch/working-memory setting and donor-block processing. An internal batch budget must not be presented as a total RAM limit.
- **Additional API features:** consider distance weighting, missing indicators, and empty-feature policies when concrete use cases justify their behavior and maintenance cost.
- **Approximate search and GPU work:** pursue a specific workload and an acceptable accuracy/performance tradeoff first. Neither changes the priority of consistent results in the existing exact modes.
