# Available-donor candidate expansion: v0.3.7 versus v0.3.8

This benchmark measures the candidate-expansion changes released in v0.3.8.
Ordinary workloads showed small timing differences, while targeted sparse
and exhausted-neighbor workloads showed substantially lower transform time
and process peak memory.

## Sources and raw results

- Baseline: v0.3.7, commit `d7d9cd5f98c3154084de8e90cbb1118f309508cd`.
- Candidate: v0.3.8, commit `099c94107839a4bf6d3811f39541ed609530ed20`.
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/34087296146).
- [Archived raw results](paired-available-expansion-34087296146-1.zip).
- [Measured workflow](https://github.com/ScionKim/FaissImputer/blob/723a892d4e1de1f6616781e9ce7cd50aad154f38/.github/workflows/paired-available-expansion.yml).

The archive contains individual measurements and outputs, comparison tables,
source hashes, dependency versions, and CPU information.

## Measurement setup

- Recorded on 2026-09-07 UTC on one GitHub-hosted Ubuntu runner.
- Intel Xeon Platinum 8573C; four logical CPUs available.
- Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1,
  scikit-learn 1.9.0, Faiss 1.15.0.
- Available-donor policy, L2 distance, mean aggregation, five neighbors.
- 16,384 or 65,536 training rows; 128 queries; 12 features; seed 101.
- One or two native threads, with FAISS/OpenMP and BLAS limits set together.
- Each configuration ran v0.3.7, v0.3.8, v0.3.8, v0.3.7 sequentially.
- Every observation used a fresh process and a small warmup.
- Reported values are arithmetic means of two observations per release.

Transform timing excludes fit, data generation, warmup, and validation.
Fit and combined fit-plus-transform times are recorded separately.

Peak RSS is the process high-water mark read immediately after transform,
before validation and output serialization. It includes imports, data
generation, warmup, and fit. It is not transform-only memory or retained
fitted-model memory.

## Workloads

- **Ordinary:** Gaussian data with random donor and query missingness.
  Masks are initially drawn at 10% and 20%, respectively; feature 0 is
  always observed, and query feature 1 is always missing.
- **Mixed sparse:** 112 easy queries and 16 hard queries are interleaved.
  The hard queries require a target observed in only two donors. Those
  donors are placed around the first quarter and third of the distance
  ranking for an anchor query, beyond the initial candidate set.
- **Finite exhaustion:** Queries have only one or two finite-distance
  donors. Cases include a partially fillable target and a target requiring
  fitted-statistic fallback.

The latter two workloads specifically exercise the optimization. They are
not estimates of average application performance.

## Transform time and memory

Changes are calculated as `100 * (v0.3.8 mean / v0.3.7 mean - 1)`.
Negative values mean less time or memory. Percentages use unrounded values.

| Workload | Donors | Threads | 0.3.7 time (s) | 0.3.8 time (s) | Time change | 0.3.7 RSS (MiB) | 0.3.8 RSS (MiB) | RSS change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 16,384 | 1 | 0.0598 | 0.0549 | -8.21% | 187.8 | 187.5 | -0.14% |
| Ordinary | 16,384 | 2 | 0.0511 | 0.0521 | +2.10% | 189.7 | 189.7 | -0.03% |
| Ordinary | 65,536 | 1 | 0.2207 | 0.2233 | +1.19% | 358.5 | 358.4 | -0.04% |
| Ordinary | 65,536 | 2 | 0.2058 | 0.2043 | -0.68% | 360.7 | 360.8 | +0.04% |
| Mixed sparse | 16,384 | 1 | 1.2939 | 0.1219 | -90.58% | 275.5 | 187.7 | -31.86% |
| Mixed sparse | 16,384 | 2 | 0.9057 | 0.0977 | -89.21% | 277.5 | 189.9 | -31.57% |
| Mixed sparse | 65,536 | 1 | 6.1956 | 0.5535 | -91.07% | 672.5 | 358.7 | -46.66% |
| Mixed sparse | 65,536 | 2 | 4.3500 | 0.3943 | -90.93% | 674.5 | 360.9 | -46.50% |
| Finite exhaustion | 16,384 | 1 | 0.5565 | 0.0590 | -89.40% | 259.8 | 187.7 | -27.74% |
| Finite exhaustion | 16,384 | 2 | 0.5158 | 0.0578 | -88.79% | 261.8 | 189.8 | -27.47% |
| Finite exhaustion | 65,536 | 1 | 2.6296 | 0.2366 | -91.00% | 574.7 | 358.6 | -37.60% |
| Finite exhaustion | 65,536 | 2 | 2.2626 | 0.2200 | -90.28% | 577.0 | 360.6 | -37.51% |

The ordinary 16,384-row, one-thread baseline had a 21.29% timing spread
between its two observations, where spread is `(maximum - minimum) / mean`.
Its apparent 8.21% improvement should not be treated as an established
speedup. The other ordinary configurations ranged from -0.68% to +2.10%.

## Fit-time tradeoff

Mean fit time increased by 4.60-8.85% across the twelve configurations,
equivalent to approximately 0.27-1.69 ms. This is consistent with the added
fit-time calculation of per-feature donor counts.

Including fit, the targeted workloads still reduced total time by
87.74-90.82%. Ordinary fit-plus-transform changes ranged from -7.05%
to +2.63%; the largest decrease includes the variable baseline noted above.

## Validation and limits

All 48 workers succeeded and all twelve comparisons passed validation.
Source imports, input fingerprints, dependencies, and requested thread
settings matched within each comparison.

Output values matched exactly across all four observations of every
configuration: maximum absolute difference was 0.0. Shape, finiteness,
observed-value preservation, input preservation, and cache cleanup checks
also passed.

This is a pilot using one synthetic seed and two observations per release.
ABBA ordering reduces order bias but does not remove runner noise or
establish statistical significance. These results do not establish a
universal speedup, general numerical equivalence, or imputation accuracy
against hidden ground truth.

## Reproduction

Run the `Paired available expansion benchmark` workflow manually through
GitHub Actions. It checks out the two release tags and records their
resolved commits.

The linked workflow preserves the measured harness. Consult the archived
`pip-freeze.txt` and `lscpu.txt` when matching the original environment:
transitive dependencies and hosted-runner hardware may change on reruns.
