# Available-transform profiles: Wine quality and Abalone

Source: [`a259000`](https://github.com/ScionKim/FaissImputer/commit/a25900033ab590ae8c8982e0d996f03cbc26591c); candidate `0.3.21+bench.a25900033ab5`.

Recorded CPU model: **AMD EPYC 7763 64-Core Processor**. Matching CPU labels do not identify the same physical runner.

Recorded environment: Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, Faiss 1.15.1; native threads: 1.

## Configuration and aggregation

Both datasets use 3,000 training rows, 1,000 held-out queries, k=5, 10% target MCAR missingness, uniform weights, l2 distance, mean aggregation, a Flat index, and available donors.

The API is `fit_then_transform`; only the first transform is timed, with fit excluded. Each dataset/dtype cell has **3 records: seeds 101, 202, 303, with one traced execution per seed**. The datasets and dtypes are kept separate. Ranges reflect different seeded cases, not repeated timing trials.

Every table summary is median [min–max] across those three records. Time values are converted from seconds to milliseconds only for display. For a stage's percentage, divide its seconds by `instrumented_transform_seconds` within each record, then summarize the three fractions. Percentages are not ratios of displayed median times.

The untraced reference transform runs first and warms the process; a separately fitted model is then traced. Wrappers add overhead. These diagnostic times do not establish benchmark speedups, and reference/traced timing ratios are not reported.

### Timing field definitions

All timer fields below are under `records[].profile.timings`. Except the root, fields use `inclusive_seconds` unless marked `self_seconds`.

| Report field | Raw field or calculation |
|---|---|
| Instrumented transform | `records[].instrumented_transform_seconds`, equal to `instrumented_transform.inclusive_seconds` |
| Full-donor refinement | `full_direct_distances.suspect.inclusive_seconds + full_direct_distances.tie.inclusive_seconds` |
| Initial distance matrix | `prepared_matrix_distances.inclusive_seconds` |
| Selected-candidate refinement | `direct_distance_kernel.selected.inclusive_seconds` |
| Faiss candidate selection | `faiss_kmin.inclusive_seconds` |
| Aggregation calls | `aggregate.inclusive_seconds` |
| Matrix preparation: self | `prepare_search_matrix.self_seconds` |
| Search: self | `search.self_seconds` |
| Available transform: self | `available_transform.self_seconds` |

Inclusive buckets overlap. Full-donor refinement already contains its direct-distance kernel; that child is not added again. Self time subtracts immediate traced children and includes tracing overhead and surrounding production work. The selected stages below are not a complete additive breakdown. Absent optional refinement buckets count as zero only when matching recorded row events are zero.

Dataset feature counts: Wine quality (white) 11; Abalone 7.

### Stage times (ms)

| Stage | Wine f32 | Wine f64 | Abalone f32 | Abalone f64 |
|---|---:|---:|---:|---:|
| Instrumented transform | 170.00 [168.56–170.86] | 205.00 [203.49–207.13] | 30.09 [29.88–30.93] | 72.31 [71.17–72.44] |
| Full-donor refinement (suspect + tie) | 114.03 [113.63–114.31] | 113.93 [112.52–114.50] | 3.36 [3.00–3.53] | 4.33 [3.85–4.78] |
| Initial distance matrix | 22.58 [22.53–23.33] | 22.31 [21.98–22.31] | 15.68 [15.59–15.76] | 16.53 [16.29–17.06] |
| Selected-candidate refinement | 0.00 [0.00–0.00] | 23.77 [22.84–24.38] | 0.00 [0.00–0.00] | 25.99 [25.68–26.09] |
| Faiss candidate selection | 3.77 [3.76–3.84] | 3.81 [3.76–3.94] | 2.83 [2.77–2.84] | 2.83 [2.74–2.87] |
| Aggregation calls | 1.51 [1.46–1.57] | 1.38 [1.36–1.41] | 0.91 [0.87–0.94] | 0.85 [0.82–0.86] |
| Matrix preparation: self | 8.94 [8.30–9.60] | 9.80 [9.28–9.88] | 3.82 [3.78–3.83] | 4.12 [3.81–4.98] |
| Search: self | 7.90 [7.51–8.50] | 20.15 [19.59–20.31] | 1.03 [0.96–1.23] | 14.65 [13.49–14.66] |
| Available transform: self | 3.77 [3.74–3.90] | 3.95 [3.91–4.05] | 2.23 [2.12–2.28] | 2.24 [2.24–2.37] |

### Stage shares of instrumented transform (%)

| Stage | Wine f32 | Wine f64 | Abalone f32 | Abalone f64 |
|---|---:|---:|---:|---:|
| Instrumented transform | 100.00 [100.00–100.00] | 100.00 [100.00–100.00] | 100.00 [100.00–100.00] | 100.00 [100.00–100.00] |
| Full-donor refinement (suspect + tie) | 66.90 [66.84–67.65] | 55.29 [55.00–55.85] | 11.16 [10.04–11.41] | 6.08 [5.31–6.62] |
| Initial distance matrix | 13.36 [13.22–13.72] | 10.80 [10.77–10.88] | 51.82 [50.96–52.47] | 22.88 [22.86–23.56] |
| Selected-candidate refinement | 0.00 [0.00–0.00] | 11.68 [11.14–11.77] | 0.00 [0.00–0.00] | 36.02 [35.95–36.08] |
| Faiss candidate selection | 2.21 [2.20–2.28] | 1.86 [1.85–1.90] | 9.28 [9.18–9.40] | 3.90 [3.85–3.96] |
| Aggregation calls | 0.89 [0.86–0.93] | 0.68 [0.67–0.68] | 3.04 [2.91–3.05] | 1.19 [1.13–1.20] |
| Matrix preparation: self | 5.26 [4.92–5.62] | 4.77 [4.56–4.78] | 12.66 [12.40–12.71] | 5.69 [5.27–6.99] |
| Search: self | 4.69 [4.42–4.97] | 9.81 [9.56–9.90] | 3.46 [3.18–3.99] | 20.23 [18.96–20.26] |
| Available transform: self | 2.24 [2.19–2.29] | 1.94 [1.90–1.96] | 7.36 [7.04–7.47] | 3.14 [3.09–3.27] |

## Search and refinement events

Counts are row events across search calls, not generally unique query rows. Full = suspect + tie. Expansion calls are cache-hit search calls. Values come from `profile.counters`, checked against `profile.search_events`; `requested_k` lists every call in order.

| Dataset | dtype | Seed | Initial query rows | Suspect | Tie | Full | Selected rows | Selected pairs | Searches | Expansion | Retain | requested_k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wine quality (white) | float32 | 101 | 706 | 167 | 69 | 236 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Wine quality (white) | float32 | 202 | 699 | 156 | 77 | 233 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Wine quality (white) | float32 | 303 | 680 | 165 | 70 | 235 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Wine quality (white) | float64 | 101 | 706 | 167 | 69 | 236 | 470 | 7520 | 3 | 0 | 0 | 16, 16, 16 |
| Wine quality (white) | float64 | 202 | 699 | 156 | 77 | 233 | 466 | 7456 | 3 | 0 | 0 | 16, 16, 16 |
| Wine quality (white) | float64 | 303 | 680 | 165 | 70 | 235 | 445 | 7120 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float32 | 101 | 536 | 5 | 2 | 7 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float32 | 202 | 522 | 8 | 0 | 8 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float32 | 303 | 534 | 6 | 2 | 8 | 0 | 0 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float64 | 101 | 536 | 5 | 4 | 9 | 527 | 8432 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float64 | 202 | 522 | 8 | 2 | 10 | 512 | 8192 | 3 | 0 | 0 | 16, 16, 16 |
| Abalone | float64 | 303 | 534 | 6 | 5 | 11 | 523 | 8368 | 3 | 0 | 0 | 16, 16, 16 |

## Instrumentation fidelity and reconstruction error

All 12 archived records have status `ok`, `checks_passed=true`, matching stored reference/traced output hashes, unchanged input hashes, and passing shape/dtype, observed-value, finite-output, cache-cleanup, and wrapper-restoration checks. These checks establish instrumentation fidelity for the recorded fixtures, not general algorithmic correctness or agreement with KNNImputer.

RMSE and MAE are `records[].quality.rmse` and `.mae`: reconstruction error against masked held-out ground truth, standardized using observed training values. Each metric below is median [min–max] across seeds. Scored cells vary by seed; errors are not pooled across cells.

| Dataset | dtype | RMSE | MAE | Scored cells |
|---|---|---:|---:|---:|
| Wine quality (white) | float32 | 0.801335988 [0.694396804–0.851476121] | 0.504907762 [0.499190728–0.554542152] | 1103 [1093–1105] |
| Wine quality (white) | float64 | 0.801335990 [0.694396804–0.851476120] | 0.504907764 [0.499190728–0.554542152] | 1103 [1093–1105] |
| Abalone | float32 | 0.274643705 [0.258502921–0.310640885] | 0.177399664 [0.176123216–0.181759846] | 708 [693–711] |
| Abalone | float64 | 0.274563045 [0.258502922–0.310640886] | 0.177163701 [0.176123216–0.181759846] | 708 [693–711] |

## Interpretation and next investigation

The largest listed main-stage bucket in each cell is determined from its median time:

- Wine quality (white) float32: Full-donor refinement (suspect + tie), 114.03 ms median; 66.90% median per-record share.
- Wine quality (white) float64: Full-donor refinement (suspect + tie), 113.93 ms median; 55.29% median per-record share.
- Abalone float32: Initial distance matrix, 15.68 ms median; 51.82% median per-record share.
- Abalone float64: Selected-candidate refinement, 25.99 ms median; 36.02% median per-record share.

Recorded expansion search calls: 0. Requested candidate counts across all calls: 16.

Read the costs separately for each dataset/dtype. Bounded batching of selected-candidate float64 distance recomputation is a shared optimization candidate. Full-donor refinement needs separate attention wherever its recorded share is large. Preserve numerical guards, overflow repair, distance semantics, and deterministic neighbor selection. The profiles do not predict a speedup: evaluation requires correctness checks and uninstrumented measurements.

## Evidence and reproduction

[Analysis script](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/analyze_available_transform_profile.py) · [Exact summary](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-transform-profile-a259000-summary.json) · [Analysis workflow](https://github.com/ScionKim/FaissImputer/blob/main/.github/workflows/analyze-available-transform-profile.yml)

Run **Analyze available-transform profiles** in GitHub Actions. It reads the two preserved ZIPs using the Python standard library and writes this report plus the JSON summary. The summary retains unrounded binary64 calculation values, per-seed derived measurements, original timer buckets/counters/search events, checks, case fingerprints, and provenance. No imputer or dataset preparation is executed and the raw ZIPs are never modified.

### Wine quality (white)

[Preserved ZIP](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-transform-profile-a259000/wine-quality-white.zip) · [Profiling run](https://github.com/ScionKim/FaissImputer/actions/runs/36668061565) (attempt 1)

- ZIP SHA-256: `3681d8a34cdabeabbc8b531c68ca73cb92a755eae739a101eef4844bc7460526`
- `available_transform_profile.json` SHA-256: `32c796d46ce3e234b17b786e4463ee981a3b6e1290fd88e97d62d9afdc14c2db`
- Candidate wheel SHA-256: `c88c85ee05047ba0fd6a8bb73c6790d575cda6197d4134e96fe9651d13c4e792`

### Abalone

[Preserved ZIP](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-transform-profile-a259000/abalone.zip) · [Profiling run](https://github.com/ScionKim/FaissImputer/actions/runs/36669353522) (attempt 1)

- ZIP SHA-256: `a008d167e6e30d87de85e5a8576308c4a31819dc9046a8153e5f17e81976b342`
- `available_transform_profile.json` SHA-256: `c882f206997b8d48a6175e730a08161363892860b0597e0f52c67d1535faa979`
- Candidate wheel SHA-256: `1b6e294cfa800ecbda359012e59223e59332d5c474b4f3a5a54f3fc136b6ffe4`

Source hashes shared by the archived records:

- `faiss_imputer_source_sha256`: `bf82b170a6af3161b49cf623f714b5b80ee849ff91a599f5fc05268f9acbd17a`
- `matrix_source_sha256`: `3454e6db843d74cde9c26628fee11cdcb658e13cfacec5319d61815bab4545a6`
- `case_helper_sha256`: `a51dc31b9a5927f75a63028bf9443797741da7851c733eb722242186335ea6ac`
- `profile_script_sha256`: `54da53ab5c424287df1f440da3ecf8845dd88e03e80b9e8e3193f2c80bfd8ae7`
- `recorded_dependencies_sha256`: `2847dbadceaf133bbdb9a975f473986cb7b3918e2e928d0abb4e110654824b5d`
