# Released-package comparison: 0.3.21 and 0.3.20

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed PyPI releases with KNNImputer on one runner, using held-out queries. It is a separate-query `fit` followed by `transform` benchmark, not the same-data OFAT benchmark.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_versions_0.3.21.zip) (uploaded without modification).
- [Full-precision analysis](../../benchmarks/results/released_versions_0.3.21-summary.json) and [generator](../../benchmarks/analyze_released_versions.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/36645461314/attempts/1).
- Benchmark source commit: `693ce8ee0f9a44cdfe7aaab5d1af1233701eb0b1`; this identifies the benchmark runner, not an editable package installation.
- Archive SHA-256: `53ef359c45525b2fff05ea7a3cb9be685ba06c18ac742d68ba8882ad1377e229`.
- `version_comparison_q300.json` SHA-256: `192ab6345645649e9a3587a50f82bd7a3359ab1de1e158414911172639354124`.
- Runner: AMD EPYC 7763 64-Core Processor; 4 logical CPUs, 4 CPUs in affinity; requested and recorded native thread counts are one.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 20,000 training rows, 300 held-out queries, 20 features, k=5, uniform weights; FaissImputer is configured with built-in L2, mean aggregation, and `index_factory="Flat"`. Each query has four missing features (20%), giving 1,200 scored cells per seed.
- Three seeds (101, 202, 303), three fresh sequential workers per seed and variant. Method order rotates across repeats. There are 108 successful workers and no failed checks.
- `complete` uses fully observed training data. `available` uses training data with 10% target random missingness. Each policy's KNN baseline receives the same inputs as its Faiss variants. These are different input regimes, so cross-policy speed and quality differences do not isolate donor policy.

The two environments have identical archived dependency freezes except for faiss-imputer (0.3.20 versus 0.3.21). KNNImputer runs in the 0.3.21 environment. The analyzer checks package versions, installed site-packages locations, common environment metadata, full worker-grid coverage, input fingerprints, and all stored summary statistics against raw records.

## Aggregation methodology

Timing and RSS cells report median [min–max] over **9 workers = 3 seeds × 3 repeats**, separately for policy, dtype, and variant. `total_seconds` is fit plus the first transform. The two additional transform calls are excluded from that total: their per-worker median is summarized across the nine workers. First and subsequent transforms remain separate.

Speedups are calculated as numerator time / denominator time for each matched worker pair, followed by median [min–max] of the **9 paired ratios**. Matching requires the same run, policy, train/query sizes, feature and neighbor counts, metric/weights configuration, target training missingness, query pattern, dtype, thread count, additional-transform count, seed, repeat, and input fingerprints. The validated run-level configuration supplies fields not repeated in each record. The reported speedups are **not ratios of method-level median times**. Ratios above one favor the denominator method.

Quality cells report median [min–max] of **3 seed metrics**, after verifying that metrics and output hashes agree exactly across the three timing repeats. Repeat 1 supplies each seed metric. RMSE/MAE measure reconstruction error against hidden synthetic ground truth; method-to-method differences measure output agreement. This archive contains worker-computed metrics and hashes, not prediction/truth arrays: the analysis recomputes their summaries, not the underlying arraywise errors.

All calculations use unrounded parsed JSON values. The machine-readable summary preserves raw timing/quality samples, per-pair numerators, denominators, ratios, input/output hashes, and aggregate binary64 values at round-trip precision. Only the Markdown presentation is rounded. Ranges are observed minima/maxima, not confidence intervals; repeated timings of the same seed are not independent datasets.

## Timing

Seconds, median [min–max] across nine workers per row. Additional-transform values are medians of the two extra calls within each worker before aggregation.

### Complete training regime

| dtype | Method | Fit (s) | First transform (s) | Fit + first transform (s) | Additional transform (s) |
| --- | --- | --- | --- | --- | --- |
| float32 | KNNImputer | 0.001705 [0.001664–0.001833] | 0.288998 [0.281579–0.317708] | 0.290662 [0.283247–0.319541] | 0.277704 [0.269780–0.299908] |
| float32 | FaissImputer 0.3.20 | 0.004823 [0.004709–0.005162] | 0.149111 [0.146596–0.151269] | 0.153932 [0.151323–0.156430] | 0.143220 [0.140383–0.152296] |
| float32 | FaissImputer 0.3.21 | 0.004822 [0.004745–0.005153] | 0.150886 [0.149892–0.154848] | 0.155656 [0.154749–0.160000] | 0.143494 [0.142922–0.148822] |
| float64 | KNNImputer | 0.002428 [0.001793–0.002630] | 0.347377 [0.344443–0.355671] | 0.349857 [0.346871–0.358253] | 0.335308 [0.328156–0.337674] |
| float64 | FaissImputer 0.3.20 | 0.006906 [0.006173–0.007225] | 0.207161 [0.203320–0.210213] | 0.213458 [0.210163–0.216716] | 0.199260 [0.196702–0.209839] |
| float64 | FaissImputer 0.3.21 | 0.006479 [0.006205–0.006988] | 0.207676 [0.206224–0.210867] | 0.214484 [0.212428–0.217815] | 0.202067 [0.199626–0.204889] |

### Available training regime

| dtype | Method | Fit (s) | First transform (s) | Fit + first transform (s) | Additional transform (s) |
| --- | --- | --- | --- | --- | --- |
| float32 | KNNImputer | 0.001740 [0.001732–0.001809] | 0.274597 [0.272577–0.282913] | 0.276336 [0.274311–0.284698] | 0.266597 [0.264987–0.276979] |
| float32 | FaissImputer 0.3.20 | 0.011751 [0.011496–0.020158] | 0.098557 [0.097797–0.105352] | 0.110645 [0.109553–0.119286] | 0.092589 [0.091738–0.097475] |
| float32 | FaissImputer 0.3.21 | 0.011529 [0.011303–0.012136] | 0.100393 [0.097960–0.104073] | 0.111695 [0.109857–0.115707] | 0.092728 [0.092239–0.099031] |
| float64 | KNNImputer | 0.002092 [0.001966–0.002734] | 0.332677 [0.320049–0.341911] | 0.334648 [0.322630–0.344645] | 0.326802 [0.324214–0.340001] |
| float64 | FaissImputer 0.3.20 | 0.013644 [0.011892–0.014842] | 0.119927 [0.116810–0.124375] | 0.133895 [0.128702–0.138019] | 0.114216 [0.112071–0.117043] |
| float64 | FaissImputer 0.3.21 | 0.012744 [0.011975–0.013577] | 0.119328 [0.117299–0.126130] | 0.132147 [0.129274–0.139530] | 0.113424 [0.111527–0.118916] |

## Matched speedups

Each entry is median [min–max] of nine paired ratios. KNN/Faiss uses the KNN baseline for the same training regime. First and additional transforms remain separate.

| Regime | dtype | Denominator | KNN/Faiss fit | KNN/Faiss first transform | KNN/Faiss total | KNN/Faiss additional transform |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | FaissImputer 0.3.20 | 0.353 [0.336–0.372] | 1.936 [1.904–2.100] | 1.885 [1.855–2.043] | 1.964 [1.798–2.066] |
| complete | float32 | FaissImputer 0.3.21 | 0.354 [0.337–0.377] | 1.894 [1.874–2.063] | 1.845 [1.826–2.008] | 1.930 [1.888–2.015] |
| complete | float64 | FaissImputer 0.3.20 | 0.352 [0.280–0.390] | 1.682 [1.641–1.749] | 1.639 [1.603–1.703] | 1.666 [1.593–1.713] |
| complete | float64 | FaissImputer 0.3.21 | 0.355 [0.258–0.399] | 1.668 [1.650–1.708] | 1.628 [1.606–1.667] | 1.656 [1.617–1.678] |
| available | float32 | FaissImputer 0.3.20 | 0.148 [0.086–0.157] | 2.782 [2.684–2.821] | 2.504 [2.309–2.522] | 2.863 [2.841–2.906] |
| available | float32 | FaissImputer 0.3.21 | 0.152 [0.143–0.160] | 2.766 [2.630–2.799] | 2.495 [2.381–2.514] | 2.871 [2.695–2.917] |
| available | float64 | FaissImputer 0.3.20 | 0.166 [0.145–0.215] | 2.755 [2.627–2.889] | 2.494 [2.381–2.630] | 2.886 [2.791–2.997] |
| available | float64 | FaissImputer 0.3.21 | 0.169 [0.156–0.212] | 2.757 [2.682–2.857] | 2.494 [2.443–2.598] | 2.859 [2.820–2.985] |

### Release-to-release comparison

Speedups use 0.3.20 time / 0.3.21 time. Total-duration change is calculated per pair as `100 * (0.3.21 total / 0.3.20 total - 1)` and then summarized; positive percentages mean 0.3.21 took longer.

| Regime | dtype | Fit ratio | First-transform ratio | Total ratio | Additional-transform ratio | Total-duration change (%) |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | 0.9997 [0.9804–1.0217] | 0.9812 [0.9759–0.9948] | 0.9818 [0.9761–0.9947] | 0.9829 [0.9754–1.0648] | 1.8541 [0.5308–2.4533] |
| complete | float64 | 1.0143 [0.8883–1.1093] | 0.9978 [0.9674–1.0122] | 0.9998 [0.9649–1.0123] | 0.9974 [0.9600–1.0369] | 0.0155 [-1.2130–3.6411] |
| available | float32 | 1.0186 [0.9617–1.7327] | 0.9859 [0.9525–1.0430] | 0.9972 [0.9727–1.0349] | 1.0010 [0.9455–1.0266] | 0.2781 [-3.3761–2.8084] |
| available | float64 | 1.0746 [0.9802–1.1691] | 1.0010 [0.9802–1.0242] | 1.0001 [0.9849–1.0344] | 1.0021 [0.9731–1.0198] | -0.0144 [-3.3222–1.5335] |

## Memory

MiB, median [min–max] over nine workers. Peak RSS covers the full worker lifetime, including warmup, fit, all three transform calls, and validation; it is not isolated first-transform memory. Post-fit RSS change is after-fit RSS minus before-fit RSS and includes allocator effects; it is not exact fitted-model size.

| Regime | dtype | Method | Full-worker peak RSS (MiB) | Post-fit RSS change (MiB) |
| --- | --- | --- | --- | --- |
| complete | float32 | KNNImputer | 236.473 [236.195–241.945] | 1.402 [1.398–1.402] |
| complete | float32 | FaissImputer 0.3.20 | 150.246 [150.094–150.395] | 2.977 [2.977–2.980] |
| complete | float32 | FaissImputer 0.3.21 | 150.230 [149.898–150.430] | 2.930 [2.926–2.930] |
| complete | float64 | KNNImputer | 249.961 [249.785–250.113] | 3.055 [3.055–3.055] |
| complete | float64 | FaissImputer 0.3.20 | 152.902 [152.578–153.227] | 6.047 [5.848–6.148] |
| complete | float64 | FaissImputer 0.3.21 | 152.785 [152.605–153.301] | 5.977 [5.977–5.980] |
| available | float32 | KNNImputer | 236.344 [236.102–236.465] | 1.402 [1.398–1.402] |
| available | float32 | FaissImputer 0.3.20 | 249.129 [248.953–249.316] | 9.910 [9.910–9.910] |
| available | float32 | FaissImputer 0.3.21 | 248.996 [248.895–249.199] | 9.477 [9.477–9.480] |
| available | float64 | KNNImputer | 249.918 [249.660–250.109] | 3.055 [3.055–3.055] |
| available | float64 | FaissImputer 0.3.20 | 251.832 [251.617–251.953] | 10.309 [10.305–10.422] |
| available | float64 | FaissImputer 0.3.21 | 251.973 [251.789–252.320] | 13.062 [13.059–13.062] |

## Reconstruction quality and output agreement

RMSE and MAE: median [min–max] across three seeds, with 1,200 missing query cells scored per seed. Values summarize error against benchmark ground truth. Similar aggregate errors do not establish equality of individual imputed values or algorithmic equivalence.

| Regime | dtype | Method | RMSE | MAE |
| --- | --- | --- | --- | --- |
| complete | float32 | KNNImputer | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float32 | FaissImputer 0.3.20 | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float32 | FaissImputer 0.3.21 | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float64 | KNNImputer | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| complete | float64 | FaissImputer 0.3.20 | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| complete | float64 | FaissImputer 0.3.21 | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| available | float32 | KNNImputer | 0.1806746367 [0.1758943104–0.1836455924] | 0.1364361146 [0.1306204932–0.1404130460] |
| available | float32 | FaissImputer 0.3.20 | 0.1806746352 [0.1758943104–0.1836455907] | 0.1364361139 [0.1306204922–0.1404130433] |
| available | float32 | FaissImputer 0.3.21 | 0.1806746352 [0.1758943104–0.1836455907] | 0.1364361139 [0.1306204922–0.1404130433] |
| available | float64 | KNNImputer | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |
| available | float64 | FaissImputer 0.3.20 | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |
| available | float64 | FaissImputer 0.3.21 | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |

Agreement below uses recorded full-output SHA-256 hashes and worker-computed maximum absolute differences on scored cells. Hash counts cover nine timing pairs, but only three distinct seed inputs. Differences are maxima over those pairs.

| Regime | dtype | Comparison | Matching full-output hashes | Max scored-cell difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | 0.3.21 vs KNNImputer | 9/9 | 0 | 0 | 0 |
| complete | float32 | 0.3.21 vs 0.3.20 | 9/9 | 0 | 0 | 0 |
| complete | float64 | 0.3.21 vs KNNImputer | 9/9 | 0 | 0 | 0 |
| complete | float64 | 0.3.21 vs 0.3.20 | 9/9 | 0 | 0 | 0 |
| available | float32 | 0.3.21 vs KNNImputer | 0/9 | 4.76837158203e-07 | 1.74059017199e-09 | 2.69850716772e-09 |
| available | float32 | 0.3.21 vs 0.3.20 | 9/9 | 0 | 0 | 0 |
| available | float64 | 0.3.21 vs KNNImputer | 0/9 | 8.881784197e-16 | 0 | 0 |
| available | float64 | 0.3.21 vs 0.3.20 | 9/9 | 0 | 0 | 0 |

## Donor counts and observed missingness

One row per policy/dtype/seed after verifying identical case metadata and input fingerprints across variants and repeats. Complete donors means fully observed training rows; available mode can also use partially observed rows as feature-specific donors, so this is not its total eligible donor count. Feature-specific donor counts were not recorded in this artifact.

| Regime | dtype | Seed | Complete donors | Observed training missing (%) | Query patterns | Scored cells |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | 101 | 20000 | 0.0000 | 296 | 1200 |
| complete | float32 | 202 | 20000 | 0.0000 | 288 | 1200 |
| complete | float32 | 303 | 20000 | 0.0000 | 286 | 1200 |
| complete | float64 | 101 | 20000 | 0.0000 | 296 | 1200 |
| complete | float64 | 202 | 20000 | 0.0000 | 288 | 1200 |
| complete | float64 | 303 | 20000 | 0.0000 | 286 | 1200 |
| available | float32 | 101 | 2447 | 9.9530 | 296 | 1200 |
| available | float32 | 202 | 2435 | 10.1072 | 288 | 1200 |
| available | float32 | 303 | 2413 | 10.0725 | 286 | 1200 |
| available | float64 | 101 | 2447 | 9.9530 | 296 | 1200 |
| available | float64 | 202 | 2435 | 10.1072 | 288 | 1200 |
| available | float64 | 303 | 2413 | 10.0725 | 286 | 1200 |

## Interpretation and limits

- 0.3.21 and 0.3.20 have matching full-output hashes in 36/36 paired workers, with zero recorded scored-cell differences. This is evidence for these measured cases, not a claim of equivalence for all inputs.
- In the available regime, 0.3.21 first-transform paired speedup against KNN is 2.757–2.766× across the two dtypes. Its full-worker peak RSS is higher than the corresponding KNN baseline. Timing improvements therefore do not imply a memory improvement for this workload.
- Complete float32 0.3.21 total duration increased by a median 1.8541% in the matched pairs. The small release-to-release differences are reported as observations from this run; no statistical significance or cause is established.
- Available-mode hashes differ from KNN despite close RMSE/MAE. Complete-mode hash agreement here occurs with fully observed training data and does not generalize to the incomplete-training OFAT setup.
- This general synthetic workload does not demonstrate that the 0.3.21 precision-repair path was triggered. It is not a targeted underflow/overflow test.
- Startup, data generation, warmup, validation, RSS sampling, and garbage collection between timed phases are excluded from the measured timings. The additional transforms are warm calls on the already fitted model. These results do not establish behavior on all data sizes, hardware, or datasets.

## Reproduction

The analysis uses only the Python standard library and the preserved ZIP; it does not install or execute either imputer. The analysis workflow regenerates this report and the full-precision JSON, compares them byte-for-byte with the committed copies, and uploads the generated outputs. It runs for relevant pull requests and can be dispatched manually once present on the default branch.

[Analysis workflow](../../.github/workflows/analyze-released-versions.yml)
