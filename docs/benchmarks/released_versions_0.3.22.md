# Released-package comparison: 0.3.22 and 0.3.21

[Benchmark index](README.md) · [Project README](../../README.md#performance)

This report compares installed PyPI releases with KNNImputer on one runner, using held-out queries. It is a separate-query `fit` followed by `transform` benchmark, not the same-data OFAT benchmark.

## Evidence and configuration

- [Preserved original artifact](../../benchmarks/results/released_versions_0.3.22.zip) (uploaded without modification).
- [Full-precision analysis](../../benchmarks/results/released_versions_0.3.22-summary.json) and [generator](../../benchmarks/analyze_released_versions.py).
- [GitHub Actions run](https://github.com/ScionKim/FaissImputer/actions/runs/36896955077/attempts/1).
- Benchmark source commit: `4b5b9cbdc6d629ffa26fb2b6b47a24c268254b81`; this identifies the benchmark runner, not an editable package installation.
- Archive SHA-256: `7a633e03bd86110380fae69d86713c420c1bfc3c81ccd61b95e56752c943532d`.
- `version_comparison_q300.json` SHA-256: `f83c041b08242c64819707ccaf3052323d6404f0b465b9e9d8961577eb902bf5`.
- Runner: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz; 4 logical CPUs, 4 CPUs in affinity; requested and recorded native thread counts are one.
- Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- 20,000 training rows, 300 held-out queries, 20 features, k=5, uniform weights; FaissImputer is configured with built-in L2, mean aggregation, and `index_factory="Flat"`. Each query has four missing features (20%), giving 1,200 scored cells per seed.
- Three seeds (101, 202, 303), three fresh sequential workers per seed and variant. Method order rotates across repeats. There are 108 successful workers and no failed checks.
- `complete` uses fully observed training data. `available` uses training data with 10% target random missingness. Each policy's KNN baseline receives the same inputs as its Faiss variants. These are different input regimes, so cross-policy speed and quality differences do not isolate donor policy.

The two environments have identical archived dependency freezes except for faiss-imputer (0.3.21 versus 0.3.22). KNNImputer runs in the 0.3.22 environment. The analyzer checks package versions, installed site-packages locations, common environment metadata, full worker-grid coverage, input fingerprints, and all stored summary statistics against raw records.

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
| float32 | KNNImputer | 0.001731 [0.001638–0.001806] | 0.334032 [0.328866–0.337097] | 0.335718 [0.330504–0.338792] | 0.330814 [0.324684–0.333268] |
| float32 | FaissImputer 0.3.21 | 0.004802 [0.004566–0.004878] | 0.179586 [0.177279–0.190163] | 0.184190 [0.181846–0.194889] | 0.177750 [0.175207–0.187654] |
| float32 | FaissImputer 0.3.22 | 0.004894 [0.004721–0.004979] | 0.179847 [0.178184–0.189446] | 0.184632 [0.183088–0.194356] | 0.179106 [0.175672–0.187546] |
| float64 | KNNImputer | 0.002589 [0.002066–0.002691] | 0.431318 [0.426136–0.447594] | 0.433917 [0.428202–0.450178] | 0.423208 [0.419775–0.433835] |
| float64 | FaissImputer 0.3.21 | 0.006993 [0.006739–0.007782] | 0.247083 [0.241084–0.259206] | 0.254865 [0.247965–0.266189] | 0.244602 [0.238706–0.256580] |
| float64 | FaissImputer 0.3.22 | 0.007272 [0.006544–0.007457] | 0.246787 [0.242797–0.262903] | 0.253577 [0.249505–0.270197] | 0.243457 [0.240091–0.257850] |

### Available training regime

| dtype | Method | Fit (s) | First transform (s) | Fit + first transform (s) | Additional transform (s) |
| --- | --- | --- | --- | --- | --- |
| float32 | KNNImputer | 0.001876 [0.001795–0.001987] | 0.315390 [0.309106–0.320321] | 0.317216 [0.310901–0.322231] | 0.310558 [0.305310–0.314608] |
| float32 | FaissImputer 0.3.21 | 0.012186 [0.012058–0.078001] | 0.146092 [0.143979–0.149455] | 0.159149 [0.156100–0.223673] | 0.140425 [0.136907–0.145045] |
| float32 | FaissImputer 0.3.22 | 0.012230 [0.011718–0.013125] | 0.146616 [0.144081–0.154068] | 0.158566 [0.156372–0.166606] | 0.142743 [0.140064–0.148397] |
| float64 | KNNImputer | 0.002423 [0.002290–0.002747] | 0.405947 [0.394843–0.412312] | 0.408319 [0.397532–0.414736] | 0.400458 [0.393009–0.408267] |
| float64 | FaissImputer 0.3.21 | 0.014000 [0.013220–0.014743] | 0.164853 [0.160738–0.166486] | 0.178853 [0.173991–0.180904] | 0.158239 [0.156342–0.162907] |
| float64 | FaissImputer 0.3.22 | 0.013310 [0.012835–0.013752] | 0.147824 [0.143693–0.151301] | 0.161417 [0.156599–0.165053] | 0.145496 [0.142070–0.147531] |

## Matched speedups

Each entry is median [min–max] of nine paired ratios. KNN/Faiss uses the KNN baseline for the same training regime. First and additional transforms remain separate.

| Regime | dtype | Denominator | KNN/Faiss fit | KNN/Faiss first transform | KNN/Faiss total | KNN/Faiss additional transform |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | FaissImputer 0.3.21 | 0.368 [0.339–0.377] | 1.868 [1.734–1.890] | 1.828 [1.699–1.850] | 1.859 [1.730–1.901] |
| complete | float32 | FaissImputer 0.3.22 | 0.360 [0.334–0.371] | 1.857 [1.736–1.880] | 1.817 [1.701–1.839] | 1.852 [1.738–1.897] |
| complete | float64 | FaissImputer 0.3.21 | 0.353 [0.296–0.382] | 1.778 [1.644–1.804] | 1.737 [1.609–1.759] | 1.740 [1.648–1.774] |
| complete | float64 | FaissImputer 0.3.22 | 0.355 [0.283–0.411] | 1.770 [1.621–1.806] | 1.733 [1.585–1.769] | 1.746 [1.632–1.770] |
| available | float32 | FaissImputer 0.3.21 | 0.150 [0.025–0.163] | 2.168 [2.088–2.193] | 2.005 [1.426–2.025] | 2.206 [2.115–2.272] |
| available | float32 | FaissImputer 0.3.22 | 0.151 [0.139–0.164] | 2.120 [2.073–2.209] | 1.971 [1.928–2.051] | 2.174 [2.085–2.217] |
| available | float64 | FaissImputer 0.3.21 | 0.178 [0.158–0.193] | 2.476 [2.372–2.520] | 2.291 [2.204–2.334] | 2.517 [2.499–2.578] |
| available | float64 | FaissImputer 0.3.22 | 0.182 [0.168–0.208] | 2.748 [2.704–2.824] | 2.539 [2.501–2.600] | 2.772 [2.700–2.837] |

### Release-to-release comparison

Speedups use 0.3.21 time / 0.3.22 time. Total-duration change is calculated per pair as `100 * (0.3.22 total / 0.3.21 total - 1)` and then summarized; positive percentages mean 0.3.22 took longer.

| Regime | dtype | Fit ratio | First-transform ratio | Total ratio | Additional-transform ratio | Total-duration change (%) |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | 0.9848 [0.9170–1.0172] | 0.9950 [0.9715–1.0078] | 0.9948 [0.9726–1.0075] | 0.9969 [0.9830–1.0118] | 0.5183 [-0.7437–2.8190] |
| complete | float64 | 0.9879 [0.9200–1.1623] | 0.9965 [0.9859–1.0065] | 0.9958 [0.9852–1.0076] | 0.9979 [0.9856–1.0073] | 0.4175 [-0.7498–1.5056] |
| available | float32 | 1.0001 [0.9389–6.6568] | 1.0026 [0.9491–1.0300] | 1.0015 [0.9530–1.4277] | 0.9948 [0.9331–1.0282] | -0.1464 [-29.9586–4.9348] |
| available | float64 | 1.0518 [0.9749–1.1486] | 1.1059 [1.0874–1.1586] | 1.1015 [1.0779–1.1520] | 1.0911 [1.0726–1.1125] | -9.2132 [-13.1962–-7.2268] |

## Memory

MiB, median [min–max] over nine workers. Peak RSS covers the full worker lifetime, including warmup, fit, all three transform calls, and validation; it is not isolated first-transform memory. Post-fit RSS change is after-fit RSS minus before-fit RSS and includes allocator effects; it is not exact fitted-model size.

| Regime | dtype | Method | Full-worker peak RSS (MiB) | Post-fit RSS change (MiB) |
| --- | --- | --- | --- | --- |
| complete | float32 | KNNImputer | 236.707 [236.434–240.516] | 1.402 [1.402–1.402] |
| complete | float32 | FaissImputer 0.3.21 | 150.684 [150.484–150.906] | 2.977 [2.777–2.980] |
| complete | float32 | FaissImputer 0.3.22 | 150.582 [150.262–150.758] | 2.926 [2.926–2.930] |
| complete | float64 | KNNImputer | 250.172 [249.988–251.547] | 3.055 [3.055–3.055] |
| complete | float64 | FaissImputer 0.3.21 | 153.344 [153.004–153.645] | 6.055 [6.055–6.086] |
| complete | float64 | FaissImputer 0.3.22 | 153.223 [152.969–153.711] | 5.980 [5.977–5.980] |
| available | float32 | KNNImputer | 236.656 [236.453–236.762] | 1.402 [1.398–1.402] |
| available | float32 | FaissImputer 0.3.21 | 249.531 [249.230–249.656] | 9.910 [9.910–9.914] |
| available | float32 | FaissImputer 0.3.22 | 249.395 [249.086–249.500] | 9.477 [9.477–9.508] |
| available | float64 | KNNImputer | 250.156 [249.887–250.227] | 3.055 [3.055–3.055] |
| available | float64 | FaissImputer 0.3.21 | 252.188 [252.078–252.473] | 10.312 [10.309–10.312] |
| available | float64 | FaissImputer 0.3.22 | 252.496 [252.309–252.645] | 13.062 [13.059–13.062] |

## Reconstruction quality and output agreement

RMSE and MAE: median [min–max] across three seeds, with 1,200 missing query cells scored per seed. Values summarize error against benchmark ground truth. Similar aggregate errors do not establish equality of individual imputed values or algorithmic equivalence.

| Regime | dtype | Method | RMSE | MAE |
| --- | --- | --- | --- | --- |
| complete | float32 | KNNImputer | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float32 | FaissImputer 0.3.21 | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float32 | FaissImputer 0.3.22 | 0.1762539355 [0.1653437192–0.1764681130] | 0.1326190845 [0.1233628692–0.1352229177] |
| complete | float64 | KNNImputer | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| complete | float64 | FaissImputer 0.3.21 | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| complete | float64 | FaissImputer 0.3.22 | 0.1762539366 [0.1653437173–0.1764681146] | 0.1326190853 [0.1233628695–0.1352229182] |
| available | float32 | KNNImputer | 0.1806746367 [0.1758943104–0.1836455924] | 0.1364361146 [0.1306204932–0.1404130460] |
| available | float32 | FaissImputer 0.3.21 | 0.1806746352 [0.1758943104–0.1836455907] | 0.1364361139 [0.1306204922–0.1404130433] |
| available | float32 | FaissImputer 0.3.22 | 0.1806746352 [0.1758943104–0.1836455907] | 0.1364361139 [0.1306204922–0.1404130433] |
| available | float64 | KNNImputer | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |
| available | float64 | FaissImputer 0.3.21 | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |
| available | float64 | FaissImputer 0.3.22 | 0.1806746338 [0.1758943090–0.1836455931] | 0.1364361131 [0.1306204930–0.1404130463] |

Agreement below uses recorded full-output SHA-256 hashes and worker-computed maximum absolute differences on scored cells. Hash counts cover nine timing pairs, but only three distinct seed inputs. Differences are maxima over those pairs.

| Regime | dtype | Comparison | Matching full-output hashes | Max scored-cell difference | Max RMSE difference | Max MAE difference |
| --- | --- | --- | --- | --- | --- | --- |
| complete | float32 | 0.3.22 vs KNNImputer | 9/9 | 0 | 0 | 0 |
| complete | float32 | 0.3.22 vs 0.3.21 | 9/9 | 0 | 0 | 0 |
| complete | float64 | 0.3.22 vs KNNImputer | 9/9 | 0 | 0 | 0 |
| complete | float64 | 0.3.22 vs 0.3.21 | 9/9 | 0 | 0 | 0 |
| available | float32 | 0.3.22 vs KNNImputer | 0/9 | 4.76837158203e-07 | 1.74059017199e-09 | 2.69850716772e-09 |
| available | float32 | 0.3.22 vs 0.3.21 | 9/9 | 0 | 0 | 0 |
| available | float64 | 0.3.22 vs KNNImputer | 0/9 | 8.881784197e-16 | 0 | 0 |
| available | float64 | 0.3.22 vs 0.3.21 | 9/9 | 0 | 0 | 0 |

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

- 0.3.22 and 0.3.21 have matching full-output hashes in 36/36 paired workers. The maximum recorded scored-cell difference is 0. This is evidence for these measured cases, not a claim of equivalence for all inputs.
- Available float64 first-transform speedup (0.3.21/0.3.22) is 1.1059×; 9/9 matched first transforms favor 0.3.22. The median paired duration changes are -9.5764% for first transform and -9.2132% for fit plus first transform; negative changes mean less time. The additional-transform paired speedup is 1.0911×.
- The other three policy/dtype cells have median release-to-release first-transform ratios of 0.9950–1.0026×. These are descriptive observations; no statistical significance is established.
- In the available regime, 0.3.22 first-transform paired speedup against KNN is 2.120–2.748× across the two dtypes. Its median full-worker peak RSS is higher than the corresponding KNN baseline for both dtypes. Peak RSS does not isolate the selected-distance workspace.
- Retained timing observation: 0.3.21 available float32, seed 101, repeat 1, has fit time 0.078001 s. The other eight fits range from 0.012058 to 0.013057 s. This worker contributes to the timing range and its matched total-time ratio; no samples are excluded.
- Available-mode full-output hashes differ from KNN despite close RMSE/MAE. The maximum scored-cell differences and metric differences are reported above. Complete-mode hash agreement here uses fully observed training data and does not generalize to incomplete-training OFAT cases or other inputs.
- This comparison uses Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz. The [earlier published 0.3.21 report](released_versions_0.3.21.md) used AMD EPYC 7763. Cross-run changes in absolute times or KNN-relative speedups do not isolate a package-version effect. Release comparisons in this report use matched records from this one Intel run.
- Startup, data generation, warmup, validation, RSS sampling, and garbage collection between timed phases are excluded from the measured timings. The additional transforms are warm calls on the already fitted model. These results do not establish behavior on all data sizes, hardware, or datasets.

## Reproduction

The analysis uses only the Python standard library and the preserved ZIP; it does not install or execute either imputer. The generator selects this archive with `--release 0.3.22`; omitting `--release` retains the historical 0.3.21 output. The Analyze released-version benchmark results workflow regenerates both releases and uploads their Markdown reports and full-precision JSON summaries. Committed outputs are compared byte-for-byte. For initial preservation, a push to `bench/released-0.3.22` or a manual run may generate the 0.3.22 files when both are absent; it still verifies the existing 0.3.21 files. Upload both new outputs, then the pull-request check requires and verifies both releases. A partially present output pair fails instead of silently skipping comparison.

[Analysis workflow](../../.github/workflows/analyze-released-versions.yml)
