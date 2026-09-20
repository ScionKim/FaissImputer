Phase-separated memory: same-data fit_transform benchmark
Main findings
Retained fitted memory is negligible for all methods: at 20,000 rows, 9–10 MiB (available-donor), 2–3 MiB (KNNImputer), 2–3 MiB (complete-donor) — 0.3–3.8% of process peak RSS.
Whole-process peak RSS is dominated by transform for KNNImputer and available-donor, where sampled transform-phase peaks closely track ru_maxrss; complete-donor shows somewhat larger gaps between sampled transform peaks and lifetime ru_maxrss.
Fit-phase peaks are flat across sizes (~135–158 MiB), dominated by baseline process memory rather than fit work.
Transform-phase peaks diverge by method. KNNImputer rises steeply then plateaus (10k→20k rows float32: 563→570 MiB), chunk-limited by sklearn_working_memory_mib=256. Available-donor grows gradually (145→257 MiB). Complete-donor stays nearly flat (~135→155 MiB).
The dtype effect is concentrated in KNNImputer (20k rows: 724 float64 vs 570 float32 MiB); the Faiss methods show small dtype differences.
Timings from --phase-memory runs are reference-only: the sampler thread adds observer overhead, so canonical performance comparisons should continue to use normal benchmark runs.
Conditions

Same workload as [fit-transform-110e37bf.md](fit-transform-110e37bf.md): sizes 1000/3000/10000/20000, float32/float64, KNNImputer / FaissImputer[complete] / FaissImputer[available], `fit_transform` + `fit_then_transform` APIs, seeds 101/202/303, 3 repeats, 20 features, 5 neighbors, 10% target missing rate, 5 guaranteed complete rows, threads=1 (pinned), 432 workers — plus `--phase-memory`.

Runner: Azure AMD EPYC 9V74, the same runner type as the canonical 835160d9034f validation run, so the two runs are comparable.

Phase-memory fields are populated for fit_then_transform records only. The fit_transform API performs a single call that cannot attribute memory to phases, so its phase-specific fields are null by design. All phase-memory measurements are Linux-only (/proc/self/statm); non-Linux workers would report null, matching the existing helpers.

Validation

Two runs support this report:

Canonical validation run (source 835160d9034f, phase_memory_measured=false): 432/432 workers passed with the instrumentation present but inactive — no sampler thread, no GC between phases, phase-memory fields null. Timing behavior matches the original benchmark exactly. Raw results: benchmarks/results/fit-transform-835160d9034f.json.
This run (source ac84bfa1abec, phase_memory_measured=true): 432/432 workers passed, all output-equality checks passed. On split-API records total_seconds equals fit_seconds + transform_seconds exactly, confirming the sampler setup/teardown, GC pauses, and RSS snapshots sit outside the timed intervals. No negative peaks; no implausible retained values.
Retained fitted memory

Median fit_retained_rss_mib (MiB), the current-RSS change across fit() with a gc.collect() pause before each snapshot:

Rows	dtype	KNNImputer	FaissImputer[complete]	FaissImputer[available]
1000	float32	0.1	0.1	0.4
1000	float64	0.1	0.1	0.4
3000	float32	0.1	0.1	1.3
3000	float64	0.2	0.3	1.7
10000	float32	0.9	0.9	4.7
10000	float64	1.5	1.7	6.5
20000	float32	1.7	1.7	9.4
20000	float64	3.2	3.3	10.1

Available-donor retention grows roughly linearly with rows, consistent with its retained donor representation. KNNImputer and complete-donor retain only a few MiB at these sizes. This is a signed RSS delta, so allocator-retained memory can exceed live model objects; even so, the magnitudes are tiny.

Phase peaks

Median fit-phase peaks sit at 134–158 MiB across every size, dtype, and method — flat because fit() allocates almost nothing beyond what it retains, leaving baseline process memory (interpreter, imports, already-materialized data) dominant.

Median transform-phase peaks (MiB) diverge sharply:

Rows	dtype	KNNImputer	FaissImputer[complete]	FaissImputer[available]
1000	float32	150.4	134.6	144.7
1000	float64	154.1	134.7	146.1
3000	float32	261.5	136.5	155.2
3000	float64	269.8	136.7	156.3
10000	float32	562.8	139.7	194.9
10000	float64	696.9	142.6	192.5
20000	float32	569.8	148.9	256.5
20000	float64	724.1	154.9	264.0

These are sampled lower bounds (5 ms RSS polls), not exact maxima.

Comparison with whole-process peak RSS

Sampled transform-phase peaks track whole-process ru_maxrss medians closely for KNNImputer and available-donor (typically within a few MiB; e.g. 20k float32 available-donor: 256.5 vs 256.5 MiB; KNNImputer: 569.8 vs 572.3 MiB). Complete-donor shows somewhat larger gaps (up to ~11 MiB at 20k float64: 154.9 vs 165.9 MiB), consistent with brief allocations the 5 ms sampler can miss — the sampled values are lower bounds by construction. In short, whole-process peak RSS is dominated by transform for KNNImputer and available-donor; complete-donor shows somewhat larger gaps between sampled transform peaks and lifetime ru_maxrss.

Retained fitted memory is 0.3–3.8% of peak RSS at 20k rows (e.g. available-donor 20k float32: 9.4 vs 256.5 MiB; KNNImputer 20k float64: 3.2 vs 724.0 MiB). This quantifies the [110e37bf report](fit-transform-110e37bf.md)'s caveat that its peak-RSS figures are not retained-model memory: the overwhelming majority of peak RSS is transient working buffers held during transform, not state the fitted model keeps.

Scaling across sizes and dtypes
KNNImputer transform peaks rise steeply from 1k to 10k rows, then plateau: 10k→20k float32 moves only 563→570 MiB. The plateau is chunk-limited — sklearn processes distance blocks within sklearn_working_memory_mib=256, so peak stops growing once chunks saturate.
Available-donor transform peaks grow gradually with rows (145→257 MiB float32), reflecting its batched matrix-distance backend over the full donor set.
Complete-donor transform peaks are nearly flat (~135→155 MiB): the donor set is small (complete rows only), so per-query work barely grows with dataset size.
dtype: KNNImputer float64 peaks run well above float32 (20k: 724 vs 570 MiB) because its working buffers double with dtype width. The Faiss methods show only small dtype differences (20k available: 264 vs 257 MiB).
Provenance and reproduction
Source commit ac84bfa1abec, candidate version 0.3.20+bench.ac84bfa1abec; 432/432 workers completed in 28.3 minutes.
Raw results: benchmarks/results/fit-transform-ac84bfa1abec-phase-memory.json (phase_memory_measured=true in run parameters).
Reproduce: dispatch the Same-data fit-transform benchmark workflow on main with the phase_memory input checked. Sizes, seeds, repeats, and timeouts are hardcoded in the workflow.
Scope

Limited to same-data imputation with 20 features, 5 neighbors, 10% missingness, and a single pinned thread. Sampled peaks are lower bounds on true phase peaks. Retained memory is an RSS delta that includes allocator effects, not a direct measure of live Python objects. Measurements are Linux-only.