# Complete-aggregation benchmark: v0.3.10 preparation

The candidate reduced first-transform time in all 10 `complete` policy groups: **16.69–38.77% for mean** and **29.69–60.96% for median**, with improvement in every seed. The `available` control was slower in 9 of 10 groups; its paired deltas ranged from −3.59% to +3.84%.

Measurements come from [Actions run 34154495281, attempt 1](https://github.com/ScionKim/FaissImputer/actions/runs/34154495281). Baseline source is [v0.3.8, `099c94107839a4bf6d3811f39541ed609530ed20`](https://github.com/ScionKim/FaissImputer/tree/099c94107839a4bf6d3811f39541ed609530ed20); candidate source is [`e5ef4825213d86228e3e041ea4a69c3396f39416`](https://github.com/ScionKim/FaissImputer/tree/e5ef4825213d86228e3e041ea4a69c3396f39416). The candidate still reported **0.3.9 metadata** and contains the runtime implementation prepared for **0.3.10**. These are source-checkout measurements, not released-wheel measurements. The [paired-workload-scaling.yml workflow at the measured commit](https://github.com/ScionKim/FaissImputer/blob/e5ef4825213d86228e3e041ea4a69c3396f39416/.github/workflows/paired-workload-scaling.yml) and archived harness establish the procedure.

To reproduce this revision in GitHub Actions, open the linked run and choose **Re-run all jobs**, retaining its recorded workflow revision. Check the new manifest for the baseline and candidate SHAs above, the dependency versions, and one-thread settings; require 60/60 valid comparisons and 20/20 valid groups. A fresh runner can produce different timings. Dispatching the current branch instead measures that branch's candidate.

Both versions ran sequentially on the same Linux/Azure runner reporting an AMD EPYC 7763 and four logical CPUs. FAISS and native thread pools were restricted to one thread. Dependencies included Python 3.12.14, NumPy 2.5.2, faiss-cpu 1.15.0, scikit-learn 1.9.0, and SciPy 1.18.1. Each configuration used seeds 101, 202, and 303, with two fresh-process observations per version per seed in **ABBA order**: baseline, candidate, candidate, baseline. Configuration order was shuffled. Fit and first transform were timed separately after a small warmup with a different model.

Fixtures used ordinary-scale float32 values, 8,192 training rows, `k=5`, eight query missingness patterns, and 25% query missingness. Half the training rows were complete: `complete` used 4,096 donors, while `available` used all 8,192 rows. Compare versions within each policy; their donor semantics differ. Query scaling held 32 features; feature scaling held 512 queries.

All **240 workers**, **60/60 seed/configuration comparisons**, and **20/20 groups** validated. Saved output arrays were byte-identical within each comparison, fitted-statistic hashes agreed, and no worker stderr was recorded. Artifact audit also verified the harness and package-source hashes. Output agreement applies to these fixtures only.

Times below are arithmetic means of six observations per version. For each seed, let `r` be candidate mean time divided by baseline mean time. Paired delta is `100 × [(r101 × r202 × r303)^(1/3) − 1]`; it can differ from the ratio of the displayed overall means. Negative means faster. Seed range shows the three paired deltas. Spread is the largest `100 × (max − min) / mean` within either version and seed, not a confidence interval.

| Policy | Strategy | Queries | Features | 0.3.8 ms | Candidate ms | Paired delta % | Seed range % | Max spread % |
|---|---|---:|---:|---:|---:|---:|---|---:|
| complete | mean | 128 | 32 | 8.328 | 6.668 | −19.92 | −21.71 to −18.55 | 4.43 |
| complete | median | 128 | 32 | 11.501 | 7.112 | −38.16 | −39.46 to −36.69 | 5.51 |
| available | mean | 128 | 32 | 55.383 | 55.821 | +0.79 | +0.29 to +1.42 | 1.64 |
| available | median | 128 | 32 | 65.653 | 66.815 | +1.77 | +1.53 to +2.09 | 0.86 |
| complete | mean | 512 | 32 | 22.531 | 15.843 | −29.88 | −34.01 to −22.17 | 29.65 |
| complete | median | 512 | 32 | 34.087 | 15.383 | −54.87 | −55.11 to −54.75 | 2.07 |
| available | mean | 512 | 32 | 212.246 | 220.415 | +3.84 | +2.43 to +6.53 | 4.97 |
| available | median | 512 | 32 | 234.762 | 236.796 | +0.86 | +0.20 to +1.71 | 4.55 |
| complete | mean | 2048 | 32 | 76.313 | 46.724 | −38.77 | −39.18 to −38.43 | 1.16 |
| complete | median | 2048 | 32 | 121.434 | 47.405 | −60.96 | −61.52 to −60.40 | 2.58 |
| available | mean | 2048 | 32 | 841.297 | 855.969 | +1.77 | −1.00 to +3.96 | 5.10 |
| available | median | 2048 | 32 | 920.312 | 928.473 | +0.88 | +0.08 to +1.56 | 0.45 |
| complete | mean | 512 | 8 | 19.759 | 12.685 | −35.80 | −36.44 to −35.14 | 3.87 |
| complete | median | 512 | 8 | 32.295 | 12.945 | −59.85 | −62.91 to −58.07 | 23.43 |
| available | mean | 512 | 8 | 138.342 | 133.328 | −3.59 | −7.01 to −1.66 | 9.36 |
| available | median | 512 | 8 | 139.817 | 139.996 | +0.13 | −2.34 to +2.39 | 4.03 |
| complete | mean | 512 | 128 | 49.815 | 41.501 | −16.69 | −16.87 to −16.42 | 9.99 |
| complete | median | 512 | 128 | 66.054 | 46.779 | −29.69 | −35.01 to −20.13 | 48.38 |
| available | mean | 512 | 128 | 562.120 | 567.726 | +1.00 | +0.16 to +1.87 | 1.34 |
| available | median | 512 | 128 | 649.309 | 657.811 | +1.31 | +1.17 to +1.45 | 0.86 |

The largest proportional `available` slowdown was mean at 512 queries/32 features: **+3.84%**, or **+8.17 ms** between displayed arithmetic means. Several control slowdowns occurred in all three seeds; dismissing all of them as noise is unsupported. Repeated runs would help establish their persistence.

Some `complete` measurements were noisy: mean at 512/32 had 29.65% maximum spread; median at 512/8 and 512/128 had 23.43% and 48.38%. Every complete seed still improved, but exact effect sizes need replication. Fit plus first-transform time also improved in every complete group and seed, with grouped reductions of 12.42–57.65%. Fit alone showed mixed changes and does not support a dependable improvement claim.

Peak RSS covers imports, warmup, input loading, fit, and transform, sampled before output validation/saving. It measures whole-process peak usage, not retained fitted memory or memory attributable to one phase. This pilot excludes extreme-value repair cost, repeated transforms, the `fit_transform` API, comparisons with `KNNImputer`, imputation quality, and other hardware/thread settings.

Raw-data download: [full benchmark ZIP for the v0.3.10 release](https://github.com/ScionKim/FaissImputer/releases/download/v0.3.10/paired-complete-aggregation-34154495281-1.zip) (27,848,786 bytes). It contains the harness, manifest and source hashes, dependency/CPU records, fixture fingerprints, worker measurements and output arrays, and seed/group CSV/JSON results, including fit, total time, and peak RSS.
