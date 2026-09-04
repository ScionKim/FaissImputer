# Available-donor batching and thread benchmarks

These measurements evaluate experimental batch budgets against FaissImputer
0.3.1. The experiments do not change the released product.

## Scope and measurement

- Recorded on 2026-09-04 UTC; 34/34 benchmark worker runs completed successfully.
- Synthetic data, seed 101, 20 features, 300 held-out query rows.
- Training missingness: 10% MCAR; each query has four missing features.
- Available-donor policy, L2 distance, mean aggregation, 5 neighbors.
- Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.0, Faiss 1.15.0.
- All runners exposed four logical CPUs and four CPUs in their affinity mask.
- Time is fit + transform; repeated timings are summarized by the median.
- Peak RSS is the maximum recorded whole-worker peak across repeats.
  Time and RSS summaries need not come from the same worker.
  RSS includes imports, data generation, warmup, inputs, and validation.
  Timing excludes those operations and process startup.
- 16/64/128 MiB are batch-sizing budgets, not total process-memory limits.
- Each measurement uses a fresh sequential worker process.
- Input fingerprints and recorded thread counts matched within each experiment.

The three experiments ran on different CPU models. Compare methods within
each experiment; do not interpret differences between their absolute times
as scaling or additional optimization gains.

## 1. Batch budgets: 500,000 training rows

CPU: Intel Xeon Platinum 8573C. One thread, three repeats; 12/12 cases succeeded.

Source revision: `ec16e7f72833bcc6806dfa7b3fa3f847b1a7b44c`.
Output: `available_batches.json`.

| Method | Median total (s) | Peak RSS (MiB) | Query rows per batch |
|---|---:|---:|---:|
| KNNImputer | 11.595 | 787.7 | N/A |
| Original available, 16 MiB | 28.962 | 569.8 | 2 |
| Experimental available, 64 MiB | 11.394 | 670.3 | 11 |
| Experimental available, 128 MiB | 9.494 | 722.9 | 22 |

The 128 MiB candidate was 3.05x as fast as the original and 1.22x as fast
as KNNImputer. Its peak RSS was 26.9% higher than the original.

## 2. Batch budgets: 1,000,000 training rows

CPU: Intel Xeon 6973P-C. One thread, one repeat; 4/4 cases succeeded.
This is a single-run pilot, not a repeated timing estimate.

Source revision: `94c152a91e6437f399e46bd9f2a1bfe95386e452`.
Output: `available_batches.json` from this separate run.

| Method | Total (s) | Peak RSS (MiB) | Query rows per batch |
|---|---:|---:|---:|
| KNNImputer | 21.724 | 1066.5 | N/A |
| Original available, 16 MiB | 81.916 | 993.5 | 1 |
| Experimental available, 64 MiB | 26.756 | 1051.0 | 5 |
| Experimental available, 128 MiB | 19.681 | 1163.1 | 11 |

The 128 MiB candidate was 4.16x as fast as the original and 1.10x as fast
as KNNImputer. Its peak RSS was 17.1% higher than the original and 9.1%
higher than KNNImputer. It is not a universal memory-saving optimization.

## 3. Thread counts: 500,000 training rows

CPU: AMD EPYC 7763. Three repeats per method and thread count;
18/18 cases succeeded. Only KNNImputer and the 128 MiB candidate were run.

Source revision: `90c8cfb81f1b1771bfdd052e84eafc6ab7b73e90`.
Output: `available_threads.json`.

| Threads | KNN total (s) | 128 MiB total (s) | KNN peak RSS (MiB) | 128 MiB peak RSS (MiB) |
|---|---:|---:|---:|---:|
| 1 | 9.027 | 6.179 | 803.6 | 722.8 |
| 2 | 8.710 | 5.858 | 789.5 | 724.9 |
| 4 | 9.035 | 6.392 | 793.7 | 728.8 |

The candidate was 1.41-1.49x as fast as KNNImputer at the same thread count.
Two threads reduced candidate time by 5.2% versus one thread; four threads
increased it by 3.4%.

FAISS/OpenMP and BLAS thread limits changed together. This does not isolate
FAISS-only parallelism, and it does not establish a universal best thread count.

## Output checks and limits

- In both batch-budget experiments, the 64 and 128 MiB candidates had
  zero output difference from the original available implementation.
- In the thread experiment, outputs were unchanged across thread counts
  and repeats for each method. The original 16 MiB variant was not rerun.
- Maximum absolute output difference from KNNImputer was
  2.384185791015625e-7 across these experiments.
- Shape, finiteness, observed-value preservation, and input preservation
  checks passed.
- These observations are not proof of general numerical equivalence.
  All experiments used one synthetic seed; the million-row case ran once.

## Reproduction

Use the source revision listed for each experiment and the dependency
versions above. These commands describe the historical experiments;
future product changes may require updating the experiment helpers.

```bash
# Experiment 1
python -u -m benchmarks.benchmark_available_batches --train-sizes 500000 --threads 1 --queries 300 --seeds 101 --patterns random --repeats 3 --timeout-seconds 120 --budget-seconds 600 --output benchmark_outputs/available_batches.json

# Experiment 2, in a separate run
python -u -m benchmarks.benchmark_available_batches --train-sizes 1000000 --threads 1 --queries 300 --seeds 101 --patterns random --repeats 1 --timeout-seconds 180 --budget-seconds 900 --output benchmark_outputs/available_batches.json

# Experiment 3
python -u -m benchmarks.benchmark_available_threads --train-sizes 500000 --threads 1 2 4 --queries 300 --seeds 101 --patterns random --repeats 3 --timeout-seconds 180 --budget-seconds 900 --output benchmark_outputs/available_threads.json
```

## Decision

The results support implementing and regression-testing the 128 MiB
batch-budget change, with its memory tradeoff documented. They do not
justify forcing a thread count or promising general speedups.

At the time of these experiments, the released implementation still used
16 MiB. Product changes and release validation are separate follow-up work.
