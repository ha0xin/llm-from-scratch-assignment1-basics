2026-02-10 23:47:34 CST | confirmed remote lfs tests pass (47 passed, 1 xpassed) after uv sync.
2026-02-10 23:50:55 CST | added tokenize/train/plot/slurm scripts for LM experiments.
2026-02-10 23:53:58 CST | fixed slurm scripts to use SLURM_SUBMIT_DIR after permission error.
2026-02-10 23:54:51 CST | added token subset utility script.
2026-02-10 23:57:36 CST | initialized writeup/checklist/manifest templates for continuous backfill.
2026-02-11 00:07:31 CST | optimized tokenizer with pretoken BPE cache; local tokenizer/bpe tests pass.
2026-02-11 00:09:26 CST | tokenizer throughput benchmarked on lfs: baseline OWT32MiB=1.077MiB/s; after cache OWT32MiB=6.420MiB/s; TinyValid=8.692MiB/s.
2026-02-11 00:37:47 CST | applied tokenize_corpus tuning (prefetch/chunk options, uint16 writes) and benchmarked on lfs valid splits.
2026-02-11 00:37:47 CST | lfs throughput (MiB/s): Tiny valid old=9.373, opt16=9.521, opt64=9.678; OWT valid old=6.777, opt16=6.567, opt64=6.459. Keeping default chunk=4MiB prefetch=0 for OWT robustness.
2026-02-11 00:45:10 CST | confirmed lfs tokenization complete for tinystories/owt train+valid; synced full tokenized binaries (~6.3GB) and meta files to local artifacts/tokenized/.
