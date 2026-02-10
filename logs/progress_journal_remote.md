2026-02-10 23:47:34 CST | confirmed remote lfs tests pass (47 passed, 1 xpassed) after uv sync.
2026-02-10 23:50:55 CST | added tokenize/train/plot/slurm scripts for LM experiments.
2026-02-10 23:53:58 CST | fixed slurm scripts to use SLURM_SUBMIT_DIR after permission error.
2026-02-10 23:54:51 CST | added token subset utility script.
2026-02-10 23:57:36 CST | initialized writeup/checklist/manifest templates for continuous backfill.
2026-02-11 00:07:31 CST | optimized tokenizer with pretoken BPE cache; local tokenizer/bpe tests pass.
2026-02-11T00:07:47+08:00 | canceled slow tokenize jobs 1411/1413 after throughput profiling; switching to optimized tokenizer cache build
2026-02-11T00:07:47+08:00 | resubmitted tokenize jobs with optimized tokenizer: 1416 1417 1418 1419
