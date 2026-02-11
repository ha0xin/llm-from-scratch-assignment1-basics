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
2026-02-11 00:49:18 CST | submitted TinyStories LR sweep jobs on lfs: 1420(lr=1e-4), 1421(3e-4), 1422(1e-3), 1423(3e-3,max_iters=2000); plus 1424(lr=1e-2,max_iters=1000,pending).
2026-02-11 00:59:25 CST | TinyStories experiments submitted on lfs: main run 1427 (lr=1e-3,max_iters=20000) and ablations 1428(no_rmsnorm,1e-3),1429(no_rmsnorm,3e-4),1430(post_norm),1431(no_rope,pending),1432(silu,pending).
2026-02-11 01:01:24 CST | submitted OpenWebText main run 1433 (owt_main_lr1e3_20k, vocab=32000, batch=64, max_iters=20000), pending in queue.
2026-02-11 01:11:19 CST | submitted TinyStories batch-size runs: 1434(bs=1,lr=3e-4), 1435(bs=32,lr=1e-3), 1436(bs=128,lr=1.5e-3,pending).
2026-02-11 01:23:39 CST | submitted token-matched batch-size runs: 1437(bs=32,max_iters=10000), 1438(bs=128,max_iters=2500).
2026-02-11 01:33:26 CST | completed TinyStories main(ts_main_lr1e3_20k,best_val=1.3730), OWT main(owt_main_lr1e3_20k,best_val=4.0370), ablations and batch-size sweeps; synced all run artifacts/slurm logs to local.
2026-02-11 23:15:17 CST | audited handout deliverables end-to-end; added per-problem deliverable mapping and completed missing written responses in writeup.
