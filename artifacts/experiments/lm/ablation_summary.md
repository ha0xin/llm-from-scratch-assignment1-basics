# TinyStories Ablation Summary

| tag | run | best_val_loss | best_iter | max_iters | tokens_seen | elapsed_sec |
|---|---|---:|---:|---:|---:|---:|
| baseline_lr1e3_5k | ts_lr_1e3 | 1.5593 | 4750 | 5000 | 81920000 | 255.65 |
| no_rmsnorm_lr1e3_5k | ts_ablate_no_rmsnorm_lr1e3_5k | 1.5773 | 4750 | 5000 | 81920000 | 235.48 |
| no_rmsnorm_lr3e4_5k | ts_ablate_no_rmsnorm_lr3e4_5k | 1.7679 | 4750 | 5000 | 81920000 | 236.66 |
| no_rope_lr1e3_5k | ts_ablate_no_rope_lr1e3_5k | 1.6537 | 4750 | 5000 | 81920000 | 243.49 |
| postnorm_lr1e3_5k | ts_ablate_postnorm_lr1e3_5k | 1.5553 | 4750 | 5000 | 81920000 | 251.35 |
| silu_lr1e3_5k | ts_ablate_silu_lr1e3_5k | 1.6110 | 4750 | 5000 | 81920000 | 240.47 |
