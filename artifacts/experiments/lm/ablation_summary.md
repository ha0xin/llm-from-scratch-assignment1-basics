# Ablation Summary

| tag | run | best_val_loss | ffn_type | d_ff | tokens_seen |
|---|---|---:|---|---:|---:|
| baseline_lr1e3_5k | ts_lr_1e3 | 1.5593 | swiglu | 1344 | 81920000 |
| no_rmsnorm_lr1e3_5k | ts_ablate_no_rmsnorm_lr1e3_5k | 1.5773 | swiglu | 1344 | 81920000 |
| no_rmsnorm_lr3e4_5k | ts_ablate_no_rmsnorm_lr3e4_5k | 1.7679 | swiglu | 1344 | 81920000 |
| postnorm_lr1e3_5k | ts_ablate_postnorm_lr1e3_5k | 1.5553 | swiglu | 1344 | 81920000 |
| no_rope_lr1e3_5k | ts_ablate_no_rope_lr1e3_5k | 1.6537 | swiglu | 1344 | 81920000 |
| silu_lr1e3_5k_dff2048 | ts_ablate_silu_dff2048_lr1e3_5k | 1.5835 | silu | 2048 | 81920000 |
