# Batch Size Summary

| batch_size | tag | run | status | best_val_loss | tokens_seen | note |
|---:|---|---|---|---:|---:|---|
| 1 | bs1_lr3e4_5k | ts_bs1_lr3e4_5k | ok | 2.9371 | 1280000 |  |
| 32 | bs32_lr1e3_5k | ts_bs32_lr1e3_5k | ok | 1.6665 | 40960000 |  |
| 64 | bs64_lr1e3_5k | ts_lr_1e3 | ok | 1.5593 | 81920000 |  |
| 128 | bs128_lr15e3_5k | ts_bs128_lr15e3_5k | ok | 1.4614 | 163840000 |  |
| 256 | bs256_limit_probe | ts_bs256_limit_probe | ok | 2.2109 | 19660800 |  |
| 512 | bs512_limit_probe | ts_bs512_limit_probe | oom | - | - | CUDA OOM at first forward pass (batch=512) |
