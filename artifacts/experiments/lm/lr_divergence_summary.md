# Learning Rate Divergence Summary

| run | lr | init_val | final_val | max_val | max_iter | final/init | max/init | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ts_lr_1e4 | 1.0e-04 | 9.2687 | 2.1890 | 9.2687 | 0 | 0.24 | 1.00 | stable_or_improved |
| ts_lr_3e4 | 3.0e-04 | 9.2687 | 1.7795 | 9.2687 | 0 | 0.19 | 1.00 | stable_or_improved |
| ts_lr_1e3 | 1.0e-03 | 9.2687 | 1.5786 | 9.2687 | 0 | 0.17 | 1.00 | stable_or_improved |
| ts_lr_3e3_div | 3.0e-03 | 9.2687 | 1.6799 | 9.2687 | 0 | 0.18 | 1.00 | stable_or_improved |
| ts_lr_1e2_div | 1.0e-02 | 9.2687 | 2.5253 | 9.2687 | 0 | 0.27 | 1.00 | stable_or_improved |
| ts_lr_3e2_div | 3.0e-02 | 9.2687 | 3.7453 | 9.2687 | 0 | 0.40 | 1.00 | stable_or_improved |
| ts_lr_1e1_div | 1.0e-01 | 9.2687 | 4.4049 | 9.2687 | 0 | 0.48 | 1.00 | stable_or_improved |
| ts_lr_3e1_div_probe2 | 3.0e-01 | 9.2687 | 4.8583 | 33.6346 | 20 | 0.52 | 3.63 | unstable_spike |
| ts_lr_1e0_div_probe2 | 1.0e+00 | 9.2687 | 57.5486 | 149.8570 | 10 | 6.21 | 16.17 | divergent |
| ts_lr_3e0_div_probe3 | 3.0e+00 | 9.2687 | 117.5958 | 876.8639 | 5 | 12.69 | 94.61 | divergent |
