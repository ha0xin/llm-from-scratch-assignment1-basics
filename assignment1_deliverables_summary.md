# Assignment 1 Deliverables 汇总（逐题）

状态说明：`已完成` / `可选未做`  
统计口径：以 `cs336_spring2025_assignment1_basics.md` 中每个 `Deliverable` 为准。

| Problem / Deliverable | 状态 | 证据（代码/结果/文档） |
|---|---|---|
| `unicode1` (a)(b)(c) 三个一句话回答 | 已完成 | `writeup.md` 2.1 |
| `unicode2` (a)(b)(c) 三个回答 | 已完成 | `writeup.md` 2.2 |
| `train_bpe`：实现训练函数 | 已完成 | `cs336_basics/bpe.py`，`tests/test_train_bpe.py` |
| `train_bpe_tinystories` (a) | 已完成 | `artifacts/bpe/tinystories/*`，`writeup.md` 2.3 |
| `train_bpe_tinystories` (b) | 已完成 | `writeup.md` 2.3，`logs/progress_journal.md` |
| `train_bpe_expts_owt` (a) | 已完成 | `artifacts/bpe/owt/*`，`writeup.md` 2.3 |
| `train_bpe_expts_owt` (b) | 已完成 | `writeup.md` 2.3 |
| `tokenizer`：实现 Tokenizer 类 | 已完成 | `cs336_basics/tokenizer.py`，`tests/test_tokenizer.py` |
| `tokenizer_experiments` (a) | 已完成 | `artifacts/experiments/*tokenizer_experiments*.json/.md`，`writeup.md` 2.4 |
| `tokenizer_experiments` (b) | 已完成 | 同上 |
| `tokenizer_experiments` (c) | 已完成 | 同上（含吞吐与 Pile 估算） |
| `tokenizer_experiments` (d) | 已完成 | `scripts/tokenize_corpus.py`，`artifacts/tokenized/*.meta.json`，`writeup.md` 2.4 |
| `linear` | 已完成 | `cs336_basics/nn.py`，`tests/test_nn.py` |
| `embedding` | 已完成 | `cs336_basics/nn.py`，`tests/test_nn.py` |
| `rmsnorm` | 已完成 | `cs336_basics/nn.py`，`tests/test_nn.py` |
| `positionwise_feedforward` (SwiGLU) | 已完成 | `cs336_basics/nn.py`，`tests/test_nn.py` |
| `rope` | 已完成 | `cs336_basics/attention.py`，`tests/test_attention.py` |
| `softmax` | 已完成 | `cs336_basics/nn_utils.py`，`tests/test_nn_utils.py` |
| `scaled_dot_product_attention` | 已完成 | `cs336_basics/attention.py`，`tests/test_attention.py` |
| `causal MHA` | 已完成 | `cs336_basics/attention.py`，`tests/test_attention.py` |
| `transformer_block` | 已完成 | `cs336_basics/transformer.py`，`tests/test_transformer.py` |
| `transformer_lm` | 已完成 | `cs336_basics/transformer.py`，`tests/test_transformer.py` |
| `transformer_accounting` (a)(b)(c)(d)(e) | 已完成 | `writeup.md` 2.6 |
| `cross_entropy` | 已完成 | `cs336_basics/nn_utils.py`，`tests/test_nn_utils.py` |
| `learning_rate_tuning` (toy SGD) | 已完成 | `writeup.md` 2.5 |
| `adamw`：实现优化器 | 已完成 | `cs336_basics/optim.py`，`tests/test_optimizer.py` |
| `adamwAccounting` (a)(b)(c)(d) | 已完成 | `writeup.md` 2.7 |
| `data-loading` (`get_batch`) | 已完成 | `cs336_basics/data.py`，`tests/test_data.py` |
| `training_together`：训练脚本 | 已完成 | `scripts/train_lm.py` |
| `decoding`：解码/采样 | 已完成 | `scripts/train_lm.py` 中 `generate_text` |
| `experiment_log` | 已完成 | `logs/progress_journal.md`，`slurm_logs/train_lm_*.out` |
| `learning_rate`：多学习率曲线+策略 | 已完成 | `artifacts/experiments/lm/lr_sweep_summary.*`，`artifacts/figures/tinystories_lr_sweep_*.png`，`writeup.md` 4.3 |
| `learning_rate`：TinyStories val loss ≤ 1.45 | 已完成 | `artifacts/experiments/lm/ts_main_lr1e3_20k/best_metrics.json`（1.373） |
| `learning_rate`：稳定边界/发散分析 | 已完成 | `writeup.md` 4.3 与 5.1 |
| `batch_size`：不同 batch 曲线 | 已完成 | `artifacts/experiments/lm/batch_size_summary.*`，`artifacts/figures/tinystories_batch_size_val.png` |
| `batch_size`：结论分析 | 已完成 | `writeup.md` 4.5 与 5 |
| `generate`：TinyStories 文本生成与评述 | 已完成 | `artifacts/experiments/lm/ts_main_lr1e3_20k/generated.txt`，`writeup.md` 4.6 |
| `layer_norm_ablation`：去 RMSNorm 曲线 | 已完成 | `artifacts/experiments/lm/ablation_summary.*`，`artifacts/figures/tinystories_rmsnorm_ablation_val.png` |
| `layer_norm_ablation`：影响分析 | 已完成 | `writeup.md` 4.4 与 5.3 |
| `pre_norm_ablation`：post-norm 对比曲线 | 已完成 | `artifacts/figures/tinystories_ablations_val.png` |
| `no_pos_emb`：RoPE vs NoPE 曲线 | 已完成 | `artifacts/figures/tinystories_ablations_val.png`，`ablation_summary.*` |
| `swiglu_ablation`：SwiGLU vs SiLU 曲线 | 已完成 | `artifacts/figures/tinystories_ablations_val.png`，`ablation_summary.*` |
| `mainExperiment`：OWT 学习曲线与损失解释 | 已完成 | `artifacts/experiments/lm/owt_main_lr1e3_20k/*`，`artifacts/figures/ts_vs_owt_main_val.png`，`writeup.md` 4.5/5.2 |
| `mainExperiment`：OWT 生成文本与分析 | 已完成 | `artifacts/experiments/lm/owt_main_lr1e3_20k/generated.txt`，`writeup.md` 4.6/5.2 |
| `leaderboard`（可选） | 可选未做 | `assignment1_checklist.md` E 节 |

补充：
- 统一测试状态：`47 passed, 1 xpassed`（lfs 环境）。
- 大体积 checkpoint/bin 已从版本管理排除，轻量结果与日志已纳入仓库。
