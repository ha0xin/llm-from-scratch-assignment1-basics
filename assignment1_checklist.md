# Assignment 1 清单（完成度跟踪）

> 状态定义：`[x]` 已完成，`[~]` 进行中，`[ ]` 未开始
> 说明：代码测试以 `lfs` 上 `uv run pytest -q` 为准（当前：`47 passed, 1 xpassed`）。

## A. 代码实现题

- [x] `train_bpe`
- [x] `Tokenizer`（encode/decode/encode_iterable/special token）
- [x] `Linear`
- [x] `Embedding`
- [x] `RMSNorm`
- [x] `SwiGLU`
- [x] `RoPE`
- [x] `softmax`
- [x] `scaled_dot_product_attention`
- [x] `multihead_self_attention`（含 causal mask）
- [x] `transformer_block`
- [x] `transformer_lm`
- [x] `cross_entropy`
- [x] `AdamW`
- [x] cosine learning rate schedule
- [x] gradient clipping
- [x] data loading (`get_batch`)
- [x] checkpoint save/load

## B. BPE/Tokenizer 实验

- [x] TinyStories BPE 训练（10k）
- [x] OWT BPE 训练（32k）
- [x] tokenizer_experiments（压缩率、吞吐估计、说明）
- [x] 自研 tokenizer 吞吐优化（仅自研实现，无外部 tokenizer）
- [x] 全量 train/valid 数据 tokenization 到 uint16

## C. LM 实验（TinyStories）

- [x] 基线训练（收敛曲线）
- [x] Learning rate sweep（含高学习率不稳定区间）
- [x] Batch size 对比
- [x] 文本生成样例

## D. LM 消融实验

- [x] 去 RMSNorm
- [x] post-norm vs pre-norm
- [x] NoPE vs RoPE
- [x] SwiGLU vs SiLU

## E. LM 实验（OWT）

- [x] OWT 主实验（学习曲线）
- [x] OWT 文本生成样例
- [ ] （可选）leaderboard 提交材料

## F. 文档与交付

- [x] `writeup.md`（中文）
- [x] `assignment1_deliverables_summary.md`（逐题 deliverable 对照）
- [x] `assignment1_checklist.md`（本文件）
- [x] `deliverables_manifest.md`（原始数据/处理数据/图表/日志清单）
- [x] 原始日志与处理结果归档
