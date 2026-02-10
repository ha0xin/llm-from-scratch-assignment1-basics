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
- [~] 全量 train/valid 数据 tokenization 到 uint16（Slurm 运行中）

## C. LM 实验（TinyStories）

- [ ] 基线训练（收敛曲线）
- [ ] Learning rate sweep（含发散案例）
- [ ] Batch size 对比
- [ ] 文本生成样例

## D. LM 消融实验

- [ ] 去 RMSNorm
- [ ] post-norm vs pre-norm
- [ ] NoPE vs RoPE
- [ ] SwiGLU vs SiLU

## E. LM 实验（OWT）

- [ ] OWT 主实验（学习曲线）
- [ ] OWT 文本生成样例
- [ ] （可选）leaderboard 提交材料

## F. 文档与交付

- [~] `writeup.md`（中文，正在回填实验结果）
- [~] `assignment1_checklist.md`（本文件）
- [~] `deliverables_manifest.md`（原始数据/处理数据/图表/日志清单）
- [~] 原始日志与处理结果归档
