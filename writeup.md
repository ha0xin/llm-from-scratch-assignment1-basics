# CS336 Assignment 1 作业文档（中文）

> 版本：2026-02-11  
> 仓库：`ha0xin/llm-from-scratch-assignment1-basics`  
> 运行环境：`lfs-dev`（8x RTX 5090，共享）

## 0. 实验与复现环境

- 依赖管理：`uv sync`
- 数据目录：`/data/share/hw1-data`
- Slurm 约束：`12h` 时限、`<=200GB` 内存
- 测试结果（lfs）：`47 passed, 1 xpassed`

本作业实现包含：BPE 训练、Tokenizer、Transformer 组件、训练与解码、实验脚本与日志。

## 1. Unicode 与 BPE 基础题

### 1.1 `unicode1`

1. `chr(0)` 返回 Unicode NUL 字符（`U+0000`）。
2. `repr(chr(0))` 显示为转义形式（如 `"\\x00"`），`print(chr(0))` 基本不可见。
3. 在文本中它通常作为不可见控制字符存在，很多显示/处理流程会忽略或特殊处理它。

### 1.2 `unicode2`

1. UTF-8 相比 UTF-16/UTF-32 对英文与网页文本更紧凑，生态兼容性也最好，适合字节级 tokenizer。
2. 错误函数逐字节解码 UTF-8，会把多字节字符拆开。例：`b'\xe7\x89\x9b'`（“牛”）会失败或得到错误结果。
3. 不能解码成合法 Unicode 的两字节例子：`b'\xc0\xaf'`（过长编码，非法 UTF-8）。

### 1.3 `train_bpe_tinystories`

1. TinyStories 10k BPE 训练耗时约 `36.58s`（`0.010h`），峰值内存约 `254.65MB`。
2. 最长 token 为 `' accomplishment'`（15 bytes），语义上合理（高频短语被合并）。
3. 训练瓶颈主要在预分词与 pair 计数更新（regex 扫描 + 统计维护）。

### 1.4 `train_bpe_expts_owt`

1. OWT 32k BPE 训练耗时约 `836.67s`（`13.94min`），峰值内存约 `17.36GB`。
2. 最长 token 是长度 64 的乱码片段（`'ÃÂÃÂ...'`），反映网页语料中的编码噪声/伪影。
3. 对比 TinyStories：OWT tokenizer 词表更大、领域更广，跨主题压缩能力更强。

### 1.5 `tokenizer_experiments`

1. 压缩率（bytes/token）：
   - TinyStories tokenizer on TinyStories：`4.248`
   - OWT tokenizer on OWT：`4.682`
2. OWT 样本用 TinyStories tokenizer 编码：`3.261 bytes/token`，压缩率下降，说明词表规模和语域不匹配会导致更碎分词。
3. 吞吐（仅 tokenizer 编码，不含落盘）：
   - OWT valid 32MiB：`7.60e6 bytes/s`（约 `7.25 MiB/s`）
   - TinyStories valid：`1.14e7 bytes/s`（约 `10.90 MiB/s`）
   - 估算 825GB Pile：约 `1.35` 天
4. `uint16` 合理：词表规模 <= 32k，token id 落在 `0..65535`，可比 `int32` 显著节省存储与 I/O。

---

## 2. Transformer 与训练理论题

说明：以下 FLOPs/参数统计默认按本仓库实现口径（SwiGLU 三矩阵、embedding 与 lm_head 不绑权、无 bias）给出，并在对应小节注明题面简化口径。

### 2.1 `transformer_accounting`

#### (a) GPT-2 XL-shaped 参数量与参数内存

配置：`V=50257, T=1024, L=48, d=1600, d_ff=6400`

- 参数量：
  - `P = 2Vd + L(4d^2 + 3dd_ff + 2d) + d`
  - `P = 2,127,057,600`
- float32 参数内存：`P * 4 bytes ≈ 7.92 GiB`

#### (b) 前向矩阵乘与 FLOPs

对 batch `B`、上下文 `T`：

- Q/K/V 投影：`6BTd^2`
- 注意力 `QK^T` 与 `AV`：`4BT^2d`
- 注意力输出投影：`2BTd^2`
- SwiGLU FFN（三矩阵）：`6BTdd_ff`
- 最终 `lm_head`：`2BTdV`

总 FLOPs：

- `F = L(8BTd^2 + 4BT^2d + 6BTdd_ff) + 2BTdV`

在 `B=1,T=1024` 下，`F ≈ 4.513e12` FLOPs。

#### (c) 哪些部分 FLOPs 最大

在 GPT-2 XL-shaped（`B=1,T=1024`）下：

- FFN 约 `66.9%`（最大）
- Attention 投影约 `22.3%`
- Attention 矩阵乘约 `7.1%`
- LM head 约 `3.6%`

#### (d) GPT-2 small/medium/large/XL 对比（固定 T=1024）

| 模型 | Attention投影 | Attention矩阵乘 | FFN | LM Head |
|---|---:|---:|---:|---:|
| GPT-2 small | 16.58% | 11.06% | 49.75% | 22.61% |
| GPT-2 medium | 19.96% | 9.98% | 59.87% | 10.20% |
| GPT-2 large | 21.40% | 8.56% | 64.20% | 5.84% |
| GPT-2 XL | 22.30% | 7.14% | 66.91% | 3.65% |

结论：模型规模增大时（固定 `T`），FFN 与投影占比上升，`lm_head` 与 `T^2` 注意力占比下降。

#### (e) GPT-2 XL 把 context length 提升到 16384

- 单次前向 FLOPs 约 `1.495e14`，相对 `T=1024` 增加约 `33.13x`。
- 组成占比变化：Attention 矩阵乘约 `55.15%`，成为主导项（`T^2` 项主导）。

### 2.2 `learning_rate_tuning`（toy SGD）

在最小示例（`loss=(weights**2).mean()`，10 steps）中：

- `lr=1`：缓慢下降（`26.27 -> 18.26`）
- `lr=10`：更快下降（`26.27 -> 0.47`）
- `lr=100`：基本不下降（在该二次目标下接近振荡）
- `lr=1000`：快速发散（`2.7e24` 量级）

### 2.3 `adamwAccounting`（按题面简化口径：`d_ff=4d`，FFN 为 `W1-SiLU-W2`）

设 `V,T,L,d,h,B` 分别表示词表、上下文、层数、宽度、头数和 batch。

1. 参数量（无 bias）：
   - `P = 2Vd + L(12d^2 + 2d) + d`
2. 内存分解（float32）：
   - 参数：`M_param = 4P`
   - 梯度：`M_grad = 4P`
   - AdamW 状态（m,v）：`M_opt = 8P`
   - 激活（近似）：`A = L(16BTd + 2BhT^2) + BTd + 2BTV`
   - 激活内存：`M_act = 4A`
   - 峰值近似：`M_total ≈ 16P + 4A`

代入 GPT-2 XL-shaped（`V=50257,T=1024,L=48,d=1600,h=25`）：

- `M_total(B) ≈ 15,517,753,344 * B + 26,168,601,600` bytes
- 即 `M_total(B) ≈ 14.452 * B + 24.371` GiB
- 在 80GiB 显存下，最大 batch size 约 `B=3`。

单步 FLOPs：

- `F_step ≈ 3F_forward + cP`（`c` 为常数，参数更新项次要）

训练时长估算（A100 float32 19.5TFLOP/s，MFU=50%，400k steps，batch=1024）：

- 约 `5115` 天（数量级结论：单卡 float32 训练该规模不现实）。

---

## 3. 训练与实验结果（TinyStories / OWT）

### 3.1 实验日志基础设施（`experiment_log`）

已实现训练日志、曲线、slurm 输出与进度记录：

- 训练日志：`artifacts/experiments/lm/*/metrics.csv`, `metrics.jsonl`
- 汇总：`artifacts/experiments/lm/*summary*`
- 曲线：`artifacts/figures/*.png`
- 运行日志：`slurm_logs/train_lm_*.out`
- 实验日记：`logs/progress_journal.md`

### 3.2 `learning_rate`：学习率扫描与收敛目标

扫描结果（TinyStories，短跑设置）：

| lr | best val loss |
|---:|---:|
| 1e-4 | 2.1700 |
| 3e-4 | 1.7620 |
| 1e-3 | 1.5593 |
| 3e-3 | 1.6799 |
| 1e-2 | 2.5253 |
| 3e-2 | 3.7422 |
| 1e-1 | 4.4049 |

策略：先粗扫（`1e-4` 到 `1e-1`），以验证集 loss 和稳定性选取 `1e-3` 作为主实验学习率。

主实验（`ts_main_lr1e3_20k`）结果：

- `best_val_loss = 1.3730`（`iter=19999`）
- 满足“TinyStories val loss <= 1.45”要求。

关于“稳定边界”：随 lr 增大，收敛先加快后明显恶化；本次未出现 NaN，但在 `3e-2`、`1e-1` 已进入高损失不稳定区，可视为“稳定边界之外”。

### 3.3 `batch_size experimented`

#### 同步数（5k steps）

| batch | lr | best val loss | tokens seen |
|---:|---:|---:|---:|
| 1 | 3e-4 | 2.9371 | 1.28M |
| 32 | 1e-3 | 1.6665 | 40.96M |
| 64 | 1e-3 | 1.5593 | 81.92M |
| 128 | 1.5e-3 | 1.4614 | 163.84M |

#### token-matched（约 81.92M tokens）

| batch | best val loss |
|---:|---:|
| 32 | 1.5761 |
| 64 | 1.5593 |
| 128 | 1.5745 |

结论：在 token 预算一致时，`batch=64` 略优；过小 batch 噪声大、过大 batch 需要更精细调参与更长训练才稳定受益。

### 3.4 `generate`：文本生成

- TinyStories：`artifacts/experiments/lm/ts_main_lr1e3_20k/generated.txt`
  - 编码长度核对为 `169` token，并在 `<|endoftext|>` 处终止（满足“直到首个 EOT”为止的要求）。
  - 流畅度：语法自然，叙事结构完整。
- OWT：`artifacts/experiments/lm/owt_main_lr1e3_20k/generated.txt`
  - 编码长度核对为 `262` token（满足“至少 256 token”要求）。
  - 流畅度：句法可读，但语义重复、论述跳跃。

影响生成质量的主要因素（至少两点）：

1. 训练数据复杂度（OWT 显著高于 TinyStories）。
2. 模型容量与训练预算（17M 级模型 + 固定 token budget 对 OWT 不足）。
3. 解码策略（temperature/top-p）会显著改变流畅度与多样性。

### 3.5 `layer_norm_ablation` / `pre_norm_ablation` / `no_pos_emb` / `swiglu_ablation`

以 TinyStories 5k steps 为对比窗口：

| 设置 | best val loss |
|---|---:|
| baseline（pre-norm + RMSNorm + RoPE + SwiGLU） | 1.5593 |
| 去 RMSNorm（lr=1e-3） | 1.5773 |
| 去 RMSNorm（lr=3e-4） | 1.7679 |
| post-norm | 1.5553 |
| NoPE（去 RoPE） | 1.6537 |
| SiLU（替换 SwiGLU） | 1.6110 |

结论：

- 去 RoPE 与改用 SiLU 都稳定退化。
- 去 RMSNorm 在较优 lr 下小幅退化，但在更保守 lr 下退化明显，说明其与稳定性强相关。
- post-norm 在短程内接近 baseline；是否长期更稳仍需更长训练窗口验证。

### 3.6 `mainExperiment`：OWT 主实验

在与 TinyStories 主实验相同模型规模和训练迭代下：

- TinyStories：`best_val_loss = 1.3730`
- OWT：`best_val_loss = 4.0370`

解释：两者 loss 不可直接做“绝对质量”等价比较；OWT 语言分布更复杂、词汇与主题熵更高，小模型在相同算力预算下拟合不足，因此 OWT loss 显著更高，生成质量也更弱。

---

## 4. 可选项说明

### 4.1 `leaderboard`

本次提交未做 leaderboard（该项在清单中标记为可选未做）。

---

## 5. 交付物索引

- 主文档：`writeup.md`
- 逐题 deliverable 对照：`assignment1_deliverables_summary.md`
- 清单：`assignment1_checklist.md`
- 产物总清单：`deliverables_manifest.md`

如需英文提交，可在此中文版基础上逐节翻译，不改变数值与结论。
