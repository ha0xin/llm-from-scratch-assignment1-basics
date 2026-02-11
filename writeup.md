# CS336 Assignment 1 Writeup（中文）

> 版本时间：待实验完成后自动更新  
> 代码仓库：`ha0xin/llm-from-scratch-assignment1-basics`  
> 远程实验环境：`lfs-dev`（8x RTX 5090，共享）

## 0. 复现实验环境

- 代码分支：`main`
- 依赖管理：`uv sync`
- 测试执行环境：**仅在 lfs 开发机执行**
- 数据目录：`/data/share/hw1-data`
- Slurm 约束：
  - 分区：`lfs-dev-gpu`
  - 时限：`12:00:00`
  - 作业内存：`<= 200GB`

## 1. 代码实现与测试状态

### 1.1 单元测试结果（lfs）

- 运行命令：`uv run pytest -q`
- 结果：`47 passed, 1 xpassed`
- 说明：当前仓库实现已通过课程提供测试点（包含 tokenizer、BPE、Transformer、optimizer、checkpoint 等）。

### 1.2 关键实现模块

- `cs336_basics/bpe.py`：BPE 训练（含 special token 边界处理与并行预分词）
- `cs336_basics/tokenizer.py`：BPE tokenizer 编解码与流式编码
- `cs336_basics/nn.py`：Linear/Embedding/RMSNorm/SwiGLU
- `cs336_basics/attention.py`：RoPE、scaled dot-product attention、MHA
- `cs336_basics/transformer.py`：Transformer block 与 LM 前向
- `cs336_basics/nn_utils.py`：softmax/cross-entropy/gradient clipping
- `cs336_basics/optim.py`：AdamW + cosine LR schedule
- `cs336_basics/data.py`：batch 采样
- `cs336_basics/checkpoint.py`：checkpoint 保存与恢复

## 2. 书面题回答

> 以下答案基于 handout 要求，用中文给出。若课程要求英文版本，可据此翻译。

### 2.1 `unicode1`

1. `chr(0)` 返回 Unicode 的空字符（NUL，`U+0000`）。
2. `repr(chr(0))` 显示转义形式（如 `'\x00'`），而 `print(chr(0))` 通常不可见、不显示字符实体。
3. 该字符可嵌入字符串但通常不可见，很多文本处理/终端显示会把它当控制字符处理。

### 2.2 `unicode2`

1. UTF-8 对英文与常见网络文本更紧凑、生态最广、与字节级 tokenizer 天然匹配，通常比 UTF-16/UTF-32 更节省存储与 I/O。
2. 错误函数按“单字节”逐个解码 UTF-8，会把多字节字符拆坏；例如 `b'\xe7\x89\x9b'`（“牛”）逐字节解码会失败或产生错误结果，因为 UTF-8 需要按完整字节序列解码。
3. 示例 `b'\xc0\xaf'` 不是合法 UTF-8（过长编码），不能解码为有效 Unicode 字符。

## 3. BPE 与 Tokenizer 实验

### 3.1 TinyStories BPE（10k）

- 产物路径：
  - `artifacts/bpe/tinystories/vocab.jsonl`
  - `artifacts/bpe/tinystories/merges.jsonl`
  - `artifacts/bpe/tinystories/meta.json`
- 状态：已完成（见仓库已有产物）

### 3.2 OWT BPE（32k）

- 产物路径：
  - `artifacts/bpe/owt/vocab.jsonl`
  - `artifacts/bpe/owt/merges.jsonl`
  - `artifacts/bpe/owt/meta.json`
- 状态：已完成（见仓库已有产物）

### 3.3 Tokenizer Experiments（已完成）

- 结果文件：
  - `artifacts/experiments/tokenizer_experiments.json`
  - `artifacts/experiments/tokenizer_experiments.md`
- 说明：**仅使用自研 `cs336_basics/tokenizer.py`，未使用外部 tokenizer 库做实验编码。**
- 关键结果（lfs 实测）：
  - TinyStories tokenizer 在 TinyStories 采样文档压缩率：`4.248 bytes/token`
  - OWT tokenizer 在 OWT 采样文档压缩率：`4.682 bytes/token`
  - OWT valid（32MiB 样本）吞吐：`7.60e6 bytes/sec`（约 `7.25 MiB/s`）
  - TinyStories valid（全量）吞吐：`1.14e7 bytes/sec`（约 `10.90 MiB/s`）

### 3.4 Tokenization 落盘吞吐调优（进行中）

- 脚本：`scripts/tokenize_corpus.py`
- 已尝试：
  - 并发预取读（线程）
  - 可调 `chunk_size`
  - `np.array(dtype=np.uint16)` 直接落盘
- `lfs` 对比结果（MiB/s）：
  - TinyStories valid：`old(4MB, no-prefetch)=9.373`，`opt16=9.521`，`opt64=9.678`
  - OWT valid：`old(4MB, no-prefetch)=6.777`，`opt16=6.567`，`opt64=6.459`
- 结论：大块 + 预取对 TinyStories 有小幅收益，但对 OWT 回退；默认参数保持 `chunk=4MB`、`prefetch=0`，并保留可调参数用于按数据集调参。

## 4. LM 训练实验（进行中）

### 4.1 训练基础设施新增

- `scripts/tokenize_corpus.py`：单遍流式 tokenization 到 `uint16` 二进制，附带 `meta.json`
- `scripts/train_lm.py`：可配置 LM 训练脚本（支持 ablation 开关与文本生成）
- `scripts/plot_curves.py`：实验曲线绘图脚本
- `scripts/make_token_subset.py`：大语料 token 子集切分
- `scripts/slurm_tokenize.sh`：tokenization Slurm 封装
- `scripts/slurm_train_lm.sh`：训练 Slurm 封装

### 4.2 当前执行进度

- 已在 `lfs` 提交 tokenization 作业：
  - TinyStories train/valid
  - OWT train/valid
- 当前状态：4 个 tokenized 二进制均已完成并回传本地
  - `artifacts/tokenized/tinystories_train.bin`
  - `artifacts/tokenized/tinystories_valid.bin`
  - `artifacts/tokenized/owt_train.bin`
  - `artifacts/tokenized/owt_valid.bin`
- 作业日志与原始输出：
  - `slurm_logs/tokenize_*.out`
  - `slurm_logs/tokenize_*.err`
- 进度记录：
  - `logs/progress_journal.md`

### 4.3 TinyStories 学习率实验（已完成第一轮）

- 结果汇总文件：
  - `artifacts/experiments/lm/lr_sweep_summary.json`
  - `artifacts/experiments/lm/lr_sweep_summary.md`
  - `artifacts/figures/tinystories_lr_sweep_val.png`
  - `artifacts/figures/tinystories_lr_sweep_train.png`
- 第一轮结论：
  - 在本轮 sweep 中，`lr=1e-3` 表现最好（`best_val_loss=1.5593`，`5k` steps）。
  - `lr=1e-4`、`3e-4` 收敛更慢；`1e-2` 及以上进入高损失平台，明显不如 `1e-3`。
  - `3e-2` 和 `1e-1` 未出现 NaN，但训练质量显著退化，可视为超过稳定高效区间。
- 已据此提交 TinyStories 主实验：
  - run: `ts_main_lr1e3_20k`
  - 目标 tokens：`64 * 20000 * 256 = 327,680,000`
  - 目标：达到课程建议的 TinyStories 收敛区间并用于生成样例。

### 4.4 TinyStories 消融实验（已完成第一轮）

- 汇总文件：
  - `artifacts/experiments/lm/ablation_summary.json`
  - `artifacts/experiments/lm/ablation_summary.md`
  - `artifacts/figures/tinystories_ablations_val.png`
  - `artifacts/figures/tinystories_rmsnorm_ablation_val.png`
- 以 `5k steps` 对比（与 baseline `ts_lr_1e3`）：
  - baseline（pre-norm + RMSNorm + RoPE + SwiGLU）：`1.5593`
  - 去 RMSNorm（lr=1e-3）：`1.5773`
  - 去 RMSNorm（lr=3e-4）：`1.7679`（显著更差）
  - post-norm：`1.5553`（本轮短程与 baseline 接近）
  - NoPE（去 RoPE）：`1.6537`（明显变差）
  - SiLU（替换 SwiGLU）：`1.6110`（变差）

### 4.5 主实验与 Batch Size 实验（已完成）

- TinyStories 主实验：
  - run: `artifacts/experiments/lm/ts_main_lr1e3_20k`
  - 结果：`best_val_loss=1.3730`（`iter=19999`，`tokens_seen=327,680,000`）
  - 曲线可视化建议路径：`artifacts/figures/ts_vs_owt_main_val.png`
- OWT 主实验：
  - run: `artifacts/experiments/lm/owt_main_lr1e3_20k`
  - 结果：`best_val_loss=4.0370`（`iter=19500`，`tokens_seen=327,680,000`）
- Batch size（同为 5k steps）：
  - 汇总：`artifacts/experiments/lm/batch_size_summary.md`
  - 曲线：`artifacts/figures/tinystories_batch_size_val.png`
- Batch size（token-matched，约 81.92M tokens）：
  - 汇总：`artifacts/experiments/lm/batch_size_token_matched_summary.md`
  - 曲线：`artifacts/figures/tinystories_batch_size_token_matched_val.png`

### 4.6 生成样例（节选）

- TinyStories（`ts_main_lr1e3_20k`，prompt=`Once upon a time`）：生成文本结构完整、语法自然，能形成短故事闭环；原文见 `artifacts/experiments/lm/ts_main_lr1e3_20k/generated.txt`。
- OWT（`owt_main_lr1e3_20k`，prompt=`The history of machine learning`）：句法流畅但内容重复、论述跳跃，体现小模型在复杂开放域数据上的语义一致性不足；原文见 `artifacts/experiments/lm/owt_main_lr1e3_20k/generated.txt`。

## 5. 结果分析

### 5.1 TinyStories 训练行为

- `lr=1e-3` 在本实验配置下最有效，`5k` steps 即可到 `~1.56`，`20k` steps 可进一步到 `1.373`。
- 从曲线看，`10k` steps 后收益变缓，但继续训练仍有稳定下降空间。

### 5.2 OWT 与 TinyStories 损失差异

- 在相同模型规模与 token budget（`327.68M`）下，OWT 的 `val_loss` 明显高于 TinyStories（`4.037` vs `1.373`）。
- 原因是 OWT 语料分布更复杂、词汇与主题多样性更高，小模型容量不足以在该预算下达到与 TinyStories 同等拟合度。

### 5.3 各消融项影响

- 去 RoPE 与改用 SiLU 都带来稳定退化（相对 baseline）。
- 去 RMSNorm 在 `lr=1e-3` 下仅小幅退化，但降低学习率后显著变差，说明其稳定性与学习率耦合明显。
- post-norm 在本次 `5k` steps 窗口与 baseline 接近，但长期稳定性仍建议在更长训练中复核。

## 6. 结论

- 仅用自研 tokenizer 与课程实现，已完成从 BPE 到 LM 训练全链路，并在 TinyStories 达到 `val_loss <= 1.45` 目标（最终 `1.373`）。
- 推荐 TinyStories 基线配置：`d_model=512, n_layers=4, n_heads=16, d_ff=1344, batch=64, lr=1e-3`。
- 对 OWT，当前模型与预算下已能生成可读文本，但语义一致性与信息密度仍不足；进一步提升需要更大模型、更多训练步数或更细致的超参搜索。
