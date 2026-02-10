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
- 作业日志与原始输出：
  - `slurm_logs/tokenize_*.out`
  - `slurm_logs/tokenize_*.err`
- 进度记录：
  - `logs/progress_journal.md`

### 4.3 待回填实验结果

- TinyStories 主实验学习曲线（目标验证损失）
- Learning rate sweep（含至少一次发散）
- Batch size 对比
- 生成文本样例（>= 256 token）
- 消融实验：
  - 去 RMSNorm
  - post-norm
  - NoPE（去 RoPE）
  - SwiGLU vs SiLU
- OWT 主实验学习曲线与生成文本

## 5. 结果分析（待回填）

### 5.1 TinyStories 训练行为

- 待回填。

### 5.2 OWT 与 TinyStories 损失差异

- 待回填。

### 5.3 各消融项影响

- 待回填。

## 6. 结论（待回填）

- 待在全部实验跑完后给出最终结论与建议配置。
