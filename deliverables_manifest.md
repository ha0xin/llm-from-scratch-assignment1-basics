# Deliverables Manifest

## 1. 代码与文档

- `writeup.md`：中文实验报告（主文档）
- `assignment1_deliverables_summary.md`：逐题 deliverable 对照汇总
- `assignment1_checklist.md`：作业清单与完成状态
- `deliverables_manifest.md`：本清单

## 2. 原始实验日志（Raw）

- `slurm_logs/*.out`
- `slurm_logs/*.err`
- `artifacts/experiments/**/*.jsonl`（训练迭代日志）
- `artifacts/experiments/**/*.csv`（结构化曲线数据）
- `logs/progress_journal.md`

## 3. 处理后数据（Processed）

- `artifacts/bpe/tinystories/meta.json`
- `artifacts/bpe/owt/meta.json`
- `artifacts/experiments/tokenizer_experiments.json`
- `artifacts/experiments/tokenizer_experiments.md`
- `artifacts/experiments/throughput_valid_32mb_own_only_v4/tokenizer_experiments.json`
- `artifacts/experiments/throughput_tinyvalid_own_only_v3/tokenizer_experiments.json`
- `artifacts/experiments/tokenize_valid_old.meta.json`
- `artifacts/experiments/tokenize_valid_opt16.meta.json`
- `artifacts/experiments/tokenize_valid_opt64.meta.json`
- `artifacts/experiments/tokenize_owt_valid_old.meta.json`
- `artifacts/experiments/tokenize_owt_valid_opt16.meta.json`
- `artifacts/experiments/tokenize_owt_valid_opt64.meta.json`
- `artifacts/tokenized/*.meta.json`
- `artifacts/experiments/lm/*/summary.json`
- `artifacts/experiments/lm/*/best_metrics.json`
- `artifacts/experiments/lm/lr_sweep_summary.json`
- `artifacts/experiments/lm/lr_sweep_summary.md`
- `artifacts/experiments/lm/ablation_summary.json`
- `artifacts/experiments/lm/ablation_summary.md`
- `artifacts/experiments/lm/batch_size_summary.json`
- `artifacts/experiments/lm/batch_size_summary.md`
- `artifacts/experiments/lm/batch_size_token_matched_summary.json`
- `artifacts/experiments/lm/batch_size_token_matched_summary.md`

## 4. 图表与可视化

- `artifacts/figures/*.png`（学习曲线与对比图）

## 5. 模型与检查点

- `artifacts/experiments/lm/*/checkpoint_latest.pt`
- `artifacts/experiments/lm/*/checkpoint_best.pt`
- `artifacts/experiments/lm/*/generated.txt`

## 6. 数据文件

- BPE 词表与合并规则：
  - `artifacts/bpe/tinystories/vocab.jsonl`
  - `artifacts/bpe/tinystories/merges.jsonl`
  - `artifacts/bpe/owt/vocab.jsonl`
  - `artifacts/bpe/owt/merges.jsonl`
- 语料 tokenized 二进制：
  - `artifacts/tokenized/tinystories_train.bin`
  - `artifacts/tokenized/tinystories_valid.bin`
  - `artifacts/tokenized/owt_train.bin`
  - `artifacts/tokenized/owt_valid.bin`

> 注：本文件会在每个实验阶段更新，保证“原始数据 + 处理数据 + 图表 + 结论”可追溯。
