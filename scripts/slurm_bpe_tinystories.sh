#!/bin/bash
#SBATCH --job-name=bpe_tinystories
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/bpe_tinystories_%j.out
#SBATCH --error=slurm_logs/bpe_tinystories_%j.err

set -euo pipefail

# Adjust partition if needed, e.g.: #SBATCH --partition=your_partition
# If your cluster needs a specific account/QOS, add it here.

uv run python -u scripts/train_bpe_tinystories.py \
  --input /data/share/hw1-data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token "<|endoftext|>" \
  --out-dir artifacts/bpe/tinystories \
  --log-every 50
