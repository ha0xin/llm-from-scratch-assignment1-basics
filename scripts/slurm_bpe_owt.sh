#!/bin/bash
#SBATCH --job-name=bpe_owt
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm_logs/bpe_owt_%j.out
#SBATCH --error=slurm_logs/bpe_owt_%j.err

set -euo pipefail

# Adjust partition if needed, e.g.: #SBATCH --partition=your_partition
# If your cluster needs a specific account/QOS, add it here.

export PYTHONUNBUFFERED=1

uv run python -u scripts/train_bpe_owt.py \
  --input /data/share/hw1-data/owt_train.txt \
  --vocab-size 32000 \
  --special-token "<|endoftext|>" \
  --log-every 50 \
  --out-dir artifacts/bpe/owt
