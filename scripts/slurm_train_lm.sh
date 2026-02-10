#!/bin/bash
#SBATCH --job-name=train_lm
#SBATCH --partition=lfs-dev-gpu
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/train_lm_%j.out
#SBATCH --error=slurm_logs/train_lm_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_DIR" ]; then
  REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$REPO_DIR"
mkdir -p slurm_logs logs

export PYTHONUNBUFFERED=1
echo "[train] host=$(hostname) date=$(date -Iseconds)"
echo "[train] cuda=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 2 | tr '\n' ';')"
echo "[train] args: $*"

uv run python -u scripts/train_lm.py "$@"
