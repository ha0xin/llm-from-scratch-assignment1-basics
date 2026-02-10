#!/bin/bash
#SBATCH --job-name=tokenize_corpus
#SBATCH --partition=lfs-dev-gpu
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm_logs/tokenize_%j.out
#SBATCH --error=slurm_logs/tokenize_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_DIR" ]; then
  REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$REPO_DIR"
mkdir -p slurm_logs logs

echo "[tokenize] host=$(hostname) date=$(date -Iseconds)"
echo "[tokenize] args: $*"

uv run python -u scripts/tokenize_corpus.py "$@"
