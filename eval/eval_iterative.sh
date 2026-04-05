#!/bin/bash
#SBATCH --job-name=eval_iterative
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/eval_iterative_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/eval_iterative_%j.err

set -euo pipefail

BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

source "$BASE/venv/bin/activate"

export PYTHONUNBUFFERED=1

cd "$BASE/repo"

python eval/eval_iterative.py \
    --checkpoint vanilla_model_weights/v_48_020.pt \
    --valid_data /n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5/valid.pt \
    --num_proteins 500 \
    --temperature 0.1 \
    --ar_samples 4 \
    --max_length 1000

echo "Done: $(date)"
