#!/bin/bash
#SBATCH --job-name=proteinmpnn_exp
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bsabatini@fas.harvard.edu

set -euo pipefail

BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Experiment: ${EXP_NAME:-unknown}"

module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

source "$BASE/venv/bin/activate"

export PYTHONUNBUFFERED=1

REPO_DIR="$BASE/repo"
DATA_DIR="${DATA_DIR:-/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5}"
OUT_DIR="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/${EXP_NAME:-exp_${SLURM_JOB_ID}}"
PREVIOUS_CHECKPOINT="${PREVIOUS_CHECKPOINT:-}"

mkdir -p "$OUT_DIR"
mkdir -p "/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs"

cd "$REPO_DIR"

python training/${TRAINING_SCRIPT:-training.py} \
    --path_for_training_data "$DATA_DIR" \
    --path_for_outputs       "$OUT_DIR" \
    --num_epochs             "${NUM_EPOCHS:-150}" \
    --num_examples_per_epoch "${NUM_EXAMPLES:-50000}" \
    --batch_size             "${BATCH_SIZE:-10000}" \
    --hidden_dim             "${HIDDEN_DIM:-128}" \
    --num_encoder_layers     "${NUM_ENCODER:-3}" \
    --num_decoder_layers     "${NUM_DECODER:-3}" \
    --num_neighbors          "${NUM_NEIGHBORS:-48}" \
    --backbone_noise         "${BACKBONE_NOISE:-0.2}" \
    --mixed_precision        "${MIXED_PRECISION:-True}" \
    --dropout                "${DROPOUT:-0.1}" \
    --save_model_every_n_epochs "${SAVE_EVERY:-10}" \
    --reload_data_every_n_epochs "${RELOAD_EVERY:-200}" \
    --previous_checkpoint    "$PREVIOUS_CHECKPOINT"

echo "Done: $(date)"
echo "=== FINAL RESULTS ==="
tail -5 "$OUT_DIR/log.txt"
grep "valid_acc" "$OUT_DIR/log.txt" | awk -F'valid_acc: ' '{print $2}' | sort -n | tail -1 | \
    awk '{print "best_valid_acc: " $1}'
