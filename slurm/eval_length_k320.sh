#!/bin/bash
#SBATCH --job-name=eval_length_k320
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evallen_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evallen_%j.err

# Length-stratified valid recovery (per-protein PRIMARY + residue-weighted)
# on processed_3p5/valid.pt (1463 proteins). Compares:
#   1. vanilla v_48_020.pt              — published baseline
#   2. attn e5d5 k=256 epoch_last       — fullcache run, 0.578 plateau
#   3. attn e5d5 k=320 epoch540 (0.592) — fullcache run, BEST saved
#   4. attn e5d5 k=320 epoch_last (~ep627, 0.587) — natural endpoint

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"
module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
source "$BASE/venv/bin/activate"
export PYTHONUNBUFFERED=1

cd "$BASE/repo"

CKPT_K256_LAST=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k256_e5d5_fullcache/model_weights/epoch_last.pt
CKPT_K320_540=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k320_e5d5_fullcache/model_weights/epoch540_step590124.pt
CKPT_K320_LAST=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k320_e5d5_fullcache/model_weights/epoch_last.pt

echo "================================================================"
echo "1. vanilla v_48_020.pt (k=48, e3d3)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint vanilla_model_weights/v_48_020.pt \
    --model vanilla \
    --num_encoder_layers 3 --num_decoder_layers 3 \
    --k_neighbors 48

echo ""
echo "================================================================"
echo "2. attn e5d5 k=256 epoch_last (fullcache plateau 0.578)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$CKPT_K256_LAST" \
    --model attn \
    --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 256

echo ""
echo "================================================================"
echo "3. attn e5d5 k=320 epoch540 (fullcache best saved, valid_acc 0.592)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$CKPT_K320_540" \
    --model attn \
    --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 320

echo ""
echo "================================================================"
echo "4. attn e5d5 k=320 epoch_last (ep627, valid_acc 0.587)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$CKPT_K320_LAST" \
    --model attn \
    --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 320

echo ""
echo "Done: $(date)"
