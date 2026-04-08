#!/bin/bash
#SBATCH --job-name=mpnn_eval_length
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/%j.err

set -euo pipefail

BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"
module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
source "$BASE/venv/bin/activate"
export PYTHONUNBUFFERED=1

cd "$BASE/repo"

K96_BEST="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/attn_e5d5_k96_h128_dropout02_resume/model_weights/best_epoch130_valid0543.pt"

# vanilla baseline (k=48, e3d3, no attention)
echo "================================================================"
echo "1. vanilla v_48_020.pt (k=48 baseline)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint vanilla_model_weights/v_48_020.pt \
    --model vanilla \
    --num_encoder_layers 3 --num_decoder_layers 3 \
    --k_neighbors 48

echo ""
echo "================================================================"
echo "2. attn_e5d5_k96 best checkpoint (best_epoch130_valid0543.pt)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$K96_BEST" \
    --model attn \
    --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 96

# k=128 optional (will only run if checkpoint passed in)
if [ -n "${K128_CKPT:-}" ] && [ -f "$K128_CKPT" ]; then
  echo ""
  echo "================================================================"
  echo "3. attn_e5d5_k128 checkpoint ($K128_CKPT)"
  echo "================================================================"
  python eval/eval_length_recovery.py \
      --checkpoint "$K128_CKPT" \
      --model attn \
      --num_encoder_layers 5 --num_decoder_layers 5 \
      --k_neighbors 128
fi

echo ""
echo "Done."
