#!/bin/bash
#SBATCH --job-name=eval_k256_k320
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evalk_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evalk_%j.err

# 4-metric eval (per-AA + per-protein, single + iterative) for the
# converged k=256 run and the still-climbing k=320 run on the full
# pdb_2021aug02 validation cache. Output rows include both the residue-
# weighted ("per amino acid") and per-protein numbers so we can compare
# to the published ProteinMPNN 0.524 (residue-weighted) baseline.

set -euo pipefail
echo "Job ID:   $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
source /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/venv/bin/activate
export PYTHONUNBUFFERED=1

REPO=/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/repo
DATA=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/pdb_2021aug02_full_cache
OUTDIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/eval_4metric
mkdir -p "$OUTDIR"

cd "$REPO"

CKPT_K256=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k256_e5d5_fullcache/model_weights/epoch_last.pt
CKPT_K320=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k320_e5d5_fullcache/model_weights/epoch_last.pt

echo "===== k256 epoch_last.pt ====="
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K256" \
    --data_dir "$DATA" \
    --output "$OUTDIR/k256_epoch_last.tsv" \
    --batch_size 5000 \
    --n_rounds 10

echo "===== k320 epoch_last.pt ====="
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K320" \
    --data_dir "$DATA" \
    --output "$OUTDIR/k320_epoch_last.tsv" \
    --batch_size 5000 \
    --n_rounds 10

echo "===== combined ====="
head -1 "$OUTDIR/k256_epoch_last.tsv"
echo "k256:"; tail -1 "$OUTDIR/k256_epoch_last.tsv"
echo "k320:"; tail -1 "$OUTDIR/k320_epoch_last.tsv"
echo "Done: $(date)"
