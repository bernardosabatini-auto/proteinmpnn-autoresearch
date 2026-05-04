#!/bin/bash
#SBATCH --job-name=eval_comp
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evalcomp_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/evalcomp_%j.err

# Comprehensive eval for the campaign report:
#   - eval_length_recovery on processed_3p5/valid.pt (single-pass, length-stratified)
#   - eval_4metric (single-pass + iterative confidence-based, per-AA + per-protein)
#
# Models (top 5 + vanilla, all evaluated on processed_3p5/valid.pt for apples-
# to-apples comparison with the published vanilla reference):
#
#   A. vanilla v_48_020 (k=48, e3d3) — published baseline
#   B. attn e5d5 k=128 best_epoch50_valid0544 — best 21k-subset checkpoint
#   C. attn e5d5 k=192 prod_k192_e5d5_4gpu_halfbatch2 ep220 — first full-cache convergence
#   D. attn e5d5 k=256 prod_k256_e5d5_fullcache epoch_last — fullcache k=256
#   E. attn e5d5 k=320 prod_k320_e5d5_fullcache epoch540 — current best (saved peak)
#   F. attn e5d5 k=320 prod_k320_e5d5_fullcache epoch_last (ep627) — latest plateau

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

DATA_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5
VALID_PT=$DATA_DIR/valid.pt

CKPT_VANILLA=vanilla_model_weights/v_48_020.pt
CKPT_K128=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/attn_e5d5_k128_h128_dropout02/model_weights/best_epoch50_valid0544.pt
CKPT_K192=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k192_e5d5_4gpu_halfbatch2/model_weights/epoch220_step119209.pt
CKPT_K256=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k256_e5d5_fullcache/model_weights/epoch_last.pt
CKPT_K320_540=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k320_e5d5_fullcache/model_weights/epoch540_step590124.pt
CKPT_K320_LAST=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k320_e5d5_fullcache/model_weights/epoch_last.pt

# ----------------------------------------------------------------
# Pass 1: length-stratified, single-pass per-protein for k=128 and k=192
#         (vanilla, k=256, k=320 ep540, k=320 ep_last already in
#          length_recovery_results.md from job 9979555 / commit dbd65fc).
# ----------------------------------------------------------------
echo "================================================================"
echo "LENGTH-STRATIFIED (new): attn k=128 (21k subset best, ep50)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$CKPT_K128" \
    --model attn --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 128

echo ""
echo "================================================================"
echo "LENGTH-STRATIFIED (new): attn k=192 (fullcache halfbatch2, ep220)"
echo "================================================================"
python eval/eval_length_recovery.py \
    --checkpoint "$CKPT_K192" \
    --model attn --num_encoder_layers 5 --num_decoder_layers 5 \
    --k_neighbors 192

# ----------------------------------------------------------------
# Pass 2: 4-metric (per-AA and per-protein, single-pass and iterative
#         confidence-based n_rounds=10) on processed_3p5/valid.pt for
#         all 6 models.
# ----------------------------------------------------------------
OUTDIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/eval_4metric
mkdir -p "$OUTDIR"

echo ""
echo "================================================================"
echo "4-METRIC (vanilla v_48_020, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric_vanilla.py \
    --checkpoint "$CKPT_VANILLA" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/vanilla_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "4-METRIC (attn k=128 21k-subset best ep50, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K128" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/k128_ep50_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "4-METRIC (attn k=192 fullcache halfbatch2 ep220, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K192" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/k192_ep220_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "4-METRIC (attn k=256 fullcache ep_last, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K256" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/k256_eplast_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "4-METRIC (attn k=320 fullcache ep540, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K320_540" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/k320_ep540_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "4-METRIC (attn k=320 fullcache ep_last, processed_3p5/valid.pt)"
echo "================================================================"
python eval/eval_4metric.py \
    --checkpoint "$CKPT_K320_LAST" \
    --data_dir "$DATA_DIR" \
    --output "$OUTDIR/k320_eplast_p3p5.tsv" \
    --batch_size 5000 --n_rounds 10

echo ""
echo "================================================================"
echo "Combined 4-metric summary (single number per metric per model):"
echo "================================================================"
for f in "$OUTDIR/vanilla_p3p5.tsv" "$OUTDIR/k128_ep50_p3p5.tsv" \
         "$OUTDIR/k192_ep220_p3p5.tsv" "$OUTDIR/k256_eplast_p3p5.tsv" \
         "$OUTDIR/k320_ep540_p3p5.tsv" "$OUTDIR/k320_eplast_p3p5.tsv"; do
    echo "--- $(basename "$f" .tsv) ---"
    head -1 "$f"
    tail -1 "$f"
done

echo ""
echo "Done: $(date)"
