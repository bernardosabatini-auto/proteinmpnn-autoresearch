#!/bin/bash
#SBATCH --job-name=mpnn_prepfull
#SBATCH --partition=kempner
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=23
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/prepfull_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/prepfull_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bsabatini@fas.harvard.edu

# =============================================================================
# Build a fully-enumerated cache from pdb_2021aug02 (all (cluster,chain) pairs).
# CPU-only job — no GPU needed. 32 CPUs feed the DataLoader workers; ~300G mem
# is enough headroom to hold the assembled pdb_dict_list before torch.save.
# =============================================================================

set -euo pipefail

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

source /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"
SCRATCH="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn"
REPO="$BASE/repo"

DATA_DIR="${DATA_DIR:-$SCRATCH/data/pdb_2021aug02}"
OUT_DIR="${OUT_DIR:-$SCRATCH/data/pdb_2021aug02_full_cache}"
RESCUT="${RESCUT:-3.5}"
MAX_LENGTH="${MAX_LENGTH:-10000}"
NUM_WORKERS="${NUM_WORKERS:-20}"

echo "DATA_DIR:  $DATA_DIR"
echo "OUT_DIR:   $OUT_DIR"
echo "RESCUT:    $RESCUT"
echo "WORKERS:   $NUM_WORKERS"

cd "$REPO"

python training/preprocess_full.py \
    --data_dir    "$DATA_DIR" \
    --out_dir     "$OUT_DIR" \
    --rescut      "$RESCUT" \
    --max_length  "$MAX_LENGTH" \
    --num_workers "$NUM_WORKERS"

echo "Done:      $(date)"
echo "Output:    $OUT_DIR"
ls -lh "$OUT_DIR"
