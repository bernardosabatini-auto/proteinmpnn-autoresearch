#!/bin/bash
#SBATCH --job-name=mpnn_prod
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=72:00:00
#SBATCH --output=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/prod_%j.out
#SBATCH --error=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/prod_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bsabatini@fas.harvard.edu

# =============================================================================
# Multi-node ProteinMPNN production training (default 2 nodes x 4 H100s).
#
# Launch: srun spawns one Python process per GPU (--ntasks-per-node=4).
# Each task reads SLURM_PROCID/SLURM_LOCALID/SLURM_NTASKS and calls
# torch.distributed.init_process_group(backend="nccl") with env:// init,
# bootstrapped from MASTER_ADDR/MASTER_PORT exported below.
#
# Self-resubmitting: each job runs for the SLURM time limit (default 3 days),
# then resubmits itself with PREVIOUS_CHECKPOINT pointing at epoch_last.pt.
# Caps at MAX_SUBMITS total submissions (default 3 → ~9 days at 3 days/job).
#
# Required env vars (set by submit wrapper or caller):
#   EXP_NAME              experiment name (controls OUT_DIR)
#   DATA_DIR              dataset path (cache OR raw pdb_2021aug02 layout)
# Optional env vars (with defaults):
#   TRAINING_SCRIPT       default training_attn.py
#   NUM_EPOCHS            default 200
#   NUM_EXAMPLES          default 50000
#   BATCH_SIZE            default 10000
#   HIDDEN_DIM            default 128
#   NUM_ENCODER           default 5
#   NUM_DECODER           default 5
#   NUM_NEIGHBORS         default 128
#   BACKBONE_NOISE        default 0.2
#   DROPOUT               default 0.2
#   SAVE_EVERY            default 5
#   RELOAD_EVERY          default 2
#   MAX_PROTEIN_LENGTH    default 10000
#   MAX_SUBMITS           default 3 (this+resubmits combined)
#   SUBMIT_INDEX          internal — incremented on each resubmit
# =============================================================================

set -euo pipefail

echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NODELIST"
echo "Nnodes:        $SLURM_NNODES"
echo "Tasks/node:    $SLURM_NTASKS_PER_NODE"
echo "Total tasks:   $SLURM_NTASKS"
echo "Start:         $(date)"
echo "Experiment:    ${EXP_NAME:?must set EXP_NAME}"
echo "Submit index:  ${SUBMIT_INDEX:=1} of ${MAX_SUBMITS:=3}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load python/3.10.13-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

source /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TORCHELASTIC_ERROR_FILE="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/logs/prod_${SLURM_JOB_ID}_error.json"

# Distributed bootstrap (read by torch.distributed env:// init).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Derive a per-job port to avoid clashes with concurrent jobs on the same node.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"

# NCCL: pin to FASRC InfiniBand fabric. Prior smoke test confirmed ib0 +
# mlx5_{2,3,4,5} HCAs are present and reachable. Without explicit pinning
# NCCL can pick up stale Ethernet routes and hang on the first allreduce
# even after a successful bootstrap.
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stream-mode pre-featurization can take 30+ minutes loading per-chain .pt
# files from a parallel filesystem before any collective is called. PyTorch
# has TWO independent watchdogs that can fire on a "long quiet period":
#   1. Per-collective timeout (default 10 min) — also set via
#      init_process_group(timeout=1h) in training_attn.py.
#   2. Heartbeat monitor (default 480s) — fires when the watchdog itself
#      has been idle too long, even if no collective is in flight. Bumped
#      here so it tolerates the slow first-epoch load.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE="/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"
SCRATCH="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn"
REPO="$BASE/repo"
SCRIPT_PATH="$REPO/slurm/train_production.sh"

DATA_DIR="${DATA_DIR:?must set DATA_DIR}"
OUT_DIR="$SCRATCH/outputs/${EXP_NAME}"
mkdir -p "$OUT_DIR"
mkdir -p "$SCRATCH/logs"

# Resume from epoch_last.pt if it exists, otherwise from explicit
# PREVIOUS_CHECKPOINT, otherwise from scratch.
LAST_CKPT="$OUT_DIR/model_weights/epoch_last.pt"
if [ -f "$LAST_CKPT" ]; then
    RESUME_CKPT="$LAST_CKPT"
elif [ -n "${PREVIOUS_CHECKPOINT:-}" ]; then
    RESUME_CKPT="$PREVIOUS_CHECKPOINT"
else
    RESUME_CKPT=""
fi
echo "Resume from:   ${RESUME_CKPT:-<scratch>}"

# Add repo to PYTHONPATH for module imports
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO/training"

# Touch a "done" sentinel only if training exits cleanly (i.e. reached
# num_epochs). SLURM SIGTERM/SIGKILL on the time limit produces a non-zero
# srun exit and leaves the sentinel absent → trigger a resubmit.
DONE_SENTINEL="$OUT_DIR/.training_done"
rm -f "$DONE_SENTINEL"

# ---------------------------------------------------------------------------
# Launch via srun: one task per GPU, env:// bootstrap from MASTER_*.
# Wrapped in `set +e` so a non-zero exit (e.g. SLURM time-limit kill)
# doesn't trip the script's `set -e`; we decide what to do with TRAIN_RC.
# ---------------------------------------------------------------------------
set +e
srun --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    python -u "${TRAINING_SCRIPT:-training_attn.py}" \
        --path_for_training_data    "$DATA_DIR" \
        --path_for_outputs          "$OUT_DIR" \
        --num_epochs                "${NUM_EPOCHS:-200}" \
        --num_examples_per_epoch    "${NUM_EXAMPLES:-50000}" \
        --batch_size                "${BATCH_SIZE:-10000}" \
        --hidden_dim                "${HIDDEN_DIM:-128}" \
        --num_encoder_layers        "${NUM_ENCODER:-5}" \
        --num_decoder_layers        "${NUM_DECODER:-5}" \
        --num_neighbors             "${NUM_NEIGHBORS:-128}" \
        --backbone_noise            "${BACKBONE_NOISE:-0.2}" \
        --dropout                   "${DROPOUT:-0.2}" \
        --mixed_precision           True \
        --save_model_every_n_epochs "${SAVE_EVERY:-5}" \
        --reload_data_every_n_epochs "${RELOAD_EVERY:-2}" \
        --max_protein_length        "${MAX_PROTEIN_LENGTH:-10000}" \
        --lr_scale                  "${LR_SCALE:-1.0}" \
        --previous_checkpoint       "$RESUME_CKPT"
TRAIN_RC=$?
set -e

if [ "$TRAIN_RC" -eq 0 ]; then
    touch "$DONE_SENTINEL"
fi

echo "srun exit:     $TRAIN_RC"
echo "Done:          $(date)"

if [ -f "$OUT_DIR/log.txt" ]; then
    echo "=== last 5 log lines ==="
    tail -5 "$OUT_DIR/log.txt"
fi

# ---------------------------------------------------------------------------
# Self-resubmit policy
#   - If training exited cleanly (sentinel exists): chain done, regardless
#     of submit index.
#   - Else (SLURM killed us at the time limit, or any other interruption):
#     resubmit from epoch_last.pt, up to MAX_SUBMITS total submissions.
# ---------------------------------------------------------------------------
NEXT_INDEX=$((SUBMIT_INDEX + 1))

if [ -f "$DONE_SENTINEL" ]; then
    echo "Clean exit — training reached num_epochs. Chain done."
    exit 0
fi

if [ "$SUBMIT_INDEX" -ge "$MAX_SUBMITS" ]; then
    echo "Reached MAX_SUBMITS=$MAX_SUBMITS — chain complete, not resubmitting."
    exit 0
fi

if [ ! -f "$OUT_DIR/model_weights/epoch_last.pt" ]; then
    echo "No epoch_last.pt found — not resubmitting (would restart from scratch)."
    exit 1
fi

echo "Resubmitting chain step $NEXT_INDEX of $MAX_SUBMITS..."
sbatch \
    --export=ALL,SUBMIT_INDEX="$NEXT_INDEX" \
    "$SCRIPT_PATH"
