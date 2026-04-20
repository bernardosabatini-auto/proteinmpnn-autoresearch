#!/bin/bash
# Relaunch script for prod_k192_e5d5_lrdecay03. Used by auto_monitor.sh on
# recoverable failures (OOM / NCCL timeout). Prints the new JobID on stdout.
set -euo pipefail
MEM_OVERRIDE="${MEM_OVERRIDE:-}"   # e.g. "--mem=1200G" when bumping after OOM
EXTRA="${EXTRA:-}"                  # additional sbatch flags
cd /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/repo
sbatch --parsable \
  --job-name=k192_lr3 \
  ${MEM_OVERRIDE} ${EXTRA} \
  --export=ALL,EXP_NAME=prod_k192_e5d5_lrdecay03,DATA_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/pdb_2021aug02_full_cache,TRAINING_SCRIPT=training_attn.py,NUM_NEIGHBORS=192,HIDDEN_DIM=128,NUM_ENCODER=5,NUM_DECODER=5,DROPOUT=0.2,BACKBONE_NOISE=0.2,LR_SCALE=0.3,NUM_EPOCHS=400,NUM_EXAMPLES=50000,BATCH_SIZE=10000,SAVE_EVERY=5,RELOAD_EVERY=2,MAX_SUBMITS=2,PREVIOUS_CHECKPOINT=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs/prod_k192_e5d5_4gpu_halfbatch2/model_weights/epoch230_step124637.pt \
  slurm/train_production.sh
