#!/bin/bash
# Relaunch script for prod_k256_e5d5_fullcache. See sibling k192 script.
set -euo pipefail
MEM_OVERRIDE="${MEM_OVERRIDE:-}"
EXTRA="${EXTRA:-}"
cd /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/repo
sbatch --parsable \
  --job-name=k256_fresh \
  ${MEM_OVERRIDE} ${EXTRA} \
  --export=ALL,EXP_NAME=prod_k256_e5d5_fullcache,DATA_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/pdb_2021aug02_full_cache,TRAINING_SCRIPT=training_attn.py,NUM_NEIGHBORS=256,HIDDEN_DIM=128,NUM_ENCODER=5,NUM_DECODER=5,DROPOUT=0.2,BACKBONE_NOISE=0.2,LR_SCALE=1.0,NUM_EPOCHS=400,NUM_EXAMPLES=50000,BATCH_SIZE=10000,SAVE_EVERY=5,RELOAD_EVERY=2,MAX_SUBMITS=2 \
  slurm/train_production.sh
