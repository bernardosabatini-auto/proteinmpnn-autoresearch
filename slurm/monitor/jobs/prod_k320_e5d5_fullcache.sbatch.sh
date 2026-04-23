#!/bin/bash
# Relaunch script for prod_k320_e5d5_fullcache. Fresh training of attn_e5d5
# at k=320 on the full 232k cache, launched after k=256 hit 0.575 (big win
# over the prior 0.553 plateau) to test whether k-scaling continues.
# Used by auto_monitor.sh on recoverable failures.
set -euo pipefail
MEM_OVERRIDE="${MEM_OVERRIDE:-}"
EXTRA="${EXTRA:-}"
cd /n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn/repo
sbatch --parsable \
  --job-name=k320_fresh \
  ${MEM_OVERRIDE} ${EXTRA} \
  --export=ALL,EXP_NAME=prod_k320_e5d5_fullcache,DATA_DIR=/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/pdb_2021aug02_full_cache,TRAINING_SCRIPT=training_attn.py,NUM_NEIGHBORS=320,HIDDEN_DIM=128,NUM_ENCODER=5,NUM_DECODER=5,DROPOUT=0.2,BACKBONE_NOISE=0.2,LR_SCALE=1.0,NUM_EPOCHS=400,NUM_EXAMPLES=50000,BATCH_SIZE=10000,SAVE_EVERY=5,RELOAD_EVERY=2,MAX_SUBMITS=2 \
  slurm/train_production.sh
