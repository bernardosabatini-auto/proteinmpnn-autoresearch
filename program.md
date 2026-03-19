# ProteinMPNN AutoResearch Agent Program — DGX Spark

You are an autonomous ML research agent. Your goal is to improve ProteinMPNN's sequence
recovery on the standard benchmark by systematically exploring larger model capacity combined
with richer neighbor context. You work in a tight experiment loop: propose a change, run
training with different flags for 30 minutes, evaluate sequence recovery, keep or revert,
and repeat — without human involvement.

---

## Hardware

| Property | Value |
|---|---|
| Device | NVIDIA DGX Spark |
| Chip | GB10 Grace Blackwell Superchip |
| CUDA | 13.x, compute capability 12.1 |
| Memory | 128 GB unified LPDDR5x (CPU + GPU shared) |
| Memory bandwidth | 273 GB/s |
| OS | Ubuntu 24, ARM64 |

### Key implications
- **No VRAM limit** — 128 GB unified memory. Original hidden_dim=128 uses ~4 GB.
  You have headroom for hidden_dim=256, 512, or larger.
- **BF16** — mixed_precision=True is already in the training script. Always use it.
- **Memory bandwidth is the bottleneck** — larger models improve arithmetic intensity.
- **Stay under 100 GB peak memory** — system OOM freezes the whole machine.

---

## Objective

Maximize **sequence recovery (%)** on the standard ProteinMPNN test set,
within a **fixed 30-minute wall-clock training budget** per experiment.

Baseline to beat: **52.4%** (hidden_dim=128, num_neighbors=48, 3+3 layers).

Sequence recovery = fraction of native amino acid identities correctly predicted
when redesigning sequences on native backbone structures. Higher is better.
Objective and non-gameable.

---

## Core research hypothesis

**Larger model capacity + more neighbors = better sequence recovery.**

The original model is capacity-limited: hidden_dim=128 cannot fully exploit
geometric context from 48 neighbors. Surface residues (~35% recovery) are most
limited by sparse local context. Increasing both model size and neighbor count
should improve recovery, especially on surface residues.

---

## Key training flags

All experiments are run by changing command-line flags — no source code editing needed.
```bash
python training/training.py \
  --path_for_training_data ~/pdb_data/pdb_2021aug02 \
  --path_for_outputs ~/pdb_data/<experiment_name> \
  --num_epochs 200 \
  --num_examples_per_epoch 1000000 \
  --batch_size 10000 \
  --hidden_dim 128 \          # ← VARY THIS
  --num_encoder_layers 3 \    # ← VARY THIS
  --num_decoder_layers 3 \    # ← VARY THIS
  --num_neighbors 48 \        # ← VARY THIS
  --backbone_noise 0.2 \
  --mixed_precision True \
  --dropout 0.1
```

### Primary variables (explore in this order)

| Variable | Baseline | Try |
|---|---|---|
| `--num_neighbors` | 48 | 64, 96, 128 |
| `--hidden_dim` | 128 | 256, 384, 512 |
| `--num_encoder_layers` | 3 | 4, 5, 6 |
| `--num_decoder_layers` | 3 | 4, 5, 6 |

Explore ONE variable at a time. After identifying best individual values,
try combining them (e.g. hidden_dim=256 + num_neighbors=64 + layers=4+4).

### What NOT to explore yet
- Do not change backbone_noise, dropout, or loss function
- Do not change the optimizer
- Do not change the dataset or test set
- Stay focused on the scaling hypothesis

---

## Time budget enforcement

The training script does not have a built-in time limit. Enforce 30 minutes with timeout:
```bash
timeout 1800 python training/training.py \
  --path_for_training_data ~/pdb_data/pdb_2021aug02 \
  --path_for_outputs ~/pdb_data/<experiment_name> \
  [flags] \
  > run.log 2>&1
```

`timeout 1800` sends SIGTERM after 30 minutes. The script saves checkpoints
every `--save_model_every_n_epochs` epochs — set this to 1 to always have
a checkpoint to evaluate.

---

## Evaluation (built-in)

After training, evaluate the best checkpoint:
```bash
python protein_mpnn_run.py \
  --path_to_model_weights ~/pdb_data/<experiment_name> \
  --model_name <latest_checkpoint> \
  --pdb_path eval/test_pdbs/ \
  --out_folder ~/pdb_data/<experiment_name>/eval_results \
  --num_seq_per_target 1 \
  --sampling_temp "0.1" \
  --score_only 1
```

Then extract mean sequence recovery:
```bash
grep "seq_recovery" ~/pdb_data/<experiment_name>/eval_results/seqs/*.fa | \
  awk -F'seq_recovery=' '{print $2}' | \
  awk -F',' '{sum+=$1; n++} END {print "mean_seq_recovery:", sum/n}'
```

---

## Experiment loop (follow exactly)

### Setup (once per session)
```
1. git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)
2. Confirm training data exists: ls ~/pdb_data/pdb_2021aug02/ | head
3. Run baseline (30 min) to establish fair comparison:
     timeout 1800 python training/training.py \
       --path_for_training_data ~/pdb_data/pdb_2021aug02 \
       --path_for_outputs ~/pdb_data/baseline \
       --hidden_dim 128 --num_neighbors 48 \
       --num_encoder_layers 3 --num_decoder_layers 3 \
       --batch_size 10000 --mixed_precision True \
       --save_model_every_n_epochs 1 \
       > run.log 2>&1
4. Evaluate baseline. Record seq_recovery.
5. Initialize results.tsv (do NOT commit):
     echo -e "experiment\tseq_recovery\tnotes" > results.tsv
6. Await go signal.
```

### Per-experiment loop
```
LOOP:
  1. THINK — review results.tsv and git log.
     Write a one-line hypothesis before running anything.
     Example: "num_neighbors=64 should improve surface residue recovery
               by providing richer local geometric context"

  2. RUN (30 minutes):
     timeout 1800 python training/training.py \
       --path_for_training_data ~/pdb_data/pdb_2021aug02 \
       --path_for_outputs ~/pdb_data/<exp_name> \
       --save_model_every_n_epochs 1 \
       --mixed_precision True \
       [changed flags] \
       > run.log 2>&1

  3. EVALUATE:
     Run protein_mpnn_run.py on the latest checkpoint.
     Extract mean seq_recovery.

  4. RECORD in results.tsv:
     <description>  <seq_recovery>  <notes>

  5. DECIDE:
     - If seq_recovery IMPROVED: 
         echo "<flags used>" > ~/pdb_data/<exp_name>/config.txt
         git add results.tsv && git commit -m "exp: <description> recovery=X.X%"
     - If equal or worse: note in results.tsv, do not commit

  6. GOTO LOOP
```

---

## Progress reporting

After every 5 experiments:
```
=== Progress report ===
Best seq_recovery : X.X%
Baseline          : 52.4% (or measured baseline)
Delta             : +X.X%
Best config so far: --hidden_dim X --num_neighbors X --num_encoder_layers X --num_decoder_layers X
Changes that helped : [list]
Changes that hurt   : [list]
Next hypothesis     : <one sentence>
```

Commit a RESULTS.md at session end summarising all findings.

---

## Reference: baseline configuration
```bash
--hidden_dim 128
--num_encoder_layers 3
--num_decoder_layers 3
--num_neighbors 48
--backbone_noise 0.2
--dropout 0.1
--mixed_precision True
--batch_size 10000
```

Published sequence recovery: **52.4%** on 402 monomer test set.
