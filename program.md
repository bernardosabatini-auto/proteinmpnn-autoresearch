# ProteinMPNN AutoResearch Agent Program — DGX Spark

You are an autonomous ML research agent. Your goal is to improve ProteinMPNN's sequence
recovery by systematically exploring larger model capacity combined with richer neighbor
context. You work in a tight experiment loop: change one flag, train for 30 minutes,
read valid_acc from the log, keep or record, repeat — without human involvement.

---

## Hardware

| Property | Value |
|---|---|
| Device | NVIDIA DGX Spark |
| Chip | GB10 Grace Blackwell Superchip |
| CUDA | 13.x, compute capability 12.1 |
| Memory | 128 GB unified LPDDR5x (CPU + GPU shared) |
| OS | Ubuntu 24, ARM64 |

### Key implications
- 128 GB unified memory — no VRAM OOM, but system OOM freezes the machine
- Stay under 100 GB peak memory at all times
- Always use --mixed_precision True
- Larger models improve arithmetic intensity on this bandwidth-bound hardware

---

## Objective

Maximize **valid_acc** (validation sequence recovery) printed during training,
within a **fixed 30-minute wall-clock budget** per experiment.

valid_acc is printed each epoch:
  epoch: 1, step: 4, time: 2.7, train: 31.747, valid: 28.309, train_acc: 0.031, valid_acc: 0.032

Your score for an experiment = highest valid_acc reached in 30 minutes.
Baseline to beat: **0.524** (52.4% sequence recovery, original paper).

---

## Core hypothesis

**Larger model + more neighbors = better sequence recovery.**

The original hidden_dim=128 cannot fully exploit geometric context from 48 neighbors.
Surface residues (~35% recovery) are most limited. Scaling both together attacks this directly.

---

## Training command
```bash
timeout 1800 python training/training.py \
  --path_for_training_data ~/pdb_data/pdb_2021aug02 \
  --path_for_outputs ~/pdb_data/<exp_name> \
  --num_epochs 200 \
  --num_examples_per_epoch 50000 \
  --batch_size 10000 \
  --hidden_dim 128 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_neighbors 48 \
  --backbone_noise 0.2 \
  --mixed_precision True \
  --dropout 0.1 \
  --save_model_every_n_epochs 1 \
  --reload_data_every_n_epochs 200 \
  > run.log 2>&1
```

`timeout 1800` enforces the 30-minute budget. Change flags as needed per experiment.

---

## Variables to explore (in priority order)

| Flag | Baseline | Try |
|---|---|---|
| `--num_neighbors` | 48 | 64, 96, 128 |
| `--hidden_dim` | 128 | 256, 384, 512 |
| `--num_encoder_layers` | 3 | 4, 5, 6 |
| `--num_decoder_layers` | 3 | 4, 5, 6 |

One variable changed per experiment. After finding best individual values,
combine them (e.g. hidden_dim=256 + num_neighbors=64 + layers=4+4).

Do NOT change: backbone_noise, dropout, loss function, optimizer, dataset.

---

## Reading results
```bash
# Best valid_acc from a run
grep "valid_acc" run.log | awk -F'valid_acc: ' '{print $2}' | sort -n | tail -1

# Full training curve
grep "valid_acc" run.log
```

---

## Experiment loop

### Setup (once per session)
```
1. git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)
2. Confirm data: ls ~/pdb_data/pdb_2021aug02/ | head -5
3. Run baseline (30 min):
     timeout 1800 python training/training.py \
       --path_for_training_data ~/pdb_data/pdb_2021aug02 \
       --path_for_outputs ~/pdb_data/baseline \
       --hidden_dim 128 --num_neighbors 48 \
       --num_encoder_layers 3 --num_decoder_layers 3 \
       --batch_size 10000 --mixed_precision True \
       --save_model_every_n_epochs 1 \
  --reload_data_every_n_epochs 200 \
       > run.log 2>&1
4. Record baseline valid_acc.
5. Create results.tsv (do NOT commit):
     echo -e "experiment\tvalid_acc\tnotes" > results.tsv
6. Await go signal.
```

### Per-experiment loop
```
LOOP:
  1. THINK — review results.tsv. Write one-line hypothesis.

  2. RUN:
     timeout 1800 python training/training.py \
       --path_for_training_data ~/pdb_data/pdb_2021aug02 \
       --path_for_outputs ~/pdb_data/<exp_name> \
       --save_model_every_n_epochs 1 \
  --reload_data_every_n_epochs 200 \
       --mixed_precision True \
       [one changed flag] \
       > run.log 2>&1

  3. READ:
     grep "valid_acc" run.log | awk -F'valid_acc: ' '{print $2}' | sort -n | tail -1

  4. RECORD in results.tsv:
     <flag change>  <best_valid_acc>  <notes>

  5. DECIDE:
     - If valid_acc IMPROVED:
         git add results.tsv
         git commit -m "exp: <description> valid_acc=X.XXX"
     - If equal or worse: record in results.tsv only

  6. GOTO LOOP
```

---

## Progress report (every 5 experiments)
```
=== Progress report ===
Best valid_acc    : X.XXX
Baseline          : 0.524
Delta             : +X.XXX
Best config       : --hidden_dim X --num_neighbors X --num_encoder_layers X --num_decoder_layers X
Helped            : [list]
Hurt or neutral   : [list]
Next hypothesis   : <one sentence>
```

Commit RESULTS.md at session end.

---

## Baseline reference
```
--hidden_dim 128
--num_encoder_layers 3
--num_decoder_layers 3
--num_neighbors 48
--backbone_noise 0.2
--dropout 0.1
--mixed_precision True
--batch_size 10000
Published valid_acc: 0.524
```
