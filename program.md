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
TORCHDYNAMO_DISABLE=1 timeout 7200 python training/training.py \
  --path_for_training_data ~/pdb_data/processed_3p5 \
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
  --rescut 2.0 \
  --save_model_every_n_epochs 1 \
  --reload_data_every_n_epochs 200 \  > run.log 2>&1
```

`timeout 7200` enforces the 30-minute budget. Change flags as needed per experiment.

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
# Full training curve
grep "valid_acc" run.log

# valid_acc at step 100 (primary metric)
# The log prints: epoch: E, step: S, ... valid_acc: X
# Find the first epoch where cumulative step S >= 100
grep "^epoch" run.log | awk -F'[,:]' '{
  for(i=1;i<=NF;i++) {
    if($i ~ /step/) step=$(i+1)
    if($i ~ /valid_acc/) acc=$(i+1)
  }
  if(step+0 >= 100 && !found) { print "valid_acc_s100: " acc; found=1 }
}'

# valid_acc at 2hr timeout (secondary metric)
grep "valid_acc" run.log | awk -F'valid_acc: ' '{print $2}' | sort -n | tail -1
```

## Results tracking

Record TWO metrics per experiment in results.tsv:

| Column | Meaning |
|---|---|
| valid_acc_s100 | valid_acc at first epoch where step >= 100 (architectural quality) |
| valid_acc_2hr | best valid_acc at 2hr timeout (practical quality) |
| steps_completed | total steps run |
| s_per_step | seconds per step |

PRIMARY comparison: valid_acc_s100
SECONDARY: valid_acc_2hr


---

## Experiment loop

#### Step 0: Warm OS file cache (every session, before baseline)
\```bash
timeout 1200 python training/training.py \
  --path_for_training_data ~/pdb_data/processed_3p5 \
  --path_for_outputs ~/pdb_data/warmup \
  --num_epochs 1 \
  --num_examples_per_epoch 50000 \
  --rescut 2.0 \
  > /dev/null 2>&1
echo "Cache warmed."
\```
This run is discarded. Do not record in results.tsv.
All subsequent runs including baseline start from warm cache and are fairly compared.

#### Steps 1-6:
\```

1. git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)
2. Confirm data: ls ~/pdb_data/processed_3p5_3p5/ | head -5
3. Run baseline (30 min):
     TORCHDYNAMO_DISABLE=1 timeout 7200 python training/training.py \
       --path_for_training_data ~/pdb_data/processed_3p5 \
       --path_for_outputs ~/pdb_data/baseline \
       --hidden_dim 128 --num_neighbors 48 \
       --num_encoder_layers 3 --num_decoder_layers 3 \
       --batch_size 10000 --mixed_precision True \
       --rescut 2.0 \
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
     TORCHDYNAMO_DISABLE=1 timeout 7200 python training/training.py \
       --path_for_training_data ~/pdb_data/processed_3p5 \
       --path_for_outputs ~/pdb_data/<exp_name> \
       --save_model_every_n_epochs 1 \
       --reload_data_every_n_epochs 200 \
       --mixed_precision True \
       --rescut 2.0 \
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
--rescut 2.0
Published valid_acc: 0.524
```
