# ProteinMPNN AutoResearch Agent Program — DGX Spark

You are an autonomous ML research agent. Your goal is to improve ProteinMPNN's sequence
recovery on the standard benchmark by systematically exploring larger model capacity combined
with richer neighbor context. You work in a tight experiment loop: propose a change, implement
it, train for a fixed 30-minute budget, evaluate sequence recovery on the held-out test set,
keep or revert, and repeat — without human involvement.

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
- **No VRAM limit** — 128 GB unified memory means you can run much larger models than
  the original ProteinMPNN paper. The original hidden_dim=128 model uses only ~4 GB.
  You have headroom for hidden_dim=256, 512, or larger.
- **BF16 preferred** — always use BF16 for training. FP32 wastes memory bandwidth.
- **Use SDPA** — use `torch.nn.functional.scaled_dot_product_attention` for any attention
  operations. Do NOT use flash_attn (incompatible with this platform).
- **torch.compile** — available and beneficial. Use `mode="reduce-overhead"`.
- **Memory bandwidth is the bottleneck** — larger models improve arithmetic intensity
  and make better use of available compute.

---

## Objective

Maximize **sequence recovery (%)** on the standard ProteinMPNN test set of 402 monomers,
within a **fixed 30-minute wall-clock training budget** per experiment.

The baseline to beat: **52.4%** sequence recovery (original ProteinMPNN paper,
hidden_dim=128, k_neighbors=48, 3 encoder + 3 decoder layers).

Sequence recovery = fraction of native amino acid identities correctly predicted
when redesigning sequences on native backbone structures.

This metric is objective and non-gameable — it is computed on a fixed held-out test
set using greedy decoding at temperature T=0.1.

---

## Core research hypothesis

**Larger model capacity + more neighbors = better sequence recovery.**

The original model is capacity-limited: hidden_dim=128 cannot fully exploit the
geometric context from 48 neighbors. Surface residues (hardest to recover, ~35%
recovery) are particularly limited by sparse local context. Increasing both model
size and neighbor count should improve recovery, especially on surface residues.

### Primary variables to explore (in priority order)

1. **k_neighbors** — number of nearest neighbors in the protein graph
   - Baseline: 48
   - Try: 64, 96, 128
   - Rationale: more structural context per residue, especially beneficial for
     surface residues and interface regions

2. **hidden_dim** — node and edge feature dimensionality
   - Baseline: 128
   - Try: 256, 384, 512
   - Rationale: larger hidden dim can represent richer geometric relationships;
     128 is very small for a modern graph neural network

3. **num_encoder_layers / num_decoder_layers** — network depth
   - Baseline: 3 encoder + 3 decoder
   - Try: 4+4, 5+5, 6+6
   - Rationale: deeper networks can learn longer-range structural dependencies

4. **Combined scaling** — after identifying best individual settings, try
   combining them (e.g. hidden_dim=256 + k=64 + depth=4+4)

### What NOT to explore yet
- Do not change the loss function
- Do not change the data augmentation (backbone noise)
- Do not change the optimizer type (Adam is fine)
- Do not explore attention mechanisms — focus on the graph network scaling first
- Do not change the dataset or test set

---

## Repository structure
```
ProteinMPNN/
├── training/
│   ├── train.py          ← PRIMARY FILE YOU MODIFY
│   └── README.md
├── protein_mpnn_utils.py ← Core model definition, may modify
├── eval/
│   └── eval_sequence_recovery.py  ← DO NOT MODIFY
└── program.md            ← This file
```

**You may modify:** `training/train.py`, `protein_mpnn_utils.py`
**Do not modify:** `eval/eval_sequence_recovery.py`, the test set

---

## Experiment loop (follow exactly)

### Setup (once per session)
```
1. git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)
2. Read protein_mpnn_utils.py — understand ProteinMPNN class parameters:
   hidden_dim, num_encoder_layers, num_decoder_layers, k_neighbors, augment_eps
3. Read training/train.py — understand training loop, optimizer, LR schedule
4. Run baseline evaluation:
     python eval/eval_sequence_recovery.py --model_path pretrained/v_48_020.pt
   Record baseline. Should be ~52.4%.
5. Initialize results.tsv (do NOT commit):
     echo -e "experiment\tseq_recovery\tpeak_mem_mb\tnotes" > results.tsv
6. Confirm data paths. Await go signal.
```

### Per-experiment loop
```
LOOP:
  1. THINK — review results.tsv and git log.
     Write a one-line hypothesis before touching any code.

  2. IMPLEMENT — make ONE focused change. One variable per experiment.

  3. TRAIN (30 minutes):
       python training/train.py \
         --[your flags] \
         --time_budget 1800 \
         > run.log 2>&1

  4. EVALUATE:
       python eval/eval_sequence_recovery.py \
         --model_path outputs/current_experiment/best_model.pt \
         >> run.log 2>&1
     
     Read result:
       grep "sequence_recovery\|mean_recovery" run.log | tail -3

  5. RECORD in results.tsv:
       <description>  <seq_recovery>  <peak_mem_mb>  <notes>

  6. DECIDE:
     - If seq_recovery IMPROVED: git add -p && git commit -m "exp: <description>"
     - If equal or worse: git checkout -- .

  7. GOTO LOOP
```

---

## Practical guidance

### Memory budget
Stay below 100 GB peak. The OS and agent occupy ~20 GB.
System OOM on the Spark freezes the whole machine — be conservative
with new size combinations.

### Batch size
Reduce batch size before abandoning a large model configuration.
A smaller batch with a bigger model is better than reverting to a small model.

### BF16 training
Always train in BF16:
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(...)
```

### Fair comparison
All evaluations: same test set, T=0.1, greedy decoding.
Do not change evaluation parameters between experiments.

---

## Progress reporting

After every 5 experiments:
```
=== Progress report ===
Best seq_recovery: X.X%
Baseline: 52.4%
Delta: +X.X%
Changes that helped: [list]
Changes that hurt: [list]
Current best config: hidden_dim=X, k=X, depth=X+X
Next hypothesis: <one sentence>
```

Commit a RESULTS.md at session end.

---

## Reference: original model parameters
```python
hidden_dim = 128
num_encoder_layers = 3
num_decoder_layers = 3
k_neighbors = 48
backbone_noise = 0.20
dropout = 0.1
vocab_size = 21
```
