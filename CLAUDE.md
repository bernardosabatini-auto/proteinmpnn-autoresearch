# ProteinMPNN AutoResearch — DGX Spark

Read `program.md` first. It contains your complete operating instructions.

## Quick reference

| File | Role | Editable? |
|------|------|-----------|
| `program.md` | Your instructions | You read this |
| `training/train.py` | Training loop | ✅ Yes |
| `protein_mpnn_utils.py` | Model architecture | ✅ Yes |
| `eval/eval_sequence_recovery.py` | Evaluation harness | ❌ No |
| `results.tsv` | Experiment log | Append only, don't commit |

## The one metric that matters
```bash
grep "sequence_recovery" run.log | tail -1
# Higher is better. Baseline: 52.4%
```

## Key model parameters to scale
```python
hidden_dim = 128        # → try 256, 384, 512
k_neighbors = 48        # → try 64, 96, 128
num_encoder_layers = 3  # → try 4, 5, 6
num_decoder_layers = 3  # → try 4, 5, 6
```

## Platform: DGX Spark
- 128 GB unified memory — no VRAM OOM, but system OOM will freeze the machine
- Stay under 100 GB peak memory
- Always use BF16
- Use SDPA not flash_attn
