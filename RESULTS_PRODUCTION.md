# ProteinMPNN Production Run Results — Kempner H100 Multi-GPU

All runs use architecture: attention aggregation, e5d5, k=192, h=128, dropout=0.2
Dataset: pdb_2021aug02_full_cache (232,895 structures, 11,442 valid)
Metric: **per-protein** sequence recovery (each protein weighted equally)

## Run 1: From-Scratch 8-GPU (2 nodes x 4 H100s)

| Setting | Value |
|---|---|
| Job ID | 4793765 |
| EXP_NAME | prod_k192_e5d5_8gpu |
| Nodes | 2 (holygpu8a[11604,17602]) |
| GPUs | 8 x H100 80GB |
| Effective batch | ~80k tokens (8 x 10k) |
| LR schedule | Noam, factor=2, warmup=4000 |
| Epochs completed | 169 |
| Wall time | ~21.7 hours |
| Per-epoch time | ~460s |
| **Best valid_acc** | **0.547** (epoch 136, also epoch 140) |
| Final valid_acc | 0.544 (epoch 169) |
| Train/valid gap | 0.658 / 0.544 = +0.114 (mild overfit) |

### Trajectory highlights
- Epoch 1: 0.270
- Epoch 10: 0.504
- Epoch 50: 0.533
- Epoch 100: 0.543
- Epoch 136: 0.547 (best)
- Plateaued in 0.539-0.547 range from epoch ~100 onward

## Run 2: Fine-Tune 8-GPU (lr_scale=0.1, resumed from Run 1 best)

| Setting | Value |
|---|---|
| Job ID | 5063068 |
| EXP_NAME | prod_k192_e5d5_8gpu_finetune |
| Nodes | 2 (holygpu8a[13204,17604]) |
| GPUs | 8 x H100 80GB |
| Effective batch | ~80k tokens (8 x 10k) |
| LR schedule | Noam, factor=0.2 (lr_scale=0.1), warmup=4000 |
| Resumed from | epoch140_step75776.pt (Run 1 best, valid_acc=0.547) |
| Epochs completed | 161 (epoch 141 to 301) |
| Wall time | ~20.7 hours |
| Per-epoch time | ~460s |
| **Best valid_acc** | **0.552** (epoch 195, also epoch 203) |
| Final valid_acc | 0.549 (epoch 301) |
| Train/valid gap | 0.671 / 0.549 = +0.122 (moderate overfit) |

### Trajectory highlights
- Epoch 141 (resume): 0.547
- Epoch 195: 0.552 (best)
- Epoch 203: 0.552 (tied best)
- Plateaued in 0.546-0.552 range from epoch ~195 onward
- lr_scale=0.1 lifted plateau by +0.005

## Run 3: 4-GPU Half-Batch (1 node x 4 H100s, in progress)

| Setting | Value |
|---|---|
| Job ID | 5333812 |
| EXP_NAME | prod_k192_e5d5_4gpu_halfbatch |
| Nodes | 1 |
| GPUs | 4 x H100 80GB |
| Effective batch | ~40k tokens (4 x 10k) |
| LR schedule | Noam, factor=0.2 (lr_scale=0.1), warmup=4000 |
| Resumed from | epoch195_step105561.pt (Run 2 best, valid_acc=0.552) |
| Status | Submitted, pending |

## Comparison to prior single-GPU results

| Run | GPUs | Metric | Best valid_acc | Notes |
|---|---|---|---|---|
| Phase 12 k192 (1-GPU) | 1 | residue-weighted | 0.562 | 89 ep, still climbing at 24hr timeout |
| Phase 11 k128 (1-GPU) | 1 | per-protein | 0.537 | 52 ep, 12hr timeout |
| Prod Run 1 (8-GPU) | 8 | per-protein | 0.547 | 169 ep from scratch |
| Prod Run 2 (8-GPU) | 8 | per-protein | 0.552 | finetune from Run 1, lr_scale=0.1 |

**Important:** Phase 12 results used residue-weighted accuracy. Per-protein is typically
~0.005-0.010 lower. The 0.552 per-protein from Run 2 likely corresponds to ~0.557-0.562
residue-weighted, approximately matching the Phase 12 best.

## Code changes for multi-GPU production

1. `model_utils_attn.py`: mask_self kept on CPU (was GPU, saving ~50+ GiB/GPU)
2. `model_utils_attn.py`: _get_rbf uses gather_nodes O(L*K) instead of O(L^2)
3. `model_utils_attn.py`: get_std_opt takes lr_scale parameter
4. `training_attn.py`: --lr_scale CLI flag for fine-tuning resumes
5. `slurm/train_production.sh`: NCCL IB config for multi-node, LR_SCALE pass-through

## Checkpoints

Best checkpoints saved at:
- Run 1: `/outputs/prod_k192_e5d5_8gpu/model_weights/epoch140_step75776.pt`
- Run 2: `/outputs/prod_k192_e5d5_8gpu_finetune/model_weights/epoch195_step105561.pt`
