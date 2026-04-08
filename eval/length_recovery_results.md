# Length-stratified validation recovery

Eval set: `processed_3p5/valid.pt` (1463 proteins).
Bins cover 1293 proteins; the remaining 170 have length ≥ 1000 and are
included in OVERALL but not in any bin row.

Two metrics are reported for every model:

- **Per-protein accuracy (PRIMARY)** — compute argmax recovery for each protein
  independently, then average across proteins. Each protein contributes equally,
  regardless of length. This is the standard sequence-recovery metric used in the
  ProteinMPNN literature.
- **Residue-weighted accuracy (SECONDARY)** — sum correct residues / total residues
  across all proteins. Long proteins dominate. This matches the training-time
  `valid_acc` print and is kept here for sanity-checking against training logs.

## Per-protein accuracy (primary)

| length bin    | n    | vanilla v_48_020 (k=48) | attn e5d5 k=96 (best_ep130) | attn e5d5 k=128 (best_ep50) |
|---------------|------|-------------------------|-----------------------------|-----------------------------|
| [0,   100)    |  52  | 0.5022 ± 0.1063         | 0.5032 ± 0.1186             | 0.4993 ± 0.1215             |
| [100, 200)    | 260  | 0.5174 ± 0.0936         | 0.5198 ± 0.0995             | **0.5271 ± 0.1165**         |
| [200, 500)    | 656  | **0.5390 ± 0.0742**     | 0.5360 ± 0.0748             | 0.5378 ± 0.0829             |
| [500, 1000)   | 325  | **0.5520 ± 0.0792**     | 0.5438 ± 0.0779             | 0.5442 ± 0.0818             |
| **OVERALL**   | 1463 | **0.5398 ± 0.0823**     | 0.5353 ± 0.0829             | 0.5378 ± 0.0908             |

## Residue-weighted accuracy (secondary, matches training valid_acc)

| length bin    | n    | vanilla v_48_020 (k=48) | attn e5d5 k=96 (best_ep130) | attn e5d5 k=128 (best_ep50) |
|---------------|------|-------------------------|-----------------------------|-----------------------------|
| [0,   100)    |  52  | 0.5093                  | 0.5095                      | 0.5059                      |
| [100, 200)    | 260  | 0.5197                  | 0.5219                      | **0.5300**                  |
| [200, 500)    | 656  | **0.5421**              | 0.5381                      | 0.5398                      |
| [500, 1000)   | 325  | **0.5542**              | 0.5452                      | 0.5454                      |
| **OVERALL**   | 1463 | **0.5529**              | 0.5422                      | 0.5451                      |

## Reading the table

- **k128 > k96 in both metrics** (per-protein: 0.5378 > 0.5353; residue-weighted:
  0.5451 > 0.5422). Pushing k from 96 → 128 helps even though both training runs
  plateaued near the same training-time `valid_acc` (~0.544).
- **The k-scaling signal is strongest on the 100–200 bin** in both metrics. With
  per-protein accuracy, k128 = 0.5271 vs k96 = 0.5198 (+0.0073) vs vanilla = 0.5174
  (+0.0097). Our k128 already beats the vanilla baseline on short chains.
- **Vanilla wins overall and on long proteins** (200–500 and 500–1000 bins). The
  vanilla v_48_020 was trained on the full ~130k PDB chains for many more
  iterations than our 21k-chain runs, so it has a substantial training-data
  advantage. The takeaway is *not* "k=48 is better than k=128"; it is "with 6×
  less training data, our k=128 model recovers most of the gap and wins on short
  chains".
- **The per-protein metric shows consistently lower numbers than residue-weighted**
  because short proteins (which are harder to recover well) are upweighted under
  per-protein averaging. The metric ranking is identical to residue-weighted in
  this comparison, but per-protein is more honest because each protein "votes"
  once.
- **Standard deviations are large (~0.08–0.12)** — most of the inter-model deltas
  here are well within 1σ of the per-protein distribution. The OVERALL k128 vs k96
  delta (+0.0025) is small relative to that noise; the 100–200-bin delta (+0.0073)
  is larger but still within σ. Treat single-decimal differences as noise.

## Caveats

- The vanilla checkpoint was trained on a different (much larger) training set;
  this is best read as "old well-trained reference vs our small-data scaling
  experiments", not as a head-to-head architectural comparison.
- The 0–100 bin (n=52) is too small to draw conclusions from.

## Checkpoints used

- `vanilla_model_weights/v_48_020.pt`
- `attn_e5d5_k96_h128_dropout02_resume/model_weights/best_epoch130_valid0543.pt`
- `attn_e5d5_k128_h128_dropout02/model_weights/best_epoch50_valid0544.pt`
