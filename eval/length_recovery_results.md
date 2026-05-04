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

## Checkpoints used (Phase 11/12)

- `vanilla_model_weights/v_48_020.pt`
- `attn_e5d5_k96_h128_dropout02_resume/model_weights/best_epoch130_valid0543.pt`
- `attn_e5d5_k128_h128_dropout02/model_weights/best_epoch50_valid0544.pt`

---

# Phase 13 — full 232k pdb_2021aug02 cache, k=256 and k=320

After moving from the 21k subset to the full ~232k pdb_2021aug02 cache and
scaling k from 128 → 256 → 320, the attention-aggregation architecture finally
beats the published vanilla v_48_020 baseline on the held-out
`processed_3p5/valid.pt` set. The bin range was also extended to ≥1000 to
expose the long-protein regime that the previous tables hid in OVERALL.

The k320 production run plateaued at training-time per-protein
`valid_acc` 0.593 (peak ep587, oscillating 0.586–0.592 over the last ~50
epochs). It was cancelled at ep627 / 21h30 of 24h limit on 2026-05-04 once
plateau was clear.

## Per-protein accuracy (primary) — Phase 13

| length bin     | n    | vanilla v_48_020 (k=48) | attn e5d5 k=256 (ep_last)   | attn e5d5 k=320 (ep540, best saved) | attn e5d5 k=320 (ep_last, ep627) |
|----------------|------|-------------------------|------------------------------|--------------------------------------|----------------------------------|
| [0,    100)    |  52  | **0.4982 ± 0.1044**     | 0.4735 ± 0.1260              | 0.4677 ± 0.1228                      | 0.4749 ± 0.1189                  |
| [100,  200)    | 260  | 0.5180 ± 0.0940         | 0.5217 ± 0.1301              | **0.5218 ± 0.1319**                  | 0.5198 ± 0.1326                  |
| [200,  500)    | 656  | 0.5392 ± 0.0745         | 0.5685 ± 0.1165              | **0.5870 ± 0.1284**                  | 0.5869 ± 0.1292                  |
| [500, 1000)    | 325  | 0.5514 ± 0.0804         | 0.5605 ± 0.0997              | 0.5795 ± 0.1090                      | **0.5799 ± 0.1092**              |
| [1000, 2000)   | 125  | 0.5652 ± 0.0768         | 0.5717 ± 0.0938              | 0.5903 ± 0.1008                      | **0.5915 ± 0.0993**              |
| [2000, 10000)  |  45  | 0.5668 ± 0.0823         | 0.5637 ± 0.0972              | 0.5860 ± 0.1074                      | **0.5876 ± 0.1075**              |
| **OVERALL**    | 1463 | 0.5398 ± 0.0828         | 0.5551 ± 0.1160              | 0.5698 ± 0.1259                      | **0.5698 ± 0.1262**              |

## Residue-weighted accuracy (secondary) — Phase 13

| length bin     | n    | vanilla v_48_020 (k=48) | attn e5d5 k=256 (ep_last) | attn e5d5 k=320 (ep540) | attn e5d5 k=320 (ep_last) |
|----------------|------|-------------------------|----------------------------|--------------------------|----------------------------|
| [0,    100)    |  52  | **0.5051**              | 0.4803                     | 0.4747                   | 0.4822                     |
| [100,  200)    | 260  | 0.5200                  | 0.5255                     | **0.5266**               | 0.5254                     |
| [200,  500)    | 656  | 0.5423                  | 0.5680                     | **0.5892**               | 0.5892                     |
| [500, 1000)    | 325  | 0.5539                  | 0.5606                     | 0.5788                   | **0.5792**                 |
| [1000, 2000)   | 125  | 0.5668                  | 0.5744                     | 0.5931                   | **0.5938**                 |
| [2000, 10000)  |  45  | 0.5637                  | 0.5694                     | 0.5971                   | **0.5994**                 |
| **OVERALL**    | 1463 | 0.5530                  | 0.5648                     | 0.5844                   | **0.5851**                 |

## Reading the Phase 13 tables

- **The headline:** k320 OVERALL per-protein **0.5698 vs vanilla 0.5398 = +0.030**. Residue-weighted: **0.5851 vs 0.5530 = +0.032**. Both are real but modest given the architectural and data scale-up. Per-protein std is ~0.10 → the OVERALL gap is roughly 0.3σ on the protein-level distribution.
- **k320 wins on 5 of 6 length bins**, losing only on [0, 100) by −0.024 (vanilla 0.498 → k320 0.475). With n=52 and std≈0.12 (SE≈0.017), that is ~1.4σ — small and could be a real architectural cost on very short chains where K ≈ L makes attention degenerate to all-to-all.
- **Long proteins do not saturate:** [1000, 2000) gain is +0.026 and [2000, 10000) is +0.021, comparable to the [200, 500) and [500, 1000) gains. The "k=320 still has more room on long proteins" hypothesis is **falsified** — adding even more long-range capacity is unlikely to be the bottleneck.
- **k256 → k320 marginal gain is +0.015** (OVERALL per-protein 0.5551 → 0.5698). The k192→k256→k320 curve flattens. We are at the architectural plateau for this design.
- **ep540 vs ep_last (ep627) are tied within noise** on every bin (max delta 0.007 on [0, 100)). The plateau-stage extra epochs neither helped nor hurt; the model has converged.

## Conclusion

The Phase-13 result settles the inverse-folding scaling question for our
attention-aggregation architecture: at 232k chains and k=320, we beat the
published vanilla baseline by +0.030 per-protein recovery overall and on
every length bin except the smallest (n=52). Further architectural sweeps
(deeper, wider, hierarchical-K) are unlikely to produce a meaningfully
larger gap. The remaining frontier is either (a) data scale (an order of
magnitude more training proteins) or (b) a different objective entirely
(fold-back consistency via ESMFold, multi-task with DMS fitness data, or
distillation from a larger model). Architectural search on this dataset is
done.

## Phase 13 checkpoints used

- `vanilla_model_weights/v_48_020.pt`
- `prod_k256_e5d5_fullcache/model_weights/epoch_last.pt`
- `prod_k320_e5d5_fullcache/model_weights/epoch540_step590124.pt`
- `prod_k320_e5d5_fullcache/model_weights/epoch_last.pt`
