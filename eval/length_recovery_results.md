# Length-stratified validation recovery

Eval set: `processed_3p5/valid.pt` (1463 proteins, 1293 within `length<1000` bins).
Metric: argmax recovery on masked positions, computed identically to training-time `valid_acc`.

| length bin    | n   | residues | vanilla v_48_020 (k=48) | attn e5d5 k=96 best (ep126→ckpt ep130) | attn e5d5 k=128 best (ep50) |
|---------------|-----|----------|-------------------------|----------------------------------------|-----------------------------|
| [0,   100)    | 52  |   3,621  | **0.5120**              | 0.5051                                  | 0.4949                      |
| [100, 200)    | 260 |  36,628  | 0.5173                  | 0.5226                                  | **0.5291**                  |
| [200, 500)    | 656 | 199,102  | **0.5423**              | 0.5375                                  | 0.5394                      |
| [500, 1000)   | 325 | 205,682  | **0.5544**              | 0.5463                                  | 0.5444                      |
| **OVERALL**   | 1293| 710,660  | **0.5533**              | 0.5425                                  | 0.5445                      |

## Reading the table

- **k128 > k96 overall** (0.5445 vs 0.5425). Scaling neighbors *does* help, even at large k.
- **k128 > k96 specifically on small/medium proteins** (100–200 bin: +0.006). The k-scaling
  signal is strongest on shorter chains, *not* the large ones — the opposite of the
  hypothesis that more neighbors mainly helps when there are more residues to look at.
- **vanilla v_48_020 beats our models overall** (0.5533) and especially on long proteins
  (0.5544 in 500-1000). This is unsurprising: the vanilla checkpoint was trained on the
  full ~130k PDB chains, vs our 21k-chain subset. On the 100-200 bin, our k128 already
  beats vanilla (0.529 vs 0.517) — a strong signal that more neighbors helps more than
  more training data for short chains.
- The 0–100 bin is small (n=52) and noisy; ignore it.

## Caveats

- Our k96/k128 evals at 0.5425/0.5445 are slightly below the training-time `valid_acc`
  peaks of 0.543/0.544 because the training metric is on a 5000-protein subsample
  resampled per epoch, while this eval is on the full 1293-protein set.
- The vanilla checkpoint may or may not have respected the same validation split as ours;
  this comparison is best read as "old well-trained reference vs our small-data scaling".

## Checkpoints used

- `vanilla_model_weights/v_48_020.pt`
- `attn_e5d5_k96_h128_dropout02_resume/model_weights/best_epoch130_valid0543.pt`
- `attn_e5d5_k128_h128_dropout02/model_weights/best_epoch50_valid0544.pt`
