"""
eval_iterative.py — Compare autoregressive vs iterative confidence-based decoding.

Strategy 1 (autoregressive): Standard model.sample() with random decoding order.
Strategy 2 (iterative): Decode all positions simultaneously via unconditional_probs(),
    fix the highest-confidence position, re-run with that position given, repeat.

Usage:
    python eval/eval_iterative.py \
        --checkpoint vanilla_model_weights/v_48_020.pt \
        --valid_data /path/to/valid.pt \
        --num_proteins 500
"""

import argparse
import sys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root to path so we can import protein_mpnn_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protein_mpnn_utils import ProteinMPNN, ProteinFeatures

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def featurize_single(protein, device):
    """Featurize a single protein dict into tensors (batch size 1)."""
    B = 1
    seq = protein['seq']
    L = len(seq)

    X = np.zeros([B, L, 4, 3])
    residue_idx = -100 * np.ones([B, L], dtype=np.int32)
    chain_M = np.zeros([B, L], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L], dtype=np.int32)
    S = np.zeros([B, L], dtype=np.int32)

    all_chains = protein['masked_list'] + protein['visible_list']

    x_chain_list = []
    chain_mask_list = []
    chain_seq_list = []
    chain_encoding_list = []
    c = 1
    l0 = 0

    for letter in all_chains:
        chain_seq = protein[f'seq_chain_{letter}']
        chain_length = len(chain_seq)
        chain_coords = protein[f'coords_chain_{letter}']

        if letter in protein['masked_list']:
            chain_mask = np.ones(chain_length)
        else:
            chain_mask = np.zeros(chain_length)

        x_chain = np.stack([
            chain_coords[f'N_chain_{letter}'],
            chain_coords[f'CA_chain_{letter}'],
            chain_coords[f'C_chain_{letter}'],
            chain_coords[f'O_chain_{letter}'],
        ], axis=1)

        x_chain_list.append(x_chain)
        chain_mask_list.append(chain_mask)
        chain_seq_list.append(chain_seq)
        chain_encoding_list.append(c * np.ones(chain_length))

        l1 = l0 + chain_length
        residue_idx[0, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
        l0 = l1
        c += 1

    x = np.concatenate(x_chain_list, 0)
    all_sequence = "".join(chain_seq_list)
    m = np.concatenate(chain_mask_list, 0)
    chain_encoding = np.concatenate(chain_encoding_list, 0)

    X[0, :L] = x
    chain_M[0, :L] = m
    chain_encoding_all[0, :L] = chain_encoding

    indices = np.array([ALPHABET.index(a) for a in all_sequence], dtype=np.int32)
    S[0, :L] = indices

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    isnan = np.isnan(X)
    X[isnan] = 0.0

    X = torch.from_numpy(X).float().to(device)
    S = torch.from_numpy(S).long().to(device)
    mask = torch.from_numpy(mask).float().to(device)
    chain_M = torch.from_numpy(chain_M).float().to(device)
    residue_idx = torch.from_numpy(residue_idx).long().to(device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).long().to(device)

    return X, S, mask, chain_M, residue_idx, chain_encoding_all


def autoregressive_decode(model, X, S_true, mask, chain_M, residue_idx,
                          chain_encoding_all, temperature=0.1, num_samples=1):
    """Standard autoregressive decoding via model.sample().
    Returns best (highest recovery) sequence across num_samples."""
    device = X.device
    N_nodes = X.size(1)
    best_recovery = -1.0
    best_S = None

    omit_AAs_np = np.zeros(21)
    bias_AAs_np = np.zeros(21)
    chain_M_pos = torch.ones_like(chain_M)
    bias_by_res = torch.zeros(X.size(0), N_nodes, 21, device=device)

    for _ in range(num_samples):
        randn = torch.randn(X.size(0), N_nodes, device=device)
        output = model.sample(
            X, randn, S_true, chain_M, chain_encoding_all, residue_idx,
            mask=mask, temperature=temperature,
            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
            chain_M_pos=chain_M_pos, omit_AA_mask=None,
            pssm_coef=None, pssm_bias=None, pssm_multi=None,
            pssm_log_odds_flag=None, pssm_log_odds_mask=None,
            pssm_bias_flag=None, bias_by_res=bias_by_res,
        )
        S_sample = output['S']
        mask_for_recovery = mask * chain_M
        correct = (S_sample == S_true).float() * mask_for_recovery
        recovery = correct.sum() / mask_for_recovery.sum()
        if recovery.item() > best_recovery:
            best_recovery = recovery.item()
            best_S = S_sample.clone()

    return best_S, best_recovery


def iterative_decode(model, X, S_true, mask, chain_M, residue_idx,
                     chain_encoding_all):
    """Iterative confidence-based decoding.

    1. Run unconditional_probs() to get log_probs for all positions (no sequence context).
    2. Pick the highest-confidence masked position, fix it.
    3. Re-run with the fixed position treated as given (unmask it).
    4. Repeat until all masked positions are filled.
    """
    device = X.device
    N_batch, N_nodes = X.size(0), X.size(1)

    # Start with the true sequence for fixed positions, zeros for masked
    S_current = S_true.clone()
    remaining_mask = (chain_M * mask).clone()  # 1 = needs prediction

    # Track which positions still need to be predicted
    remaining_indices = remaining_mask[0].nonzero(as_tuple=True)[0].tolist()

    while remaining_indices:
        # Build current chain_M: 1 for positions still to be predicted
        current_chain_M = remaining_mask.clone()

        # Use conditional_probs approach: for each remaining position,
        # compute log_probs conditioned on all already-fixed positions.
        # This is expensive (one forward pass per remaining position),
        # so we use a simpler approach: run the full decoder with all
        # fixed positions visible and predict all remaining positions at once.

        # Run forward pass with current partial sequence
        randn = torch.randn(N_batch, N_nodes, device=device)
        log_probs = model.forward(
            X, S_current, mask, current_chain_M, residue_idx,
            chain_encoding_all, randn
        )

        # For remaining positions, find the one with highest confidence
        probs = torch.exp(log_probs)  # [B, N, 21]
        max_prob, max_aa = probs[0].max(dim=-1)  # [N], [N]

        # Among remaining positions, find highest confidence
        best_pos = None
        best_conf = -1.0
        for idx in remaining_indices:
            conf = max_prob[idx].item()
            if conf > best_conf:
                best_conf = conf
                best_pos = idx

        # Fix this position
        S_current[0, best_pos] = max_aa[best_pos]
        remaining_mask[0, best_pos] = 0.0
        remaining_indices.remove(best_pos)

    # Compute recovery
    mask_for_recovery = mask * chain_M
    correct = (S_current == S_true).float() * mask_for_recovery
    recovery = correct.sum() / mask_for_recovery.sum()

    return S_current, recovery.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='vanilla_model_weights/v_48_020.pt')
    parser.add_argument('--valid_data', type=str,
                        default='/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5/valid.pt')
    parser.add_argument('--num_proteins', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for autoregressive sampling')
    parser.add_argument('--ar_samples', type=int, default=4,
                        help='Number of autoregressive samples (take best)')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Skip proteins longer than this')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_edges = ckpt.get('num_edges', 48)

    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=num_edges, augment_eps=0.0, dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Load validation data
    print(f"Loading validation data: {args.valid_data}")
    all_valid = torch.load(args.valid_data, map_location='cpu', weights_only=False)
    print(f"  {len(all_valid)} validation proteins")

    # Filter by length and take first N
    valid_proteins = [p for p in all_valid if len(p['seq']) <= args.max_length]
    valid_proteins = valid_proteins[:args.num_proteins]
    print(f"  Evaluating {len(valid_proteins)} proteins (max_length={args.max_length})")

    # Evaluate both strategies
    ar_recoveries = []
    iter_recoveries = []
    ar_better = 0
    iter_better = 0
    tied = 0

    t0 = time.time()
    with torch.no_grad():
        for i, protein in enumerate(valid_proteins):
            X, S, mask, chain_M, residue_idx, chain_encoding_all = \
                featurize_single(protein, device)

            n_masked = int((chain_M * mask).sum().item())
            if n_masked == 0:
                continue

            # Strategy 1: Autoregressive
            _, ar_rec = autoregressive_decode(
                model, X, S, mask, chain_M, residue_idx,
                chain_encoding_all, temperature=args.temperature,
                num_samples=args.ar_samples,
            )
            ar_recoveries.append(ar_rec)

            # Strategy 2: Iterative
            _, iter_rec = iterative_decode(
                model, X, S, mask, chain_M, residue_idx,
                chain_encoding_all,
            )
            iter_recoveries.append(iter_rec)

            if ar_rec > iter_rec + 0.001:
                ar_better += 1
            elif iter_rec > ar_rec + 0.001:
                iter_better += 1
            else:
                tied += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(valid_proteins)}] "
                      f"AR={np.mean(ar_recoveries):.4f} "
                      f"Iter={np.mean(iter_recoveries):.4f} "
                      f"({elapsed:.0f}s)")

    # Final results
    elapsed = time.time() - t0
    ar_mean = np.mean(ar_recoveries)
    iter_mean = np.mean(iter_recoveries)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Proteins evaluated: {len(ar_recoveries)}")
    print(f"Time: {elapsed:.0f}s")
    print()
    print(f"Autoregressive (T={args.temperature}, best of {args.ar_samples}):")
    print(f"  Mean recovery: {ar_mean:.4f} ({ar_mean*100:.1f}%)")
    print(f"  Median:        {np.median(ar_recoveries):.4f}")
    print(f"  Std:           {np.std(ar_recoveries):.4f}")
    print()
    print(f"Iterative confidence-based:")
    print(f"  Mean recovery: {iter_mean:.4f} ({iter_mean*100:.1f}%)")
    print(f"  Median:        {np.median(iter_recoveries):.4f}")
    print(f"  Std:           {np.std(iter_recoveries):.4f}")
    print()
    print(f"Delta (iter - AR): {iter_mean - ar_mean:+.4f} ({(iter_mean-ar_mean)*100:+.2f}%)")
    print(f"AR better: {ar_better}, Iter better: {iter_better}, Tied: {tied}")

    # Per-protein breakdown by length
    print("\n--- Recovery by protein length ---")
    lengths = [len(p['seq']) for p in valid_proteins[:len(ar_recoveries)]]
    for lo, hi in [(0, 100), (100, 200), (200, 500), (500, 1000)]:
        idx = [j for j, l in enumerate(lengths) if lo <= l < hi]
        if idx:
            ar_sub = [ar_recoveries[j] for j in idx]
            it_sub = [iter_recoveries[j] for j in idx]
            print(f"  [{lo:4d}-{hi:4d}): n={len(idx):3d}  "
                  f"AR={np.mean(ar_sub):.4f}  Iter={np.mean(it_sub):.4f}  "
                  f"delta={np.mean(it_sub)-np.mean(ar_sub):+.4f}")


if __name__ == '__main__':
    main()
