"""
eval_length_recovery.py — Length-stratified valid_acc on the held-out set.

Reports mean argmax recovery (matching the training-time `valid_acc` metric)
broken down by protein length bin: <100, 100-200, 200-500, 500-1000.

Supports both the vanilla ProteinMPNN architecture (e3d3, no attention,
defined in protein_mpnn_utils.py) and our attention-aggregation variant
(e5d5 by default, defined in training/model_utils_attn.py).

Usage:
    python eval/eval_length_recovery.py \\
        --checkpoint vanilla_model_weights/v_48_020.pt \\
        --model vanilla --num_encoder_layers 3 --num_decoder_layers 3 \\
        --k_neighbors 48

    python eval/eval_length_recovery.py \\
        --checkpoint /path/to/attn_e5d5_k96.../model_weights/best_epoch130_valid0543.pt \\
        --model attn --num_encoder_layers 5 --num_decoder_layers 5 \\
        --k_neighbors 96
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Allow imports from both repo root (for protein_mpnn_utils) and training/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'training'))


def build_model(args, device):
    if args.model == 'vanilla':
        from protein_mpnn_utils import ProteinMPNN
    elif args.model == 'attn':
        from model_utils_attn import ProteinMPNN
    else:
        raise ValueError(f'unknown --model {args.model}')

    model = ProteinMPNN(
        num_letters=21,
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.k_neighbors,
        augment_eps=0.0,
        dropout=0.0,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def featurize_legacy(batch, device):
    """featurize_single style (one protein at a time) — works for both vanilla
    and attn ProteinMPNN since they share the same featurize signature."""
    from model_utils_attn import featurize
    return featurize(batch, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', choices=['vanilla', 'attn'], required=True)
    parser.add_argument('--valid_data', default='/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5/valid.pt')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--k_neighbors', type=int, default=48)
    parser.add_argument('--max_length', type=int, default=10000)
    parser.add_argument('--num_proteins', type=int, default=0,
                        help='0 = all valid proteins')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Model: {args.model} '
          f'(h={args.hidden_dim}, e{args.num_encoder_layers}d{args.num_decoder_layers}, '
          f'k={args.k_neighbors})')

    model = build_model(args, device)
    print('Model loaded.')

    # Validation data
    print(f'Loading valid_data: {args.valid_data}')
    all_valid = torch.load(args.valid_data, map_location='cpu', weights_only=False)
    print(f'  {len(all_valid)} validation proteins')

    proteins = [p for p in all_valid if len(p['seq']) <= args.max_length]
    if args.num_proteins > 0:
        proteins = proteins[:args.num_proteins]
    print(f'  Evaluating {len(proteins)} proteins')

    # Per-protein accuracies (each protein weighted equally regardless of length)
    bins = [(0, 100), (100, 200), (200, 500), (500, 1000)]
    bin_per_protein = {b: [] for b in bins}
    all_per_protein = []

    # Residue-weighted accumulators (kept as a secondary metric for sanity)
    bin_correct = {b: 0.0 for b in bins}
    bin_total = {b: 0.0 for b in bins}
    overall_correct = 0.0
    overall_total = 0.0

    t0 = time.time()
    with torch.no_grad():
        for i, protein in enumerate(proteins):
            try:
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = \
                    featurize_legacy([protein], device)
            except Exception as e:
                print(f'  skip {i}: featurize failed: {e}')
                continue

            with torch.cuda.amp.autocast():
                if args.model == 'attn':
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                else:
                    randn = torch.randn(X.size(0), X.size(1), device=device)
                    log_probs = model(X, S, mask, chain_M, residue_idx,
                                      chain_encoding_all, randn)

            pred = log_probs.argmax(dim=-1)  # [B, L]
            mask_for_loss = mask * chain_M
            correct = ((pred == S).float() * mask_for_loss).sum().item()
            total = mask_for_loss.sum().item()
            if total <= 0:
                continue

            # Per-protein accuracy: this protein's recovery on its own masked residues
            protein_acc = correct / total
            all_per_protein.append(protein_acc)

            # Residue-weighted accumulators
            overall_correct += correct
            overall_total += total

            L = len(protein['seq'])
            for lo, hi in bins:
                if lo <= L < hi:
                    bin_per_protein[(lo, hi)].append(protein_acc)
                    bin_correct[(lo, hi)] += correct
                    bin_total[(lo, hi)] += total
                    break

            if (i + 1) % 100 == 0:
                running_pp = float(np.mean(all_per_protein)) if all_per_protein else 0.0
                running_rw = overall_correct / max(overall_total, 1)
                print(f'  [{i+1}/{len(proteins)}] '
                      f'per-protein={running_pp:.4f}  '
                      f'residue-weighted={running_rw:.4f}  '
                      f'({time.time() - t0:.0f}s)')

    elapsed = time.time() - t0
    overall_pp = float(np.mean(all_per_protein)) if all_per_protein else 0.0
    overall_rw = overall_correct / max(overall_total, 1)

    print('\n' + '=' * 78)
    print('LENGTH-STRATIFIED RECOVERY')
    print('=' * 78)
    print(f'Checkpoint:  {args.checkpoint}')
    print(f'Architecture: {args.model} h={args.hidden_dim} '
          f'e{args.num_encoder_layers}d{args.num_decoder_layers} k={args.k_neighbors}')
    print(f'Time: {elapsed:.0f}s, proteins: {len(all_per_protein)}')
    print()
    print('PRIMARY metric — per-protein accuracy (each protein weighted equally):')
    print(f'  {"length bin":<14} {"n":>5} {"per_protein_acc":>16} {"std":>8}')
    print('  ' + '-' * 48)
    for lo, hi in bins:
        accs = bin_per_protein[(lo, hi)]
        n = len(accs)
        if n == 0:
            print(f'  [{lo:4d}, {hi:4d})   {n:>5d} {"-":>16} {"-":>8}')
            continue
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        print(f'  [{lo:4d}, {hi:4d})   {n:>5d} {mean:>16.4f} {std:>8.4f}')
    print('  ' + '-' * 48)
    print(f'  {"OVERALL":<14} {len(all_per_protein):>5d} {overall_pp:>16.4f} '
          f'{float(np.std(all_per_protein)):>8.4f}')

    print()
    print('SECONDARY metric — residue-weighted accuracy (matches training valid_acc):')
    print(f'  {"length bin":<14} {"n":>5} {"residues":>12} {"residue_wt_acc":>16}')
    print('  ' + '-' * 51)
    for lo, hi in bins:
        n = len(bin_per_protein[(lo, hi)])
        tot = bin_total[(lo, hi)]
        acc = bin_correct[(lo, hi)] / max(tot, 1)
        print(f'  [{lo:4d}, {hi:4d})   {n:>5d} {int(tot):>12d} {acc:>16.4f}')
    print('  ' + '-' * 51)
    print(f'  {"OVERALL":<14} {len(all_per_protein):>5d} {int(overall_total):>12d} '
          f'{overall_rw:>16.4f}')


if __name__ == '__main__':
    main()
