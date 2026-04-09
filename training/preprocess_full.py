"""
Preprocess pdb_2021aug02 into a fully-enumerated, shard-on-disk cache.

Builds the same per-chain dict format that training/utils.py:get_pdbs
produces (so featurize() can consume it unchanged), but with two key
differences vs preprocess.py:

  1. Enumerates EVERY (cluster, chain) pair, not one random chain per
     cluster. The result spans all 438k+17k chain entries from
     pdb_2021aug02 instead of the ~21k that PDB_dataset would emit.

  2. Streams output to numbered shards on disk and clears the in-memory
     list between shards. Avoids the ~220 GB peak RSS that an in-memory
     accumulate of 438k chains hits (Python list overhead from .tolist()
     of coordinate arrays bloats per-entry size to ~500KB).

  3. Stores per-chain coordinates as numpy arrays (not Python lists),
     cutting on-the-wire size ~10x. featurize() consumes them via
     np.stack which is happy with either lists or arrays.

Usage:
    python training/preprocess_full.py \\
        --data_dir /n/netscratch/.../data/pdb_2021aug02 \\
        --out_dir  /n/netscratch/.../data/pdb_2021aug02_full_cache \\
        --rescut   3.5 \\
        --max_length 10000 \\
        --shard_size 50000 \\
        --num_workers 8

Output layout:
    <out_dir>/train_shard_0000.pt
    <out_dir>/train_shard_0001.pt
    ...
    <out_dir>/valid_shard_0000.pt
    <out_dir>/manifest.json    (lists every shard + total counts)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import build_training_clusters, loader_pdb, worker_init_fn


# ---------------------------------------------------------------------------
# AllChainsDataset: yields one assembly-format dict per (cluster, chain).
# ---------------------------------------------------------------------------

class AllChainsDataset(torch.utils.data.Dataset):
    """One entry per (cluster, chain) pair, no random sampling."""

    def __init__(self, cluster_dict, loader, params):
        self.items = []
        for items in cluster_dict.values():
            self.items.extend(items)
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            return self.loader(self.items[idx], self.params)
        except (FileNotFoundError, KeyError, RuntimeError, EOFError):
            # Sentinel that the format conversion below filters out.
            return {'seq': np.zeros(5)}


# ---------------------------------------------------------------------------
# Inlined get_pdbs format conversion. Differs from utils.py:get_pdbs by:
#   - Storing coordinate arrays as np.ndarray (NOT .tolist())
#   - Returning None on entries without 'label' (which the caller drops)
# ---------------------------------------------------------------------------

_INIT_ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
_EXTRA_ALPHABET = [str(i) for i in range(300)]
_CHAIN_ALPHABET = _INIT_ALPHABET + _EXTRA_ALPHABET


def _convert_one(t, max_length):
    """Convert one assembly-format dict (from loader_pdb) to the per-chain
    dict format that featurize() consumes. Returns None if the entry is
    invalid (sentinel, too long, no usable chains).
    """
    if 'label' not in t:
        return None

    idx_arr = t['idx']
    if len(np.unique(idx_arr)) >= 352:
        return None

    seq_chars = np.array(list(t['seq']))
    xyz = t['xyz']  # torch.Tensor [L_total, 14, 3]

    my_dict = {}
    mask_list = []
    visible_list = []
    concat_seq = ''
    masked_chains_set = set(int(x) for x in t['masked'].tolist()) if hasattr(t['masked'], 'tolist') else set(t['masked'])

    for chain_idx in np.unique(idx_arr):
        letter = _CHAIN_ALPHABET[int(chain_idx)]
        res = np.argwhere(idx_arr == chain_idx)
        initial_sequence = ''.join(seq_chars[res][0, ].tolist())

        # Strip 6-residue HHHHHH tags at various positions (matches get_pdbs).
        if initial_sequence[-6:] == 'HHHHHH':
            res = res[:, :-6]
        if initial_sequence[0:6] == 'HHHHHH':
            res = res[:, 6:]
        if initial_sequence[-7:-1] == 'HHHHHH':
            res = res[:, :-7]
        if initial_sequence[-8:-2] == 'HHHHHH':
            res = res[:, :-8]
        if initial_sequence[-9:-3] == 'HHHHHH':
            res = res[:, :-9]
        if initial_sequence[-10:-4] == 'HHHHHH':
            res = res[:, :-10]
        if initial_sequence[1:7] == 'HHHHHH':
            res = res[:, 7:]
        if initial_sequence[2:8] == 'HHHHHH':
            res = res[:, 8:]
        if initial_sequence[3:9] == 'HHHHHH':
            res = res[:, 9:]
        if initial_sequence[4:10] == 'HHHHHH':
            res = res[:, 10:]

        if res.shape[1] < 4:
            continue

        chain_seq = ''.join(seq_chars[res][0, ].tolist())
        my_dict[f'seq_chain_{letter}'] = chain_seq
        concat_seq += chain_seq

        if int(chain_idx) in masked_chains_set:
            mask_list.append(letter)
        else:
            visible_list.append(letter)

        # all_atoms shape: [L, 14, 3] as np.ndarray (compact, ~12 bytes/coord)
        all_atoms = np.asarray(xyz[res,])[0, ]
        coords_dict_chain = {
            f'N_chain_{letter}':  all_atoms[:, 0, :].astype(np.float32),
            f'CA_chain_{letter}': all_atoms[:, 1, :].astype(np.float32),
            f'C_chain_{letter}':  all_atoms[:, 2, :].astype(np.float32),
            f'O_chain_{letter}':  all_atoms[:, 3, :].astype(np.float32),
        }
        my_dict[f'coords_chain_{letter}'] = coords_dict_chain

    if len(concat_seq) == 0 or len(concat_seq) > max_length:
        return None
    if not (mask_list or visible_list):
        return None

    my_dict['name'] = t['label']
    my_dict['masked_list'] = mask_list
    my_dict['visible_list'] = visible_list
    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
    my_dict['seq'] = concat_seq
    return my_dict


def _unpack_loader_item(t):
    """torch DataLoader collate wraps every value in a length-1 batch dim
    when batch_size=1. Unwrap that here so _convert_one sees the same
    shape that the original get_pdbs sees."""
    return {k: v[0] for k, v in t.items()}


# ---------------------------------------------------------------------------
# Stream-write shards from a DataLoader.
# ---------------------------------------------------------------------------

def stream_to_shards(loader, out_dir, split_name, max_length, shard_size):
    shard_idx = 0
    in_shard = []
    n_total_emit = 0
    n_total_seen = 0
    n_dropped = 0
    t0 = time.time()
    shard_files = []

    def flush():
        nonlocal shard_idx, in_shard, shard_files
        if not in_shard:
            return
        path = os.path.join(out_dir, f'{split_name}_shard_{shard_idx:04d}.pt')
        torch.save(in_shard, path)
        size_gb = os.path.getsize(path) / 1e9
        print(f"  shard {shard_idx:04d} → {len(in_shard)} entries, {size_gb:.2f} GB",
              flush=True)
        shard_files.append(os.path.basename(path))
        shard_idx += 1
        in_shard = []

    for raw in loader:
        n_total_seen += 1
        t = _unpack_loader_item(raw)
        entry = _convert_one(t, max_length)
        if entry is None:
            n_dropped += 1
            continue
        in_shard.append(entry)
        n_total_emit += 1
        if len(in_shard) >= shard_size:
            flush()
        if n_total_seen % 5000 == 0:
            elapsed = time.time() - t0
            rate = n_total_seen / max(elapsed, 1e-9)
            print(f"  ... seen {n_total_seen}, kept {n_total_emit}, "
                  f"dropped {n_dropped}, {rate:.0f}/s, "
                  f"{elapsed/60:.1f} min", flush=True)

    flush()
    return shard_files, n_total_emit, n_dropped, time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     required=True)
    parser.add_argument('--out_dir',      required=True)
    parser.add_argument('--rescut',       type=float, default=3.5)
    parser.add_argument('--max_length',   type=int,   default=10000)
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--shard_size',   type=int,   default=50000,
                        help='entries per shard file')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    params = {
        'DIR'    : args.data_dir,
        'DATCUT' : '2030-Jan-01',
        'RESCUT' : args.rescut,
        'HOMO'   : 0.70,
        'LIST'   : f'{args.data_dir}/list.csv',
        'VAL'    : f'{args.data_dir}/valid_clusters.txt',
        'TEST'   : f'{args.data_dir}/test_clusters.txt',
    }

    print("Building cluster splits...", flush=True)
    t0 = time.time()
    train, valid, test = build_training_clusters(params, False)
    print(f"  cluster build: {time.time()-t0:.1f}s", flush=True)
    print(f"  train: {len(train)} clusters, "
          f"{sum(len(v) for v in train.values())} chains", flush=True)
    print(f"  valid: {len(valid)} clusters, "
          f"{sum(len(v) for v in valid.values())} chains", flush=True)

    LOAD_PARAM = {
        'batch_size': 1,
        'shuffle': False,
        'pin_memory': False,
        'num_workers': args.num_workers,
        'persistent_workers': args.num_workers > 0,
    }

    manifest = {
        'data_dir':  args.data_dir,
        'rescut':    args.rescut,
        'max_length': args.max_length,
        'shard_size': args.shard_size,
        'splits': {},
    }

    for split_name, split in [('train', train), ('valid', valid)]:
        print(f"\nProcessing {split_name}...", flush=True)
        dataset = AllChainsDataset(split, loader_pdb, params)
        n_total = len(dataset)
        print(f"  total chain entries: {n_total}", flush=True)

        loader = torch.utils.data.DataLoader(
            dataset, worker_init_fn=worker_init_fn, **LOAD_PARAM)

        shard_files, kept, dropped, elapsed = stream_to_shards(
            loader, args.out_dir, split_name, args.max_length, args.shard_size)

        manifest['splits'][split_name] = {
            'shards':  shard_files,
            'kept':    kept,
            'dropped': dropped,
            'seconds': elapsed,
        }
        print(f"  {split_name} done: {kept} kept, {dropped} dropped, "
              f"{elapsed/60:.1f} min, {len(shard_files)} shards",
              flush=True)

    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}", flush=True)

    # Symlink cluster files so the cache directory is self-contained.
    for fname in ['list.csv', 'valid_clusters.txt', 'test_clusters.txt']:
        src = f'{args.data_dir}/{fname}'
        dst = f'{args.out_dir}/{fname}'
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    print("\nDone.", flush=True)
    print(f"Output dir: {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
