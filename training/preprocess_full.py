"""
Preprocess pdb_2021aug02 into a fully-enumerated cache.

Unlike preprocess.py, which uses PDB_dataset's random per-cluster sampling
(emitting ONE chain per cluster), this script enumerates EVERY (cluster,
chain) pair across train/valid clusters and emits one cache entry per
chain. The result is a much larger cache (~hundreds of thousands of
chains) suitable for training over the full pdb_2021aug02 dataset
without runtime stream-mode FS contention.

Usage:
    python training/preprocess_full.py \\
        --data_dir /n/netscratch/.../data/pdb_2021aug02 \\
        --out_dir  /n/netscratch/.../data/pdb_2021aug02_full_cache \\
        --rescut   3.5 \\
        --max_length 10000 \\
        --num_workers 16
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import build_training_clusters, loader_pdb, get_pdbs, worker_init_fn


class AllChainsDataset(torch.utils.data.Dataset):
    """Yields one assembly-format dict per (cluster, chain) pair.

    Each entry in the underlying cluster dict has shape [pdbid_chain, hash].
    We flatten across all clusters so a DataLoader iterates every chain
    exactly once (no random sampling like PDB_dataset.__getitem__).
    """

    def __init__(self, cluster_dict, loader, params):
        self.items = []
        for items in cluster_dict.values():
            self.items.extend(items)
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # loader_pdb may raise on missing/corrupt entries; the caller drops
        # those when building the cache.
        return self.loader(self.items[idx], self.params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     required=True)
    parser.add_argument('--out_dir',      required=True)
    parser.add_argument('--rescut',       type=float, default=3.5)
    parser.add_argument('--max_length',   type=int,   default=10000)
    parser.add_argument('--num_workers',  type=int,   default=16)
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

    for split_name, split in [('train', train), ('valid', valid)]:
        print(f"\nProcessing {split_name}...", flush=True)
        t0 = time.time()

        dataset = AllChainsDataset(split, loader_pdb, params)
        n_total = len(dataset)
        print(f"  total chain entries: {n_total}", flush=True)

        loader = torch.utils.data.DataLoader(
            dataset, worker_init_fn=worker_init_fn, **LOAD_PARAM)

        # get_pdbs handles the assembly→per-chain dict conversion. With
        # num_units=999999999 it iterates the full loader.
        pdb_dict_list = get_pdbs(
            loader, repeat=1,
            max_length=args.max_length,
            num_units=999999999,
        )

        out_path = f'{args.out_dir}/{split_name}.pt'
        torch.save(pdb_dict_list, out_path)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"  saved {len(pdb_dict_list)} chains → {out_path}", flush=True)
        print(f"  size: {size_gb:.1f} GB, time: {elapsed/60:.1f} min", flush=True)

    # Copy cluster files so the cache directory is self-contained for
    # training.utils.build_training_clusters (only needed if a downstream
    # tool falls back to stream mode against the same dir).
    for fname in ['list.csv', 'valid_clusters.txt', 'test_clusters.txt']:
        src = f'{args.data_dir}/{fname}'
        dst = f'{args.out_dir}/{fname}'
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    print("\nDone.", flush=True)
    print(f"Output dir: {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
