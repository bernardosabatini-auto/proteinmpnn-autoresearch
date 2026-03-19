"""
Run once to pre-process the full ProteinMPNN dataset.
Converts ~130k individual .pt files + assembly logic into
two fast-loading files (train.pt, valid.pt).

Usage:
  python training/preprocess.py \
    --data_dir ~/pdb_data/pdb_2021aug02 \
    --out_dir ~/pdb_data/processed \
    --rescut 2.0 \
    --max_length 10000

After running, set --path_for_training_data ~/pdb_data/processed
"""
import argparse, os, sys, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, worker_init_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--rescut',     type=float, default=2.0)
    parser.add_argument('--max_length', type=int,   default=10000)
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

    print("Building cluster splits...")
    train, valid, test = build_training_clusters(params, False)
    print(f"  Train: {len(train)} clusters, Valid: {len(valid)}, Test: {len(test)}")

    LOAD_PARAM = {'batch_size': 1, 'shuffle': True,
                  'pin_memory': False, 'num_workers': 0}

    for split_name, split in [('train', train), ('valid', valid)]:
        print(f"\nProcessing {split_name}...")
        t0 = time.time()

        dataset = PDB_dataset(list(split.keys()), loader_pdb, split, params)
        loader  = torch.utils.data.DataLoader(
                    dataset, worker_init_fn=worker_init_fn, **LOAD_PARAM)

        # num_units=999999999 means load everything
        pdb_dict_list = get_pdbs(loader, repeat=1,
                                  max_length=args.max_length,
                                  num_units=999999999)

        out_path = f'{args.out_dir}/{split_name}.pt'
        torch.save(pdb_dict_list, out_path)
        elapsed = time.time() - t0
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"  Saved {len(pdb_dict_list)} structures → {out_path}")
        print(f"  Size: {size_gb:.1f} GB, Time: {elapsed/60:.1f} min")

    print("\nDone.")
    print(f"Output dir: {args.out_dir}")
    print("Next: update training.py to load from processed files.")

if __name__ == '__main__':
    main()
