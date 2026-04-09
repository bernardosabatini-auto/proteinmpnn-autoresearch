import argparse
import os.path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from datetime import timedelta
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils_attn import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    # ---------------------------------------------------------------------
    # Distributed setup
    #   Two launch modes are supported:
    #     (a) torchrun (single- or multi-node):
    #           RANK, LOCAL_RANK, WORLD_SIZE set automatically
    #     (b) srun (multi-node, --ntasks-per-node=N_GPUS):
    #           SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS set automatically;
    #           caller must export MASTER_ADDR + MASTER_PORT in the shell
    #   In both cases the backend is NCCL with the default env:// init.
    # ---------------------------------------------------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        # Export so any downstream library expecting torchrun env vars works.
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
    else:
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        # init_method defaults to env://, which uses MASTER_ADDR/MASTER_PORT
        # plus RANK/WORLD_SIZE from os.environ.
        # Long timeout: stream-mode pre-featurization can take 30+ minutes on
        # the first epoch, and ranks drift apart while loading per-chain .pt
        # files from a parallel filesystem. The default 10-min watchdog
        # times out the first all_reduce(MIN) sync. 1h is safe headroom.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(local_rank)
    is_main = (rank == 0)

    scaler = torch.cuda.amp.GradScaler()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if is_main:
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        for subfolder in ['model_weights']:
            if not os.path.exists(base_folder + subfolder):
                os.makedirs(base_folder + subfolder)
    if world_size > 1:
        dist.barrier()

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if is_main and not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv",
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000


    model = ProteinMPNN(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise)
    model.to(device)


    if PATH:
        checkpoint = torch.load(PATH, map_location=device)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # DDP wrap
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank])
        raw_model = ddp_model.module
    else:
        ddp_model = model
        raw_model = model


    # -------------------------------------------------------------------
    # Data loading: two modes
    #   1) cache mode  — pre-processed train.pt/valid.pt (small subsets)
    #   2) stream mode — original ProteinMPNN per-chain .pt files under
    #                    {data_path}/pdb/, sampled via PDB_dataset
    # -------------------------------------------------------------------
    import os as _os
    import random as _random
    _processed_dir = args.path_for_training_data
    _train_cache = _os.path.join(_processed_dir, 'train.pt')
    _valid_cache = _os.path.join(_processed_dir, 'valid.pt')
    use_cache = _os.path.exists(_train_cache) and _os.path.exists(_valid_cache)

    if use_cache:
        if is_main:
            print(f"Loading pre-processed data from {_processed_dir}...", flush=True)
        _all_train = torch.load(_train_cache)
        _all_valid = torch.load(_valid_cache)
        if is_main:
            print(f"  Train: {len(_all_train)} structures, Valid: {len(_all_valid)}", flush=True)

        def _sample_global(seed_offset):
            # Deterministic global sample so all ranks see the same draw,
            # then shard by rank stride.
            _random.seed(42 + seed_offset)
            n_train = min(args.num_examples_per_epoch, len(_all_train))
            n_valid = min(max(args.num_examples_per_epoch // 10, 1), len(_all_valid))
            train_sample = _random.sample(_all_train, n_train)
            valid_sample = _random.sample(_all_valid, n_valid)
            if world_size > 1:
                train_sample = train_sample[rank::world_size]
                valid_sample = valid_sample[rank::world_size]
            return train_sample, valid_sample

        pdb_dict_train, pdb_dict_valid = _sample_global(0)
        dataset_train = StructureDataset(pdb_dict_train, truncate=None,
                                         max_length=args.max_protein_length, verbose=False)
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None,
                                         max_length=args.max_protein_length, verbose=False)
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    else:
        # Stream mode: read raw per-chain .pt files via PDB_dataset.
        # Each rank shards the cluster IDs so workers see disjoint clusters.
        if is_main:
            print(f"Stream mode: building cluster splits from {data_path}...", flush=True)
        train_clusters, valid_clusters, test_clusters = build_training_clusters(params, args.debug)
        if is_main:
            print(f"  Train clusters: {len(train_clusters)}  Valid clusters: {len(valid_clusters)}",
                  flush=True)

        # Shard cluster IDs by rank
        train_keys_all = sorted(train_clusters.keys())
        valid_keys_all = sorted(valid_clusters.keys())
        if world_size > 1:
            train_keys = train_keys_all[rank::world_size]
            valid_keys = valid_keys_all[rank::world_size]
        else:
            train_keys = train_keys_all
            valid_keys = valid_keys_all

        train_set = PDB_dataset(train_keys, loader_pdb, train_clusters, params)
        valid_set = PDB_dataset(valid_keys, loader_pdb, valid_clusters, params)

        LOAD_PARAM = {'batch_size': 1, 'shuffle': True,
                      'pin_memory': False, 'num_workers': 4}
        train_loader_raw = torch.utils.data.DataLoader(
            train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
        valid_loader_raw = torch.utils.data.DataLoader(
            valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

        # Per-rank target so the global epoch sees ~num_examples_per_epoch
        per_rank_train = max(1, args.num_examples_per_epoch // max(world_size, 1))
        per_rank_valid = max(1, (args.num_examples_per_epoch // 10) // max(world_size, 1))

        def _stream_sample():
            pdb_dict_train = get_pdbs(train_loader_raw, repeat=1,
                                      max_length=args.max_protein_length,
                                      num_units=per_rank_train)
            pdb_dict_valid = get_pdbs(valid_loader_raw, repeat=1,
                                      max_length=args.max_protein_length,
                                      num_units=per_rank_valid)
            return pdb_dict_train, pdb_dict_valid

        pdb_dict_train, pdb_dict_valid = _stream_sample()
        dataset_train = StructureDataset(pdb_dict_train, truncate=None,
                                         max_length=args.max_protein_length, verbose=False)
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None,
                                         max_length=args.max_protein_length, verbose=False)
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

    # -------------------------------------------------------------------
    # DDP helper: cap each rank's per-epoch step count to the global min
    # so all ranks call forward+backward the same number of times.
    # -------------------------------------------------------------------
    def _sync_min_steps(local_count):
        if world_size <= 1:
            return local_count
        t = torch.tensor([local_count], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())

    def _all_reduce_sum(values):
        if world_size <= 1:
            return values
        t = torch.tensor(values, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return [float(x.item()) for x in t]

    reload_c = 0
    for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            ddp_model.train()
            train_sum, train_weights = 0., 0.
            # Per-protein accuracy: sum of per-protein recovery rates and a count
            # of contributing proteins. valid_acc / train_acc are reported as the
            # mean per-protein recovery (each protein weighted equally regardless
            # of length), the field-standard sequence-recovery metric.
            train_acc_pp_sum = 0.
            train_protein_count = 0
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    if use_cache:
                        pdb_dict_train, pdb_dict_valid = _sample_global(reload_c)
                    else:
                        pdb_dict_train, pdb_dict_valid = _stream_sample()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None,
                                                     max_length=args.max_protein_length, verbose=False)
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None,
                                                     max_length=args.max_protein_length, verbose=False)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                reload_c += 1

            # Pre-featurize so we can sync the per-epoch step count across ranks.
            # Print periodic progress so a stalled rank is observable.
            _t_pre = time.time()
            cached_train = []
            for _bi, batch in enumerate(loader_train):
                try:
                    cached_train.append(featurize(batch, device))
                except Exception:
                    continue
                if is_main and (_bi + 1) % 50 == 0:
                    print(f"[rank0] pre-featurize train: {_bi+1} batches, "
                          f"{time.time()-_t_pre:.1f}s elapsed", flush=True)
            if is_main:
                print(f"[rank0] pre-featurize train DONE: {len(cached_train)} batches, "
                      f"{time.time()-_t_pre:.1f}s", flush=True)
            n_train_global = _sync_min_steps(len(cached_train))
            cached_train = cached_train[:n_train_global]

            cached_valid = []
            for batch in loader_valid:
                try:
                    cached_valid.append(featurize(batch, device))
                except Exception:
                    continue
            if is_main:
                print(f"[rank0] pre-featurize valid DONE: {len(cached_valid)} batches", flush=True)
            n_valid_global = _sync_min_steps(len(cached_valid))
            cached_valid = cached_valid[:n_valid_global]

            for X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all in cached_train:
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = ddp_model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                    scaler.scale(loss_av_smoothed).backward()

                    if args.gradient_norm > 0.0:
                        scaler.unscale_(optimizer.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = ddp_model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.gradient_norm)

                    optimizer.step()

                with torch.no_grad():
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                    train_sum += float(torch.sum(loss * mask_for_loss).item())
                    train_weights += float(torch.sum(mask_for_loss).item())

                    pp_correct = (true_false * mask_for_loss).sum(dim=1)  # [B]
                    pp_total = mask_for_loss.sum(dim=1)                   # [B]
                    valid_proteins = pp_total > 0
                    if valid_proteins.any():
                        per_protein_acc = pp_correct[valid_proteins] / pp_total[valid_proteins]
                        train_acc_pp_sum += float(per_protein_acc.sum().item())
                        train_protein_count += int(valid_proteins.sum().item())

                total_step += 1

            raw_model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc_pp_sum = 0.
                validation_protein_count = 0
                for X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all in cached_valid:
                    log_probs = raw_model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                    validation_sum += float(torch.sum(loss * mask_for_loss).item())
                    validation_weights += float(torch.sum(mask_for_loss).item())

                    pp_correct = (true_false * mask_for_loss).sum(dim=1)  # [B]
                    pp_total = mask_for_loss.sum(dim=1)                   # [B]
                    valid_proteins = pp_total > 0
                    if valid_proteins.any():
                        per_protein_acc = pp_correct[valid_proteins] / pp_total[valid_proteins]
                        validation_acc_pp_sum += float(per_protein_acc.sum().item())
                        validation_protein_count += int(valid_proteins.sum().item())

            # Aggregate metrics across ranks
            (train_sum, train_weights, train_acc_pp_sum, train_protein_count_f,
             validation_sum, validation_weights,
             validation_acc_pp_sum, validation_protein_count_f) = _all_reduce_sum([
                train_sum, train_weights, train_acc_pp_sum, float(train_protein_count),
                validation_sum, validation_weights,
                validation_acc_pp_sum, float(validation_protein_count),
            ])
            train_protein_count = int(train_protein_count_f)
            validation_protein_count = int(validation_protein_count_f)

            train_loss = train_sum / max(train_weights, 1e-9)
            train_accuracy = train_acc_pp_sum / max(train_protein_count, 1)
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / max(validation_weights, 1e-9)
            validation_accuracy = validation_acc_pp_sum / max(validation_protein_count, 1)
            validation_perplexity = np.exp(validation_loss)

            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1)
            if is_main:
                with open(logfile, 'a') as f:
                    f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
                print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}', flush=True)

                checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
                torch.save({
                            'epoch': e+1,
                            'step': total_step,
                            'num_edges' : args.num_neighbors,
                            'noise_level': args.backbone_noise,
                            'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.optimizer.state_dict(),
                            }, checkpoint_filename_last)

                if (e+1) % args.save_model_every_n_epochs == 0:
                    checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                    torch.save({
                            'epoch': e+1,
                            'step': total_step,
                            'num_edges' : args.num_neighbors,
                            'noise_level': args.backbone_noise,
                            'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.optimizer.state_dict(),
                            }, checkpoint_filename)

            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")

    args = argparser.parse_args()
    main(args)
