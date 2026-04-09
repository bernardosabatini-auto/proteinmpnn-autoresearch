#!/usr/bin/env python3
"""
submit_experiment.py — Claude Code calls this to run one ProteinMPNN experiment.

Usage:
    python slurm/submit_experiment.py \
        --exp_name baseline_k48_h128 \
        --num_neighbors 48 \
        --hidden_dim 128 \
        --num_encoder_layers 3 \
        --num_decoder_layers 3 \
        --num_epochs 150

Output:
    job_id: <SLURM job ID>
    status: COMPLETED | FAILED | TIMEOUT
    best_valid_acc: <float>
    valid_acc_epoch1: <float>
    valid_acc_epoch10: <float>
    epochs_completed: <int>
    log_path: <path>
"""

import argparse
import os
import subprocess
import sys
import time
import re
from pathlib import Path

BASE          = "/n/holylabs/bsabatini_lab/Users/bsabatini/proteinmpnn"
OUTPUTS_ROOT  = "/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/outputs"
SCRIPT_PATH   = Path(__file__).parent / "experiment.sh"
POLL_INTERVAL = 60


def submit_job(args) -> str:
    env = os.environ.copy()
    env.update({
        "EXP_NAME":            args.exp_name,
        "TRAINING_SCRIPT":     args.training_script,
        "NUM_EPOCHS":          str(args.num_epochs),
        "NUM_EXAMPLES":        str(args.num_examples),
        "BATCH_SIZE":          str(args.batch_size),
        "HIDDEN_DIM":          str(args.hidden_dim),
        "NUM_ENCODER":         str(args.num_encoder_layers),
        "NUM_DECODER":         str(args.num_decoder_layers),
        "NUM_NEIGHBORS":       str(args.num_neighbors),
        "BACKBONE_NOISE":      str(args.backbone_noise),
        "MIXED_PRECISION":     str(args.mixed_precision),
        "DROPOUT":             str(args.dropout),
        "SAVE_EVERY":          str(args.save_every),
        "RELOAD_EVERY":        str(args.reload_every),
        "DATA_DIR":            args.data_dir,
        "PREVIOUS_CHECKPOINT": args.previous_checkpoint,
        "MAX_PROTEIN_LENGTH":  str(args.max_protein_length),
    })

    sbatch_cmd = ["sbatch", "--parsable"]
    if args.time:
        sbatch_cmd += ["--time", args.time]
    sbatch_cmd.append(str(SCRIPT_PATH))
    result = subprocess.run(
        sbatch_cmd, env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip().split(";")[0]
    print(f"job_id: {job_id}")
    return job_id


def poll_job(job_id: str) -> str:
    print(f"Polling job {job_id} every {POLL_INTERVAL}s...", file=sys.stderr)
    while True:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True
        )
        state = result.stdout.strip()
        if not state:
            sacct = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State", "--parsable2"],
                capture_output=True, text=True
            )
            states = [s.strip() for s in sacct.stdout.strip().split("\n") if s.strip()]
            return states[0].split("+")[0] if states else "UNKNOWN"
        if state in ("RUNNING", "PENDING", "CONFIGURING"):
            print(f"  [{time.strftime('%H:%M:%S')}] {state}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)
        else:
            return state


def read_results(exp_name: str) -> dict:
    log_path = Path(OUTPUTS_ROOT) / exp_name / "log.txt"
    if not log_path.exists():
        return {"error": f"log.txt not found at {log_path}"}

    results = {
        "log_path": str(log_path),
        "best_valid_acc": None,
        "epochs_completed": 0,
        "valid_acc_epoch1": None,
        "valid_acc_epoch10": None,
        "learning_curve": []
    }

    with open(log_path) as f:
        for line in f:
            m = re.search(r"epoch:\s*(\d+).*?step:\s*(\d+).*?valid_acc:\s*([\d.]+)", line)
            if not m:
                continue
            epoch, step, vacc = int(m.group(1)), int(m.group(2)), float(m.group(3))
            results["epochs_completed"] = epoch
            results["learning_curve"].append((epoch, step, vacc))
            if results["best_valid_acc"] is None or vacc > results["best_valid_acc"]:
                results["best_valid_acc"] = vacc
            if epoch == 1:
                results["valid_acc_epoch1"] = vacc
            if epoch == 10:
                results["valid_acc_epoch10"] = vacc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",           required=True)
    parser.add_argument("--num_neighbors",       type=int,   default=48)
    parser.add_argument("--hidden_dim",          type=int,   default=128)
    parser.add_argument("--num_encoder_layers",  type=int,   default=3)
    parser.add_argument("--num_decoder_layers",  type=int,   default=3)
    parser.add_argument("--num_epochs",          type=int,   default=150)
    parser.add_argument("--num_examples",        type=int,   default=50000)
    parser.add_argument("--batch_size",          type=int,   default=10000)
    parser.add_argument("--backbone_noise",      type=float, default=0.2)
    parser.add_argument("--mixed_precision",     default="True")
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--save_every",          type=int,   default=10)
    parser.add_argument("--reload_every",        type=int,   default=200)
    parser.add_argument("--training_script", default="training.py")
    parser.add_argument("--data_dir", default="/n/netscratch/bsabatini_lab/Users/bsabatini/proteinmpnn/data/processed_3p5",
                        help="path to pre-processed training data directory")
    parser.add_argument("--previous_checkpoint", default="",
                        help="path to .pt checkpoint to fine-tune from (empty = train from scratch)")
    parser.add_argument("--time", default="",
                        help="SLURM time limit override, e.g. 24:00:00 (empty = use #SBATCH default)")
    parser.add_argument("--max_protein_length", type=int, default=10000,
                        help="Filter out proteins longer than this (must be ≥ batch_size)")
    parser.add_argument("--no-wait",             action="store_true")
    args = parser.parse_args()

    job_id = submit_job(args)

    if args.no_wait:
        return

    final_state = poll_job(job_id)
    print(f"status: {final_state}")

    results = read_results(args.exp_name)
    if "error" in results:
        print(f"error: {results['error']}")
        sys.exit(1)

    print(f"best_valid_acc: {results['best_valid_acc']}")
    print(f"epochs_completed: {results['epochs_completed']}")
    print(f"valid_acc_epoch1: {results['valid_acc_epoch1']}")
    print(f"valid_acc_epoch10: {results['valid_acc_epoch10']}")
    print(f"log_path: {results['log_path']}")

    curve = results["learning_curve"]
    if curve:
        print("\nLearning curve (last 5 epochs):")
        for ep, step, vacc in curve[-5:]:
            print(f"  epoch {ep:3d}  step {step:6d}  valid_acc {vacc:.4f}")


if __name__ == "__main__":
    main()
