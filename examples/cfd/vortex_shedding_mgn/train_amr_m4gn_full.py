"""
train_amr_m4gn_full.py — multi-case batched trainer for AMRM4GN (M5 step 3)
===========================================================================
Extends the M4 overfit script (train_amr_m4gn.py, kept as-is) into a proper
training entry: multiple cases, PyG batched DataLoader (batch_size>1),
per-channel NMSE, exponential LR decay, and checkpointing. Single-machine /
single-GPU (no DDP/apex/wandb — those are added when real multi-GPU runs are
needed; the model/data pieces here are DDP-ready).

Key batching detail: the DataLoader collates B graphs into one PyG Batch
(carrying `ptr` and per-graph `gidx`); we fetch the matching list of per-case
caches via `gidx` and pass it to `AMRM4GN.forward(batch, caches)` (M5 step 2).

Prerequisite: per-case partition caches must exist (run preprocess first):
    python preprocess_partitions.py --data_dir <...> --split train \
        --num_cases <N> --out_dir ./amr_cache

Usage (small smoke run):
    python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --batch_size 2 \
        --epochs 50 --omega_thresh 30
"""

from __future__ import annotations

import os
import argparse

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import per_channel_nmse, move_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True,
                   help="dir with partition_cache_train_{gidx}.pt (preprocess first)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_cases", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.9999991)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=30.0,
                   help="fixed vorticity threshold for routing (None ranges -> "
                        "use --sample_thresh for train-time sampling)")
    p.add_argument("--sample_thresh", action="store_true", default=False,
                   help="sample thresholds per graph (Design Doc 4.7) instead of "
                        "the fixed omega threshold")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints_amr")
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ds = VortexSheddingDatasetAMR(
        name="amr_train", data_dir=args.data_dir, split=args.split,
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=args.noise_std, cache_dir=args.cache_dir,
    )
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # preload all per-case caches onto the device
    caches = {g: move_cache(ds.get_cache(g), device) for g in range(args.num_cases)}

    vel_mean = ds.node_stats["velocity_mean"]
    vel_std = ds.node_stats["velocity_std"]
    model = AMRM4GN(
        in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
        processor_size=args.processor_size, vel_mean=vel_mean, vel_std=vel_std,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

    thr = None if args.sample_thresh else {
        "G": float("inf"), "omega": args.omega_thresh,
        "M": float("inf"), "S": float("inf")}

    model.train()
    print(f"[train] {args.num_cases} cases x {args.num_steps-1} steps, "
          f"batch_size={args.batch_size}, {args.epochs} epochs, device={device}")
    for epoch in range(args.epochs):
        epoch_loss, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            cs = [caches[int(g)] for g in batch.gidx]
            opt.zero_grad()
            pred = model(batch, cs, thresholds=thr)
            loss = per_channel_nmse(pred, batch.y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        sched.step()
        epoch_loss /= max(nb, 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  NMSE {epoch_loss:.4e}  lr {sched.get_last_lr()[0]:.2e}")
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.ckpt_dir, f"amr_m4gn_epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict()}, path)
    print(f"done. checkpoints in {args.ckpt_dir}")


if __name__ == "__main__":
    main()
